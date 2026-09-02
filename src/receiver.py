import io
import socket
import struct
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

import constants

OUT_DIR = Path("received_chunks")
OUT_DIR.mkdir(exist_ok=True)


class Receiver:
    def __init__(
            self,
            host: str = constants.receiver_host,
            port: int = constants.receiver_port,
            tcp_backlog: int = 200,
            max_queue: int = 20,
            accept_window_sec: float = 0.15,
            process_time_per_chunk: float = 0.20,
            persist_chunks: bool = True,
            controller=None,
    ):
        # [SAFETAIL][RECEIVER][FIX][D-31] The wire payload is pickled Request
        # objects loaded with np.load(allow_pickle=True) -> arbitrary code
        # execution if this port is ever reachable off-host. Hard-refuse any
        # non-loopback bind, and cap the payload size, so the unpickle can only
        # ever process data this machine sent to itself.
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise ValueError(
                f"[SAFETAIL][RECEIVER][D-31] refusing to bind {host!r}: the receiver "
                f"unpickles its payloads and must stay loopback-only. Set "
                f"constants.receiver_host = '127.0.0.1'."
            )
        self.MAX_PAYLOAD_BYTES = 8 * 1024 * 1024  # 8 MiB: a chunk of 5 Requests is ~few KB

        self.host = host
        self.port = port
        self.tcp_backlog = tcp_backlog
        self.MAX_QUEUE = max_queue
        self.ACCEPT_WINDOW_SEC = accept_window_sec
        self.PROCESS_TIME_PER_CHUNK = process_time_per_chunk
        self.PERSIST_CHUNKS = persist_chunks

        self.controller = controller  # backlink to controller
        if controller is not None:
            controller.receiver_queue = self  # mutual link

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _recv_all(self, conn, n: int) -> Optional[bytes]:
        data = b""
        while len(data) < n:
            packet = conn.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def _visualize_queue(self, queue_len: int):
        maxq = max(1, self.MAX_QUEUE)
        filled = int((queue_len / maxq) * 20)
        empty = 20 - filled
        bar = "#" * filled + "-" * empty
        print(f"[QUEUE] [{bar}] ({queue_len}/{self.MAX_QUEUE})")

    def _extract_request_id(self, req):
        # try direct attr names
        for name in ("request_id", "requestId", "req_id", "rid", "id", "_id", "process_id"):
            try:
                if hasattr(req, name):
                    v = getattr(req, name)
                    if callable(v):
                        try:
                            return v()
                        except Exception:
                            continue
                    return v
            except Exception:
                pass
        # try __dict__
        try:
            d = getattr(req, "__dict__", None)
            if isinstance(d, dict):
                for k in ("request_id", "id", "process_id"):
                    if k in d:
                        return d[k]
                for k, v in d.items():
                    if isinstance(v, (int, str)):
                        return v
        except Exception:
            pass
        # try common getters
        for method in ("get_request_id", "getRequestId", "get_id", "getId", "id"):
            try:
                if hasattr(req, method) and callable(getattr(req, method)):
                    try:
                        return getattr(req, method)()
                    except Exception:
                        pass
            except Exception:
                pass
        # try to_dict/as_dict
        for method in ("as_dict", "to_dict", "toJSON", "to_json"):
            try:
                if hasattr(req, method) and callable(getattr(req, method)):
                    try:
                        d = getattr(req, method)()
                        if isinstance(d, dict):
                            for k in ("request_id", "id", "process_id"):
                                if k in d:
                                    return d[k]
                            for v in d.values():
                                if isinstance(v, (int, str)):
                                    return v
                    except Exception:
                        pass
            except Exception:
                pass
        # dir scan for id-like attribute
        try:
            for attr in dir(req):
                if attr.startswith("_"):
                    continue
                if "id" in attr.lower() or "req" in attr.lower() or "pid" in attr.lower():
                    try:
                        v = getattr(req, attr)
                        if isinstance(v, (int, str)):
                            return v
                    except Exception:
                        pass
        except Exception:
            pass
        return None

    def _process_connection(self, conn, addr, simulate_processing=True):
        try:
            with conn:
                while True:
                    # ---- Receive header ----
                    try:
                        header = self._recv_all(conn, 8)
                    except Exception as e:
                        print(f"[RECEIVER, !] Failed to recv header from {addr}: {e}")
                        break

                    if not header:
                        break

                    try:
                        length = struct.unpack(">Q", header)[0]
                    except struct.error as e:
                        print(f"[RECEIVER, !] Invalid header from {addr}: {e}")
                        break

                    if length <= 0:
                        print(f"[RECEIVER, !] Invalid payload length {length} from {addr}")
                        break

                    if length > self.MAX_PAYLOAD_BYTES:
                        # [SAFETAIL][RECEIVER][D-31] refuse oversized payloads
                        print(f"[SAFETAIL][RECEIVER][D-31] payload {length} B from {addr} "
                              f"exceeds cap {self.MAX_PAYLOAD_BYTES} B; dropping connection")
                        break

                    # ---- Receive payload ----
                    try:
                        payload = self._recv_all(conn, length)
                    except Exception as e:
                        print(f"[RECEIVER, !] Failed to recv payload from {addr}: {e}")
                        break

                    if payload is None:
                        break

                    # ---- Deserialize ----
                    try:
                        f = io.BytesIO(payload)
                        arr = np.load(f, allow_pickle=True)
                    except Exception as e:
                        print(f"[RECEIVER, !] np.load failed from {addr}: {e}")
                        break

                    # ---- Coerce to object ndarray ----
                    if not isinstance(arr, np.ndarray):
                        print(f"[RECEIVER, !] Payload from {addr} is not ndarray; skipping chunk")
                        continue

                    if arr.dtype != object:
                        try:
                            arr = np.asarray(arr, dtype=object)
                        except Exception as e:
                            print(f"[RECEIVER, !] Failed to coerce payload from {addr}: {e}")
                            continue

                    # ---- Debug summary + ID extraction ----
                    ids = []
                    for idx, req in enumerate(arr):
                        try:
                            ids.append(self._extract_request_id(req))
                        except Exception as e:
                            print(
                                f"[RECEIVER, !] Failed to extract request id "
                                f"from {addr} item[{idx}]: {e}"
                            )
                            ids.append(None)

                    print(
                        f"[RECEIVER, >] Received chunk from {addr}: "
                        f"len={len(arr)}, ids={ids}"
                    )

                    # ---- Persist chunk ----
                    if self.PERSIST_CHUNKS:
                        ts = int(time.time() * 1000)
                        filename = OUT_DIR / (
                            f"chunk_{addr[0].replace('.', '_')}_{addr[1]}_{ts}.npy"
                        )
                        try:
                            np.save(filename, arr, allow_pickle=True)
                            print(f"[RECEIVER, ✓] Saved chunk -> {filename}")
                        except Exception as e:
                            print(f"[RECEIVER, !] Failed to save chunk from {addr}: {e}")

                    # ---- Update arrival time when entering queue ----
                    now_ms = time.time() * 1000.0
                    for req in arr:
                        req.arrival_time = now_ms

                    # ---- Dispatch to controller ----
                    try:
                        self.controller.send_to_server(arr)
                    except Exception as e:
                        print(
                            f"[RECEIVER, !] Failed to dispatch chunk from {addr}: {e}"
                        )
                        # Dispatch failure is serious → stop processing this connection
                        break

                    # ---- Optional simulated processing ----
                    # if simulate_processing and self.PROCESS_TIME_PER_CHUNK > 0:
                    #     time.sleep(self.PROCESS_TIME_PER_CHUNK)

                    # ---- ACK ----
                    try:
                        conn.sendall(b"\x01")
                    except Exception:
                        # ACK failure should not crash receiver
                        pass

        except Exception as e:
            print(f"[RECEIVER, !] Fatal exception while processing {addr}: {e}")

        finally:
            print(f"[RECEIVER, -] Finished processing {addr}")

    def run(self):
        pending = deque()
        print(f"[RECEIVER] Listening on {self.host}:{self.port} (tcp_backlog={self.tcp_backlog})")
        print(f"[RECEIVER] MAX_QUEUE={self.MAX_QUEUE}, ACCEPT_WINDOW_SEC={self.ACCEPT_WINDOW_SEC}\n")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            max_retries = 10
            retry_delay = 5  # seconds
            for attempt in range(max_retries):
                try:
                    s.bind((self.host, self.port))
                    break
                except OSError as e:
                    import errno
                    if e.errno == errno.EADDRINUSE:
                        if attempt < max_retries - 1:
                            print(
                                f"[RECEIVER] Port {self.port} in use, retrying in {retry_delay}s... ({attempt + 1}/{max_retries})")
                            time.sleep(retry_delay)
                        else:
                            print(f"[RECEIVER] Port {self.port} still in use after {max_retries} attempts. Giving up.")
                            raise
                    else:
                        raise

            s.listen(self.tcp_backlog)
            s.settimeout(0.5)

            try:
                while not self._stop_event.is_set():
                    accept_deadline = time.time() + self.ACCEPT_WINDOW_SEC
                    while time.time() < accept_deadline and len(
                            pending) < self.MAX_QUEUE and not self._stop_event.is_set():
                        try:
                            remain = accept_deadline - time.time()
                            s.settimeout(remain if remain > 0.01 else 0.01)
                            conn, addr = s.accept()
                            pending.append((conn, addr))
                            print(f"[+] Accepted and queued {addr} (pending={len(pending)})")
                        except socket.timeout:
                            continue
                        except Exception as e:
                            print(f"[!] Accept error: {e}")
                            continue

                    self._visualize_queue(len(pending))

                    if pending:
                        conn, addr = pending.popleft()
                        self._visualize_queue(len(pending))
                        print(f"[>] Processing {addr}")
                        self._process_connection(conn, addr, simulate_processing=True)

                    if not pending and not self._stop_event.is_set():
                        try:
                            s.settimeout(None)
                            conn, addr = s.accept()
                            if len(pending) < self.MAX_QUEUE:
                                pending.append((conn, addr))
                                print(f"[+] Accepted and queued {addr} (pending={len(pending)})")
                                self._visualize_queue(len(pending))
                            else:
                                try:
                                    conn.sendall(b"BUSY")
                                except Exception:
                                    pass
                                conn.close()
                        except KeyboardInterrupt:
                            raise
                        except Exception as e:
                            print(f"[!] Accept error while idle: {e}")
                            continue

            except KeyboardInterrupt:
                print("\n[RECEIVER] Shutting down (KeyboardInterrupt).")
            finally:
                print("[RECEIVER] Closing remaining pending connections.")
                while pending:
                    conn, _ = pending.popleft()
                    try:
                        conn.close()
                    except Exception:
                        pass

    def run_async(self):
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Receiver already running")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        print("[RECEIVER] Stop requested.")
