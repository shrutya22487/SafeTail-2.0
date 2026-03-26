#!/usr/bin/env python3
import threading
import struct
import socket
import io
import time
import random
import numpy as np
from typing import Callable, Optional
import user
import constants

def _send_length_prefixed(conn, data_bytes: bytes):
    conn.sendall(struct.pack(">Q", len(data_bytes)) + data_bytes)

def _make_payload_from_array(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=True)
    return buf.getvalue()

class SenderBursts:
    def __init__(
        self,
        arr: Optional[np.ndarray] = None,
        host: str = constants.receiver_host,
        port: int = constants.receiver_port,
        total: Optional[int] = None,
        chunk_size: int = 10,
        bursts: int = 8,
        min_burst: int = 1,
        max_burst: int = 4,
        min_interval: float = 0.2,
        max_interval: float = 1.0,
        jitter: float = 0.02,
        worker_timeout: float = 5.0,
        sample_count: int = 0,
        request_factory: Optional[Callable[[int], object]] = None,
    ):
        self.host = host
        self.port = port
        self.chunk_size = chunk_size
        self.bursts = bursts
        self.min_burst = min_burst
        self.max_burst = max_burst
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.jitter = jitter
        self.worker_timeout = worker_timeout
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._stats = {"dispatched_chunks": 0, "failed_chunks": 0}

        if arr is None:
            if sample_count <= 0:
                raise ValueError("Either provide arr or sample_count > 0")
            # default factory that matches your user.Request signature:
            if request_factory is None:
                def default_factory(i):
                    try:
                        return user.Request(request_id=int(i), process_id=int(i))
                    except Exception:
                        return user.Request(int(i), int(i))
                request_factory = default_factory

            generated = [request_factory(i) for i in range(sample_count)]
            arr = np.array(generated, dtype=object)

        if not isinstance(arr, np.ndarray) or arr.dtype != object:
            raise TypeError("arr must be a NumPy array with dtype=object")

        self.arr = arr[:total] if total else arr
        self.total_chunks = int(np.ceil(len(self.arr) / chunk_size))

    def _sender_worker(self, payload: bytes, idx: int):
        if self._stop_event.is_set():
            return False, 0.0, RuntimeError("stopped")
        start = time.time()
        try:
            with socket.create_connection((self.host, self.port), timeout=5.0) as s:
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                _send_length_prefixed(s, payload)
                try:
                    s.settimeout(2.0)
                    _ = s.recv(1)
                except Exception:
                    pass
            duration = time.time() - start
            return True, duration, None
        except Exception as e:
            duration = time.time() - start
            return False, duration, e

    def run(self) -> dict:
        n = len(self.arr)
        print(f"[SENDER] Built NumPy array of {n} requests -> {self.total_chunks} chunks (size={self.chunk_size})")

        ptr = 0
        burst_no = 0

        while burst_no < self.bursts and ptr < self.total_chunks and not self._stop_event.is_set():
            burst_no += 1
            burst_size = random.randint(self.min_burst, self.max_burst)
            burst_size = min(burst_size, self.total_chunks - ptr)

            print(f"\n[SENDER] Burst {burst_no}/{self.bursts}: launching {burst_size} workers (chunks {ptr}..{ptr+burst_size-1})")

            threads = []
            results = [None] * burst_size

            for i in range(burst_size):
                start_idx = (ptr + i) * self.chunk_size
                end_idx = min(start_idx + self.chunk_size, n)
                chunk_arr = self.arr[start_idx:end_idx]
                payload = _make_payload_from_array(chunk_arr)

                def worker_func(i=i, idx_global=ptr + i, payload=payload):
                    success, dur, exc = self._sender_worker(payload, idx_global)
                    results[i] = (success, dur, exc)
                    with self._lock:
                        self._stats["dispatched_chunks"] += 1
                        if not success:
                            self._stats["failed_chunks"] += 1

                t = threading.Thread(target=worker_func, daemon=True)
                t.start()
                threads.append(t)

            time.sleep(random.uniform(0.0, self.jitter))

            for t in threads:
                t.join(timeout=self.worker_timeout)

            ok_count = sum(1 for r in results if r and r[0])
            print(f"[SENDER] Burst {burst_no} done: {ok_count}/{burst_size} succeeded")

            ptr += burst_size

            if ptr < self.total_chunks and burst_no < self.bursts:
                sleep_time = random.uniform(self.min_interval, self.max_interval)
                print(f"[SENDER] Sleeping {sleep_time:.2f}s until next burst")
                time.sleep(sleep_time)

        print(f"\n[SENDER] Finished. Dispatched {self._stats['dispatched_chunks']}/{self.total_chunks} chunks total.")
        return dict(self._stats)

    def stop(self):
        self._stop_event.set()
