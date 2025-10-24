#!/usr/bin/env python3
"""
Single-threaded receiver that:
- Accepts many incoming connections quickly (accept window) to fill app-level pending queue
- Then processes queued connections one-by-one (single-threaded)
- Optionally simulates processing time so queue visibly fills
- Shows live queue visualization

Tune:
  MAX_QUEUE            -> how many accepted sockets to hold
  ACCEPT_WINDOW_SEC    -> how long to try accepting (seconds) before processing
  PROCESS_TIME_PER_CHUNK -> artificial processing time (seconds) to simulate heavy work
"""

import socket
import struct
import io
import numpy as np
import pickle
import time
from pathlib import Path
from collections import deque

OUT_DIR = Path("received_chunks")
OUT_DIR.mkdir(exist_ok=True)

HOST = "127.0.0.1"
PORT = 6000
TCP_BACKLOG = 200

# tuning knobs
MAX_QUEUE = 20               # how many accepted sockets to hold before rejecting
ACCEPT_WINDOW_SEC = 0.15     # accept loop time window (seconds) to gather bursts
PROCESS_TIME_PER_CHUNK = 0.20  # artificial processing time per received chunk (seconds)

def recv_all(conn, n):
    data = b""
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def visualize_queue(queue_len, max_queue):
    filled = int((queue_len / max_queue) * 20)
    empty = 20 - filled
    bar = "#" * filled + "-" * empty
    print(f"[QUEUE] [{bar}] ({queue_len}/{max_queue})")

def process_connection(conn, addr, simulate_processing=True):
    """Process the connection until the client closes. Simulated processing time added."""
    try:
        with conn:
            while True:
                header = recv_all(conn, 8)
                if not header:
                    break
                length = struct.unpack(">Q", header)[0]
                payload = recv_all(conn, length)
                if payload is None:
                    break

                # deserialize numpy array
                f = io.BytesIO(payload)
                try:
                    arr = np.load(f, allow_pickle=True)
                except Exception as e:
                    print(f"[!] np.load failed from {addr}: {e}")
                    break

                # simple summary
                ids = []
                for i, req in enumerate(arr):
                    try:
                        ids.append(getattr(req, "request_id", None))
                    except Exception:
                        ids.append(None)
                print(f"[>] Received chunk from {addr}: len={len(arr)}, ids={ids}")

                # persist chunk
                ts = int(time.time() * 1000)
                filename = OUT_DIR / f"chunk_{addr[0].replace('.', '_')}_{addr[1]}_{ts}.pkl"
                try:
                    with open(filename, "wb") as fo:
                        pickle.dump(list(arr), fo, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"[✓] Saved chunk -> {filename}")
                except Exception as e:
                    print(f"[!] Failed to save chunk: {e}")

                # artificial processing delay to simulate CPU/IO work (so queue can fill)
                if simulate_processing and PROCESS_TIME_PER_CHUNK > 0:
                    time.sleep(PROCESS_TIME_PER_CHUNK)

                # ACK
                try:
                    conn.sendall(b"\x01")
                except Exception:
                    pass

    except Exception as e:
        print(f"[!] Exception while processing {addr}: {e}")
    finally:
        print(f"[-] Finished processing {addr}")

def run_server():
    pending = deque()
    print(f"[SERVER] Listening on {HOST}:{PORT} (tcp_backlog={TCP_BACKLOG})")
    print(f"[CONFIG] MAX_QUEUE={MAX_QUEUE}, ACCEPT_WINDOW_SEC={ACCEPT_WINDOW_SEC}, PROCESS_TIME_PER_CHUNK={PROCESS_TIME_PER_CHUNK}\n")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(TCP_BACKLOG)
        s.settimeout(0.5)  # moderate blocking for accept calls

        try:
            while True:
                # Phase A: Rapid accept window — try to accept repeatedly for a short time
                accept_deadline = time.time() + ACCEPT_WINDOW_SEC
                while time.time() < accept_deadline and len(pending) < MAX_QUEUE:
                    try:
                        # small timeout ensures accept does not block forever and allows loop to check deadline
                        s.settimeout(accept_deadline - time.time() if (accept_deadline - time.time()) > 0.01 else 0.01)
                        conn, addr = s.accept()
                        pending.append((conn, addr))
                        print(f"[+] Accepted and queued {addr} (pending={len(pending)})")
                    except socket.timeout:
                        # no incoming connection at the moment — continue to check deadline
                        continue
                    except Exception as e:
                        print(f"[!] Accept error: {e}")
                        continue

                # If pending queue is full, reject extra connections immediately until we process some
                # (the kernel backlog will still hold SYNs for a while)
                # Show queue visualization
                visualize_queue(len(pending), MAX_QUEUE)

                # Phase B: Process queued connections one by one (single-threaded)
                if pending:
                    conn, addr = pending.popleft()
                    visualize_queue(len(pending), MAX_QUEUE)
                    print(f"[>] Processing {addr}")
                    process_connection(conn, addr, simulate_processing=True)
                    # after processing, loop back to accept window

                # If no pending connections, block on accept to avoid busy loop
                if not pending:
                    try:
                        s.settimeout(None)  # block until a new connection arrives
                        conn, addr = s.accept()
                        # immediately enqueue (we accepted outside accept-window because pend empty)
                        if len(pending) < MAX_QUEUE:
                            pending.append((conn, addr))
                            print(f"[+] Accepted and queued {addr} (pending={len(pending)})")
                            visualize_queue(len(pending), MAX_QUEUE)
                        else:
                            # should rarely happen due to immediate process, but handle
                            try:
                                conn.sendall(b"BUSY")
                            except Exception:
                                pass
                            conn.close()
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        # some other socket error — print and continue
                        print(f"[!] Accept error while idle: {e}")
                        continue

        except KeyboardInterrupt:
            print("\n[SERVER] Shutting down (KeyboardInterrupt).")

if __name__ == "__main__":
    run_server()
