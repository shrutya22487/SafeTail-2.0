# request_reciver_with_ack.py
"""
Receiver that accepts numpy array chunks (np.save bytes), inspects them,
saves chunks to disk (pickle) and sends a 1-byte ACK back to the sender.
"""
import socket
import struct
import io
import numpy as np
import threading
import time
import os
import pickle
from pathlib import Path

HOST = "127.0.0.1"
PORT = 6000
OUT_DIR = Path("received_chunks")
OUT_DIR.mkdir(exist_ok=True)

def recv_all(conn, n):
    data = b""
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def handle_client(conn, addr):
    print(f"[+] Connected by {addr}")
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

                # Load numpy array from bytes
                f = io.BytesIO(payload)
                try:
                    arr = np.load(f, allow_pickle=True)
                except Exception as e:
                    print(f"[!] np.load failed from {addr}: {e}")
                    continue

                # arr is an object-dtype 1D array of Request objects
                try:
                    n = arr.shape[0] if hasattr(arr, "shape") else len(arr)
                except Exception:
                    n = None

                # Print a short summary for each Request
                ids = []
                for i, req in enumerate(arr):
                    try:
                        rid = getattr(req, "request_id", None)
                        pid = getattr(req, "process_id", None)
                        msg = getattr(req, "message_size", None)
                        at = getattr(req, "arrival_time", None)
                        ids.append(rid)
                        print(f"  req[{i}] id={rid} pid={pid} msg={msg} arrival={at}")
                    except Exception as e:
                        print(f"  [!] Error reading request at index {i}: {e}")

                print(f"Received chunk from {addr}: len={n}, ids={ids}")

                # Persist chunk to disk (pickle) with timestamped name
                out_name = OUT_DIR / f"chunk_{addr[1]}_{int(time.time())}.pkl"

                try:
                    with open(out_name, "wb") as f_out:
                        pickle.dump(list(arr), f_out, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"Saved chunk to {out_name}")
                except Exception as e:
                    print(f"[!] Failed to save chunk: {e}")

                # Send 1-byte ACK to sender (non-blocking best-effort)
                try:
                    conn.sendall(b"\x01")
                except Exception:
                    # ignore send failures (sender might close immediately)
                    pass

    except Exception as e:
        print(f"[!] Exception handling {addr}: {e}")
    finally:
        print(f"[-] Disconnected {addr}")

def main():
    print(f"Listening on {HOST}:{PORT} for incoming numpy array chunks...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        try:
            while True:
                conn, addr = s.accept()
                threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
        except KeyboardInterrupt:
            print("Shutting down receiver...")

if __name__ == "__main__":
    main()
