# sender_threads.py
"""
Build Users.requests, convert to numpy array, split into chunks,
and send each chunk concurrently using threads.
"""
import socket
import struct
import io
import threading
import numpy as np
from user import Users

HOST = "127.0.0.1"
PORT = 6000
NUM_REQUESTS = 50
CHUNK_SIZE = 10

def send_length_prefixed(sock, data_bytes):
    sock.sendall(struct.pack(">Q", len(data_bytes)) + data_bytes)

def chunk_array(arr, chunk_size):
    return [arr[i:i+chunk_size] for i in range(0, len(arr), chunk_size)]

def send_chunk(chunk, idx):
    try:
        with socket.create_connection((HOST, PORT), timeout=5.0) as s:
            buf = io.BytesIO()
            np.save(buf, chunk, allow_pickle=True)
            payload = buf.getvalue()
            send_length_prefixed(s, payload)
        print(f"[Thread {idx}] Sent chunk of shape={chunk.shape}")
    except Exception as e:
        print(f"[Thread {idx}] Error: {e}")

def main():
    users = Users(
    NUM_REQUESTS,
    detect_csv="../data/updated_Detect.csv",
    server_state_csv="../data/server_state.csv",
    )
    req_list = users.requests

    # Convert to numpy array (dtype=object, since Request is a Python object)
    arr = np.array(req_list, dtype=object)
    print(f"Created numpy array of Requests: shape={arr.shape}, dtype={arr.dtype}")

    # Split into chunks
    chunks = chunk_array(arr, CHUNK_SIZE)
    print(f"Total chunks: {len(chunks)}")

    threads = []
    for i, chunk in enumerate(chunks):
        t = threading.Thread(target=send_chunk, args=(chunk, i), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("All chunks sent successfully!")

if __name__ == "__main__":
    main()
