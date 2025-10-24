#!/usr/bin/env python3
"""
sender_bursts.py

Send numpy-array chunks to receiver in randomized bursts/surges.

Usage:
  python3 sender_bursts.py [--host HOST] [--port PORT] [--total N] [--chunk K]
                          [--bursts B] [--min-burst S] [--max-burst M]
                          [--min-interval s] [--max-interval s] [--jitter s]

Example:

python3 request_generator.py --total 120 --chunk 10 --bursts 12 --min-burst 5 --max-burst 8 --min-interval 0.1 --max-interval 0.6 --jitter 0.01

Example:
  # total 100 requests, chunk_size=10 => 10 chunks. 8 bursts random (each burst size 1..4)
  python3 sender_bursts.py --total 100 --chunk 10 --bursts 8 --min-burst 1 --max-burst 4

Important: run receiver first (single-threaded server with queue).
"""
import argparse
import threading
import struct
import socket
import io
import time
import random
import math
import numpy as np
from user import Users  # your user.py must be in same folder

# wire helpers
def send_length_prefixed(conn, data_bytes):
    conn.sendall(struct.pack(">Q", len(data_bytes)) + data_bytes)

def chunk_array(arr, chunk_size):
    return [arr[i:i+chunk_size] for i in range(0, len(arr), chunk_size)]

def make_payload_from_chunk(chunk: np.ndarray):
    buf = io.BytesIO()
    np.save(buf, chunk, allow_pickle=True)
    return buf.getvalue()

def sender_worker(host, port, payload, idx, wait_for_start_event):
    """
    Each worker optionally waits on an Event (barrier). When released,
    it connects, sends payload, optionally waits for 1-byte ACK and closes.
    """
    wait_for_start_event.wait()
    try:
        with socket.create_connection((host, port), timeout=5.0) as s:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            send_length_prefixed(s, payload)
            # try to read 1-byte ACK (non-blocking-ish with timeout)
            try:
                s.settimeout(2.0)
                ack = s.recv(1)
                # ack may be empty if server closed immediately
            except Exception:
                ack = None
    except Exception as e:
        print(f"[worker {idx}] send failed: {e}")

def run_bursts(host, port, users, chunk_size, bursts,
               min_burst, max_burst, min_interval, max_interval, jitter):
    # build chunks (numpy arrays)
    arr = np.array(users.requests, dtype=object)
    chunks = chunk_array(arr, chunk_size)
    total_chunks = len(chunks)
    print(f"[SENDER] Built {len(arr)} requests -> {total_chunks} chunks (chunk_size={chunk_size})")

    # pointer into chunk list
    ptr = 0
    burst_no = 0
    while burst_no < bursts and ptr < total_chunks:
        burst_no += 1
        # decide burst size (how many simultaneous chunk sends in this burst)
        burst_size = random.randint(min_burst, max_burst)
        # limit by remaining chunks
        burst_size = min(burst_size, total_chunks - ptr)
        # create a barrier event that all workers will wait on
        start_event = threading.Event()
        workers = []
        print(f"\n[SENDER] Burst {burst_no}/{bursts}: launching {burst_size} workers (chunks idx {ptr}..{ptr+burst_size-1})")

        # start workers (they wait on start_event)
        for i in range(burst_size):
            chunk = chunks[ptr + i]
            payload = make_payload_from_chunk(chunk)
            t = threading.Thread(target=sender_worker, args=(host, port, payload, ptr + i, start_event))
            t.daemon = True
            t.start()
            workers.append(t)

        # short random jitter before releasing them to better simulate network jitter
        pre_sleep = random.uniform(0.0, jitter)
        if pre_sleep > 0:
            time.sleep(pre_sleep)

        # release all workers in the burst at once
        t0 = time.time()
        start_event.set()

        # Optionally, wait a small amount for workers to finish sending (not required)
        for t in workers:
            t.join(timeout=5.0)

        t1 = time.time()
        print(f"[SENDER] Burst {burst_no} dispatched in {(t1 - t0)*1000:.1f} ms")

        ptr += burst_size

        # wait until next burst
        if ptr < total_chunks and burst_no < bursts:
            interval = random.uniform(min_interval, max_interval)
            print(f"[SENDER] Sleeping {interval:.2f}s until next burst")
            time.sleep(interval)

    print(f"\n[SENDER] Done. Dispatched {ptr}/{total_chunks} chunks across {burst_no} bursts.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6000)
    ap.add_argument("--total", type=int, default=50, help="total requests to generate")
    ap.add_argument("--chunk", type=int, default=10, help="requests per chunk")
    ap.add_argument("--bursts", type=int, default=8, help="number of bursts")
    ap.add_argument("--min-burst", type=int, default=1, help="min simultaneous sends in burst")
    ap.add_argument("--max-burst", type=int, default=4, help="max simultaneous sends in burst")
    ap.add_argument("--min-interval", type=float, default=0.2, help="min seconds between bursts")
    ap.add_argument("--max-interval", type=float, default=1.0, help="max seconds between bursts")
    ap.add_argument("--jitter", type=float, default=0.02, help="pre-release jitter (s)")
    ap.add_argument("--detect", default=None, help="detect_csv path (passed to Users)")
    ap.add_argument("--state", default=None, help="server_state_csv path (passed to Users)")
    args = ap.parse_args()

    users = Users(args.total, detect_csv=args.detect, server_state_csv=args.state)
    run_bursts(args.host, args.port, users, args.chunk, args.bursts,
               args.min_burst, args.max_burst, args.min_interval, args.max_interval, args.jitter)


if __name__ == "__main__":
    main()

