#!/usr/bin/env python3
"""
main.py

Run Receiver in background and SenderBursts in foreground.
This request_factory constructs user.Request objects using the correct signature.
"""

import time
from types import SimpleNamespace

from Receiver import Receiver
from sender_bursts import SenderBursts
import user
from controller import Controller

# --- request factory that uses the actual Request signature ---
def request_factory(i: int):
    """
    Construct a user.Request with required args (request_id, process_id)
    and a couple of safe optional values.
    """
    # minimal required args: request_id, process_id
    # other optional fields we set to sane defaults so Receiver sees request_id
    try:
        return user.Request(
            request_id=int(i),
            process_id=int(i),
            message_size=0.0,
            bandwidth=0.0,
            load=None,
            ram_usage=0.0,
            cpu_usage=None,
            arrival_time=0.0,
            duration=0.0,
            extras=None,
            gpu_usage=None,
            cpu_model=None,
            gpu_model=None,
            cpu_clock=None,
            gpu_clock=None,
            time_util=None,
        )
    except Exception:
        # final fallback: try positional
        try:
            return user.Request(int(i), int(i))
        except Exception:
            # ultimate fallback: a simple object with request_id attribute
            ns = SimpleNamespace(request_id=int(i), process_id=int(i))
            return ns


def main():
    # --------------- setup controller ---------------
    controller = Controller(num_servers=1)  #Change number of servers accordingly
    # ---------------- setup receiver ----------------
    receiver = Receiver(persist_chunks=True, process_time_per_chunk=0.2, controller=controller)
    
    controller.run()


    # ---------------- create sender ----------------
    sender = SenderBursts(
        arr=None,               # auto-generate using request_factory
        sample_count=12,       # total requests
        chunk_size=12,          # requests per chunk
        bursts=5,
        min_burst=1,
        max_burst=8,
        min_interval=0.2,
        max_interval=0.8,
        jitter=0.02,
        request_factory=request_factory,   # use correct Request constructor
        host="127.0.0.1",
        port=6000,
    )

    # ---------------- run sender (blocking) ----------------
    try:
        print("[MAIN] Starting sender bursts...")
        stats = sender.run()
        print("[MAIN] Sender finished. Stats:", stats)
    except KeyboardInterrupt:
        print("[MAIN] Sender interrupted.")
    finally:
        # give some time for receiver to finish processing
        time.sleep(1.0)
        print("[MAIN] Stopping receiver...")
        receiver.stop()
        if getattr(receiver, "_thread", None):
            receiver._thread.join(timeout=3.0)
        print("[MAIN] Exiting.")

if __name__ == "__main__":
    main()
