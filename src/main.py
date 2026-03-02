#!/usr/bin/env python3
"""
main.py

Run Receiver in background and SenderBursts in foreground.
This request_factory constructs user.Request objects using the correct signature.
"""

import time
from types import SimpleNamespace
from receiver import Receiver
from sender_bursts import SenderBursts
import user
from controller import Controller
import constants
import random
import pandas as pd
from pathlib import Path
import numpy as np
import logging
import warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---- load server CSVs once (NOT per request) ----
BASE_DIR = Path(__file__).resolve().parent.parent  # go from src/ -> project root
DATA_DIR = BASE_DIR / "data"

SERVER_CSVS = [
    DATA_DIR / "server1.csv",
    DATA_DIR / "server2.csv",
    DATA_DIR / "server3.csv",
    DATA_DIR / "server4.csv",
    DATA_DIR / "server5.csv",
]
import inspect

def print_constants(module):
    print("\n" + "=" * 50)
    print(f"  CONSTANTS ({module.__name__})")
    print("=" * 50)
    for name, value in inspect.getmembers(module):
        # Skip built-ins, modules, and callables
        if not name.startswith("_") and not inspect.ismodule(value) and not callable(value):
            print(f"  {name:<30} = {value}")
    print("=" * 50 + "\n")

print_constants(constants)

SERVER_DFS = []
for p in SERVER_CSVS:
    if not p.exists():
        raise FileNotFoundError(f"Missing CSV: {p}")
    SERVER_DFS.append(pd.read_csv(p))

# --- request factory that uses the actual Request signature ---
def request_factory(i: int):
    """
    Pure Request factory.

    Responsibilities:
    - create Request object
    - set ids, combination, arrival time
    - initialize base arrays ONLY

    Non-responsibilities:
    - NO server CSV access
    - NO filling server dicts
    - NO filling server NP arrays
    - NO computation

    This function MUST NOT crash the sender.
    """

    # -------- safest defaults --------
    server_count = 5
    # added the deadlines these are in milli- second ms
    
    deadlines = np.asarray([
        [100,400],
        [30,200]
    ])

    try:
        # ---------------- combination ----------------
        try:
            combination = random.choice(["s", "p", "d"])
            if(combination == "s"):
                deadline  = deadlines[0]
            else:
                deadline = deadlines[1]
        
        except Exception:
            # absolute fallback
            combination = "s"
            deadline = deadlines[0]

        # ---------------- construct request ----------------
        req = user.Request(
            request_id=int(i),
            process_id=int(i),
            combination=combination,
            message_size=1024,
            bandwidth=20,
            load=np.zeros(server_count, dtype=int),
            deadline = deadline
        )

        return req

    except Exception as e:
        # ==================================================
        # HARD FAILURE: Request constructor failed
        # ==================================================
        try:
            logger.error(
                "[request_factory] Failed to construct Request. "
                f"i={i}, error={e}",
                exc_info=True
            )
        except Exception:
            pass  # logging must never break execution

        # ==================================================
        # FALLBACK 1: minimal valid Request
        # ==================================================
        try:
            return user.Request(
                request_id=int(i),
                process_id=int(i),
                combination="s",
                message_size=np.zeros(server_count, dtype=float),
                bandwidth=np.zeros(server_count, dtype=float),
                load=np.zeros(server_count, dtype=int),
                deadline = deadlines[0]
            )
        except Exception:
            # ==================================================
            # FALLBACK 2: bare minimum namespace (last resort)
            # ==================================================
            try:
                return SimpleNamespace(
                    request_id=int(i),
                    process_id=int(i),
                    combination="s",
                    arrival_time=time.time() * 1000.0,
                    deadline = deadlines[0]
                )
            except Exception:
                # ==================================================
                # FALLBACK 3: absolute last resort
                # ==================================================
                return None

def main():
    # --------------- setup controller ---------------
    controller = Controller(num_servers=constants.beta)  #Change number of servers accordingly
    # ---------------- setup receiver ----------------
    receiver = Receiver(persist_chunks=True, process_time_per_chunk=0.2, controller=controller)
    
    controller.run()


    # ---------------- create sender ----------------
    sender = SenderBursts(
        arr=None,
        sample_count=constants.total_no_request,        # total requests
        chunk_size=constants.chunk_size,          # requests per chunk
        bursts=constants.no_of_burst,
        min_burst=constants.min_burst,
        max_burst=constants.max_burst,
        min_interval=constants.min_interval,
        max_interval=constants.max_interval,
        jitter=constants.jitter,
        request_factory=request_factory,
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
        print("[MAIN] Waiting for training to finish...")

        try:
            controller.training_done.wait()   # BLOCKS safely
        except KeyboardInterrupt:
            print("\n[MAIN] Ctrl+C received early.")
        finally:
            print("[MAIN] Stopping receiver...")
            receiver.stop()
            receiver._thread.join(timeout=5)
            print("[MAIN] Exiting cleanly.")


if __name__ == "__main__":
    main()
