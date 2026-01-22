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
import os
import random
import pandas as pd
from pathlib import Path
import numpy as np
import logging

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

    try:
        # ---------------- combination ----------------
        try:
            combination = random.choice(["s", "p", "d"])
        except Exception:
            # absolute fallback
            combination = "s"

        # ---------------- construct request ----------------
        req = user.Request(
            request_id=int(i),
            process_id=int(i),
            combination=combination,

            message_size=np.zeros(server_count, dtype=float),
            bandwidth=np.zeros(server_count, dtype=float),
            load=np.zeros(server_count, dtype=int),
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
        arr=None,               # auto-generate using request_factory
        sample_count=15,        # total requests
        chunk_size=5,          # requests per chunk
        bursts=3,
        min_burst=1,
        max_burst=2,
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
