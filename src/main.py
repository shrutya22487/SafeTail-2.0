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
    
    try:
        # ---------------- combination ----------------
        combination = random.choice(["s", "p", "d"])

        duration = []
        proc_time = []
        ram = []
        cpu = []

        extras = {
            "scripts_executed": [],
            "individual_script_time": []
        }

        for df in SERVER_DFS:
            row = df[df["Combination"] == combination]

            if row.empty:
                raise ValueError(f"No row for combination '{combination}'")

            row = row.iloc[0]

            duration.append(float(row["Total Execution Time (sec)"]))
            proc_time.append(float(row["Total Processing Time (sec)"]))

            ram.append(float(row.get("RAM Usage (MB)", 0.0)))
            cpu.append(float(row.get("CPU Usage (%)", 0.0)))

            extras["scripts_executed"].append(row.get("Scripts Executed"))
            extras["individual_script_time"].append(
                float(row.get("Individual Script Time (sec)", 0.0))
            )

        time_util = np.column_stack([proc_time, duration])

        return user.Request(
            
            # @shivankar: pl complete
            # Keep these...
            request_id=i,
            process_id=i,
            combination=combination,
            message_size=np.zeros(len(SERVER_DFS)),
            bandwidth=np.zeros(len(SERVER_DFS)),
            load=np.zeros(len(SERVER_DFS), dtype=int),
            arrival_time=time.time() * 1000.0,  # ms
            # ######################################
            
            # server1_dict = : 
            # server2_dict = : 
            # server3_dict = : 
            # server4_dict = : 
            # server5_dict = :

            # server1 _ Np =: {}
            # server2 _ Np =: {}
            # server3 _ Np =: {}
            # server4 _ Np =: {}
            # server5 _ Np =: {}
            
            # ##########################

            
            ram_usage=ram,
            cpu_usage=cpu,

            
            duration=duration,
            time_util=time_util,

            gpu_usage=None,
            cpu_model=None,
            gpu_model=None,
            cpu_clock=None,
            gpu_clock=None,

            extras=extras,
        )

    except Exception as e:
        # -------- fallback 1: minimal Request --------
        try:
            return user.Request(
                request_id=int(i),
                process_id=int(i),
                combination="s",
                message_size=np.zeros(len(SERVER_DFS)),
                bandwidth=np.zeros(len(SERVER_DFS)),
                load=np.zeros(len(SERVER_DFS), dtype=int),
                ram_usage=np.zeros(len(SERVER_DFS)),
                cpu_usage=np.zeros(len(SERVER_DFS)),
                arrival_time=time.time() * 1000.0,
                duration=np.zeros(len(SERVER_DFS)),
                time_util=np.zeros((len(SERVER_DFS), 2)),
                extras={},
            )
        except Exception:
            # -------- fallback 2: bare minimum --------
            return SimpleNamespace(
                request_id=int(i),
                process_id=int(i),
                combination="s",
                arrival_time=time.time() * 1000.0,
            )


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
