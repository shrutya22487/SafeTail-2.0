#!/usr/bin/env python3
import threading
import time
import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path

import servers
import constants
from agent import DQNAgent, get_state_input

class Controller:
    def __init__(self, num_servers=5, server_state_csv="server_state.csv"):
        base_dir = Path(__file__).resolve().parent  # This points to src/
        self.server_state_csv = base_dir.parent / "data" / server_state_csv
        if not self.server_state_csv.exists():
            raise FileNotFoundError(f"Detect CSV not found: {self.detect_csv}")
        
        self.num_servers = num_servers
        self.server_list = [servers.Server(i + 1) for i in range(num_servers)]
        self.queue = deque(maxlen=50)
        self.lock = threading.Lock()
        self.receiver_queue = None  # to be attached later


    # ---- Internal utility methods ----
    def read_server_state(self):
        """
        Randomly sample one row from the CSV (ignoring Timestamp column)
        and assign it as current load (num_requests) for each server.
        """
        try:
            df = pd.read_csv(self.server_state_csv)
            # Drop the Timestamp column
            df = df.drop(columns=["Timestamp"])
            
            # Randomly pick one row
            sampled_row = df.sample(n=1).iloc[0]

            # Assign to servers
            for i, srv in enumerate(self.server_list):
                srv.num_requests = int(sampled_row[i])

        except Exception as e:
            print(f"[!] Could not read {self.server_state_csv}: {e}")

    def find_free_servers(self):
        now = time.time()
        free = [i for i, s in enumerate(self.server_list) if s.check_server_availability(now)]
        return np.array(free)

    def dispatch_to_agent(self, request):
        state = request.to_state()
        agent = DQNAgent(
            states=constants.nS,
            actions=constants.nA,
            alpha=constants.alpha,
            reward_gamma=constants.discount_rate,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=constants.gamma_decay,
            batch_size=constants.batch_size,
            beta=constants.beta,
            median_computation_delay=constants.median_computation_delay,
            learning_rate=constants.learning_rate,
            task=None,
            epochs=constants.no_of_episodes,
            server_list = self.server_list,
            request=None  # will be set per request
        )
        return agent.get_action(state)[0]

    def assign_request(self, request, indices):
        for i in indices:
            ok, finish, proc = self.server_list[i].schedule_request(request)
            if ok:
                print(f"[ASSIGN] Req {request.request_id} → Server {i+1} ({proc:.3f}s)")
            else:
                print(f"[BUSY] Server {i+1} full for Req {request.request_id}")

    # ---- Controller loop ----
    def send_to_server(self, chunk):
        print("[CONTROLLER] Starting main control loop.")
        self.read_server_state()

        while True:
            with self.lock:
                arr = chunk
                
            print(f"[PROCESS] Dequeued chunk with {len(arr)} requests.")
            for req in arr:
                free = self.find_free_servers()
                if not len(free):
                    print("[QUEUE] All servers busy. Retrying shortly.")
                    time.sleep(0.1)
                    continue
                req.load = np.array([s.num_requests for s in self.server_list])
                decision = self.dispatch_to_agent(req)
                self.assign_request(req, decision)
                
    def run(self):
        # ---------------- start receiver ----------------
        self.receiver_queue.run_async()
        print("[CONTROLLER] Receiver started in background thread.")
        time.sleep(1.0)  # let it bind
