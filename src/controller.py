#!/usr/bin/env python3
import threading
import time
import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path

import servers
import constants
from agent import DQNAgent

class Controller:
    def __init__(self, num_servers=5):
        
        self.num_servers = num_servers
        self.server_list = [servers.Server(i + 1) for i in range(num_servers)]
        self.queue = deque(maxlen=50)
        self.lock = threading.Lock()
        self.receiver_queue = None  # to be attached later

    def find_free_servers(self):
        now = time.time()
        load = [(i, s.check_server_availability(now)) for i, s in enumerate(self.server_list)]
        return np.array(load)

    def dispatch_to_agent(self, request):
        
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
        
        return agent.get_action(request)

    def assign_request(self, request, indices):
        
        print(f"[CONTROLLER, ASSIGN] Assigning Request {request.request_id} to servers: {indices}")

        for i in indices:
            
            # Debug statements
            # print(f"Index is {i}")
            # print(f"server_index is {self.server_list[i].server_index}")
            # print(f"server_list length is {len(self.server_list)}")
            
            try:
                ok, finish, proc = self.server_list[i].schedule_request(request)

                if ok:
                    print(f"[CONTROLLER, ASSIGN] Req {request.request_id} → Server {i+1} ({proc:.3f}s)")
                else:
                    print(f"[CONTROLLER, BUSY] Server {i+1} full for Req {request.request_id}")

            except (IndexError, TypeError) as e:
                print(f"[CONTROLLER, ERROR:1] {e}")
                break
            except Exception as e:
                print(f"[CONTROLLER, ERROR:2] Unexpected error for Req {request.request_id} on server {i}: {e}")
            
        print("\n---- ---- ---- \n")




    # ---- Controller loop ----
    def send_to_server(self, chunk):
        print("[CONTROLLER] Starting main control loop.")

        while True:
            with self.lock:
                arr = chunk
                
            print(f"[CONTROLLER, PROCESS] Dequeued chunk with {len(arr)} requests.")
            for req in arr:
                load = self.find_free_servers()

                num_free = 0
                for _ in range(len(load)):
                    if(load[_] != 1e9):
                        num_free += 1
                if(num_free == 0):
                    print("[CONTROLLER, QUEUE] All servers busy. Retrying shortly.")
                    time.sleep(0.1)
                    continue
                
                req.load = np.array(load)
                
                req.total_delay = [self.server_list[i].compute_request_time(req) for i in range(self.num_servers)]
                
                # decision
                action_subset, action_index = self.dispatch_to_agent(req)
                
                self.assign_request(req, action_subset)
                
    def run(self):
        # ---------------- start receiver ----------------
        self.receiver_queue.run_async()
        print("[CONTROLLER] Receiver started in background thread.")
        time.sleep(1.0)  # let it bind
