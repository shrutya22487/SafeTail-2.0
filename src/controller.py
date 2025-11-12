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
        
        self.agent = DQNAgent(
            states=constants.nS,
            actions=constants.nA,
            alpha=constants.alpha,
            reward_gamma=constants.discount_rate,
            epsilon=1.0,
            epsilon_min=0.000001,
            epsilon_decay=constants.gamma_decay,
            batch_size=constants.batch_size,
            beta=constants.beta,
            median_computation_delay=constants.median_computation_delay,
            learning_rate=constants.learning_rate,
            task=None,
            epochs=constants.no_of_episodes,
            server_list=self.server_list,
            request=None
        )

    def find_free_servers(self):
        now = time.time()
        load = [s.check_server_availability(now) for s in self.server_list]
        return np.array(load)

    def dispatch_to_agent(self, request):
        
        return self.agent.get_action(request)

    def assign_request(self, request, indices):
        
        print(f"[CONTROLLER, ASSIGN] Assigning Request {request.request_id} to servers: {indices}")

        for i in indices:
            
            # # Debug statements
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

                num_free = sum(l != 1e9 for l in load)
                if num_free == 0:
                    print("[CONTROLLER, QUEUE] All servers busy. Retrying shortly.")
                    time.sleep(0.1)
                    continue

                req.load = np.array(load)
                req.total_delay = [self.server_list[i].compute_request_time(req) for i in range(self.num_servers)]
                
                
                #TODO:
                """
                Have restructured the training component in our edge controller system.
                In the original implementation, the DQN agent was trained offline using a fixed number of episodes and steps (no_of_episodes X len_of_episode), 
                with pre-recorded data used to simulate requests and states.

                However, in our current controller-based setup, 
                requests arrive dynamically and continuously from the network 
                
                — meaning the “steps” and “episodes” occur naturally during runtime???? 
                
                Not sure how to best adapt the training loop to this new context."""
                
                
                
                """
                the controller now:

                1. Receives real incoming Request objects,

                2. Uses the RL agent’s get_action() to select servers,

                3. Computes latency-based rewards in real time,

                4. Stores (state, action, reward, next_state) into replay memory, and

                5. Performs training (experience_replay()) incrementally after each request.

                6. also added periodic model checkpointing and metric logging every 500 requests to monitor training progress over time.
                """

                # ---------------- RL Decision ----------------
                action_subset, action_index = self.agent.get_action(req)

                # ---------------- Execute Action ----------------
                self.assign_request(req, action_subset)

                # ---------------- Compute Reward ----------------
                reward_value = self.agent.reward(action_subset, req)  # uses latency feedback

                # ---------------- Store Experience ----------------
                # For now, next_state = current_state (can extend later)
                self.agent.store(req, action_index, reward_value, req)

                # ---------------- Optional: Log Epsilon ----------------
                self.agent.epsilon_curve = np.append(self.agent.epsilon_curve, self.agent.epsilon)

                # # ---------------- Periodic Training ----------------
                # # The reward() method already calls replay internally when memory > batch_size.
                # # But for explicit control, we can do:
                # if len(self.agent.memory) > self.agent.batch_size:
                #     self.agent.experience_replay(self.agent.batch_size)

            # ---------------- ✅ Checkpoint & Metrics Logging ----------------
            # Save model and metrics every 500 requests (adjust as needed)
            if len(self.agent.memory) % 500 == 0:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_dir = Path("training_logs")
                save_dir.mkdir(exist_ok=True)

                # Save model
                model_path = save_dir / f"model_checkpoint_{timestamp}.keras"
                self.agent.model.save(model_path)
                print(f"[CONTROLLER] ✅ Model checkpoint saved at {model_path}")

                # Save metrics
                metrics_path = save_dir / f"metrics_{timestamp}.csv"
                pd.DataFrame({
                    "epsilon": self.agent.epsilon_curve,
                    "reward": self.agent.rewards,
                    "loss": self.agent.loss,
                    "val_loss": self.agent.val_loss,
                    "latency": self.agent.latencies,
                    "deviation": self.agent.deviations,
                }).to_csv(metrics_path, index=False)
                print(f"[CONTROLLER] 📊 Metrics logged to {metrics_path}")


                
    def run(self):
        # ---------------- start receiver ----------------
        self.receiver_queue.run_async()
        print("[CONTROLLER] Receiver started in background thread.")
        time.sleep(1.0)  # let it bind
