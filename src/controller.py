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
    def __init__(self, num_servers=5, steps_per_episode=10, chunks_per_episode=3):

        self.num_servers = num_servers
        self.server_list = [servers.Server(i + 1) for i in range(num_servers)]

        self.lock = threading.Lock()
        self.receiver_queue = None  # attached externally

        # ---------------- Episode / Step tracking ----------------
        self.chunks_per_episode = chunks_per_episode
        self.steps_per_episode = steps_per_episode # Use if required.
        self.current_chunk = 0
        self.current_step = 0
        self.current_episode = 0
        self.training_done = threading.Event()
        self.expected_episodes = constants.no_of_episodes
        self.step_experiences = []
        self.step_rewards = []

        # ---------------- Agent ----------------
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
    
    # ------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------
    def find_free_servers(self):
        now = time.time()
        load = [s.check_server_availability(now) for s in self.server_list]
        return np.array(load)

    def get_queue_lengths(self):
        """Get current queue length for each server."""
        return np.array([len(s.active_requests) for s in self.server_list])

    def dispatch_to_agent(self, request):
        return self.agent.get_action(request)

    def assign_request(self, request, indices):

        print(f"[CONTROLLER, ASSIGN] Assigning Request {request.request_id} to servers: {indices}")

        for i in indices:
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
        
    # ------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------
    def compute_step_reward(self, request, action_subset=None):
        """
        Step reward computed PER REQUEST (per server).

        Reward:
        R = 1 + log( exp((1 - cm)(1 - cu)(1 - gm)(1 - gu) - 1 ))
        
        Where:
        cm = RAM utilization = RAM_used / Total_RAM_available
        cu = CPU core utilization = (sum of CPU core percentages/100) / Number_of_cores
        gm = GPU memory utilization = peak GPU_memory / Total_GPU_memory_available
        gu = GPU core utilization = GPU_core_utilization

        If action_subset is None → compute for all servers
        Otherwise → compute only for servers in action_subset
        """

        num_servers = len(request.server_dicts)
        rewards = np.zeros(num_servers, dtype=float)

        # Decide which servers to evaluate (1-based)
        if action_subset is None:
            server_indices = range(num_servers)
        else:
            server_indices = action_subset

        for server_idx in server_indices:
            try:
                # ---- Index validation ----
                if not (0 <= server_idx < num_servers):
                    raise IndexError(f"[CONTROLLER]...[STEP REWARD][ERROR]Invalid server index: {server_idx}")

                i = server_idx  # 0-based
                d = request.server_dicts[i]  # 1-based in request ??? or 0-based???

                # Skip if server dict not populated
                if not d:
                    rewards[i] = 0.0
                    continue

                # ---- Required fields ----
                required_keys = [
                    "ram_usage", "total_ram",
                    "cpu_core_usage", "total_cpu_cores",
                    "gpu_memory", "total_gpu_memory",
                    "gpu_usage",
                ]
                for k in required_keys:
                    if k not in d:
                        raise KeyError(f"[CONTROLLER]...[STEP REWARD][ERROR] Missing key '{k}' for server {server_idx}")

                # ---- Utilizations ----
                # print()
                GB_TO_MB = 1024.0

                total_ram_mb = float(d["total_ram"]) * GB_TO_MB
                cm = d["ram_usage"] / max(total_ram_mb, 1e-8)
                # print(f"RAM Usage: {d['ram_usage']}, Total RAM: {d['total_ram']}, CM: {cm}")

                cpu_cores_utilised = np.sum(d["cpu_usage"])/100.0
                cu = cpu_cores_utilised / (float(d["total_cpu_cores"]))
                # print(f"CPU Usage: {cpu_cores_utilised}, Total CPU Cores: {d['total_cpu_cores']}, CU: {cu}")

                total_gpu_mem_mb = float(d["total_gpu_memory"]) * GB_TO_MB
                gm = d["gpu_memory"] / max(total_gpu_mem_mb, 1e-8)
                # print(f"GPU Memory Usage: {d['gpu_memory']}, Total GPU Memory: {d['total_gpu_memory']}, GM: {gm}")

                gu = float(d["gpu_usage"]) / 100.0
                # print(f"GPU Usage: {d['gpu_usage']}, GU: {gu}")
                # print()

                # ---- Reward ----
                product = (1 - cm) * (1 - cu) * (1 - gm) * (1 - gu)

                reward = np.log(np.exp(product + 1.0))
                # print(f"[CONTROLLER]...[STEP REWARD] Server {server_idx}: Reward={reward:.6f}")
                
                # print()
                # print()

                # Final sanity check
                if not np.isfinite(reward):
                    raise ValueError(f"[CONTROLLER]...[STEP REWARD][ERROR] Non-finite reward computed for server {server_idx}")

                rewards[i] = reward

            except Exception as e:
                # Fail-safe: zero reward + log
                rewards[server_idx - 1] = 0.0
                print(
                    f"[CONTROLLER]...[STEP REWARD][ERROR] Server {server_idx}: {type(e).__name__} - {e}"
                )

        return rewards


    def compute_episodic_reward(self):
        """
        Compute episodic reward as per new architecture:
        R_episode = Σ(γ^i * (i+1th R_step)) + ω - W_avg

        Where:
        - γ is discount factor
        - R_step are the step rewards collected
        - ω is degree of satisfaction (requests within deadline)
        - W_avg is average waiting time
        """
        gamma = self.agent.reward_gamma

        # Discounted sum of step rewards
        discounted_step_rewards = sum(
            (gamma ** i) * r for i, r in enumerate(self.step_rewards)
        )

        # Degree of satisfaction (simplified: percentage of successful assignments)
        # TODO: Implement proper deadline-based satisfaction metric
        omega = 1.0  # Placeholder: assume all requests satisfied

        # Average waiting time across the episode
        avg_waiting_time = 0 # Placeholder: implement actual waiting time calculation

        # Episodic reward
        episodic_reward = discounted_step_rewards + omega - avg_waiting_time

        print(f"[CONTROLLER, EPISODE REWARD] Discounted steps: {discounted_step_rewards:.3f}, "
              f"Satisfaction: {omega:.3f}, Avg wait: {avg_waiting_time:.3f}, "
              f"R_episode: {episodic_reward:.3f}")

        return episodic_reward


    def process_step(self, request):
        """
        Process one step (ONE STEP = processing ONE request).
        Collects experiences but doesn't train yet.
        """
        try:
            print(f"\n[CONTROLLER, STEP {self.current_step}] Processing request {request.request_id}.")
        except Exception:
            print(f"\n[CONTROLLER, STEP {self.current_step}] Processing request <unknown>.")

        try:
            load = self.find_free_servers()
            num_free = sum(l != 1e9 for l in load)
        except Exception as e:
            print(f"[CONTROLLER, !] Failed to find free servers: {type(e).__name__} - {e}")
            return
        
        if num_free == 0:
            print("[CONTROLLER, QUEUE] All servers busy. Retrying shortly.")
            time.sleep(0.1)
            return

        # update request state
        try:
            request.load = load
            request.total_delay = []
            combined_strs = []
        except Exception as e:
            print(f"[CONTROLLER, !] Failed to update request state: {type(e).__name__} - {e}")
            return

        for i in range(self.num_servers):
            try:
                delay, combined_str = self.server_list[i].compute_request_time(request)
                request.total_delay.append(delay)
                combined_strs.append(combined_str)
            except Exception as e:
                print(
                    f"[CONTROLLER, !] Error computing request time "
                    f"for server {i + 1}: {type(e).__name__} - {e}"
                )
                request.total_delay.append(float("inf"))
                combined_strs.append(None)

        # Update request state to add data from servers
        for i in range(len(combined_strs)):
            server_index = i
            try:
                if combined_strs[i] is not None:
                    request.populate_request_from_csv(server_index, combined_strs[i])
            except Exception as e:
                print(
                    f"[CONTROLLER, !] Failed to populate request from CSV "
                    f"for server {server_index}: {type(e).__name__} - {e}"
                )

        try:
            request.step_reward_list = self.compute_step_reward(request)
        except Exception as e:
            print(f"[CONTROLLER, !] Failed to compute initial step reward: {type(e).__name__} - {e}")
            request.step_reward_list = np.zeros(self.num_servers, dtype=float)
        
        # agent action
        try:
            action_subset, action_index = self.agent.get_action(request)
        except Exception as e:
            print(f"[CONTROLLER, !] Agent failed to produce action: {type(e).__name__} - {e}")
            return
        
        # assign request
        for i in action_subset:
            try:
                self.server_list[i].schedule_request(request)
            except Exception as e:
                print(
                    f"[CONTROLLER, !] Failed to schedule request on server {i}: "
                    f"{type(e).__name__} - {e}"
                )
        
        # compute reward
        try:
            final_step_reward_list = self.compute_step_reward(request, action_subset)
            combined_step_reward = np.mean(final_step_reward_list)
        except Exception as e:
            print(f"[CONTROLLER, !] Failed to compute final step reward: {type(e).__name__} - {e}")
            combined_step_reward = 0.0
        
        try:
            print(
                f"[CONTROLLER, STEP {self.current_step}] Completed. "
                f"Step reward: {combined_step_reward:.3f}"
            )
        except Exception:
            pass
        
        # store experience
        try:
            self.step_experiences.append({
                "state": request,
                "action": action_index,
                "reward": combined_step_reward,
                "next_state": request  # environment is partially observable anyway
            })
        except Exception as e:
            print(f"[CONTROLLER, !] Failed to store experience: {type(e).__name__} - {e}")
        
        try:
            self.step_rewards.append(combined_step_reward)
            self.agent.epsilon_curve = np.append(
                self.agent.epsilon_curve, self.agent.epsilon
            )
        except Exception as e:
            print(f"[CONTROLLER, !] Failed to update reward/epsilon tracking: {type(e).__name__} - {e}")

        self.current_step += 1


    def finalize_episode(self):
        """
        Finalize the episode:
        1. Compute episodic reward
        2. Store all experiences in replay buffer
        3. Train the agent
        """
        print(f"\n{'='*60}")
        print(f"[CONTROLLER, EPISODE {self.current_episode}] Finalizing...")

        # Compute episodic reward
        episodic_reward = self.compute_episodic_reward()
        
        print(
            f"[EPISODE {self.current_episode}] "
            f"Steps={len(self.step_rewards)}, "
            f"Reward={episodic_reward:.4f}"
        )

        # Store all experiences from this episode with episodic reward
        for exp in self.step_experiences:
            self.agent.store(
                state_request=exp['state'],
                action=exp['action'],
                reward=episodic_reward,  # Use episodic reward for all experiences
                next_state_request=exp['next_state']
            )

        # Train agent once per episode
        if len(self.agent.memory) >= self.agent.batch_size:
            print(f"[CONTROLLER, EPISODE {self.current_episode}] Training agent...")
            self.agent.experience_replay(self.agent.batch_size)
            print(f"[CONTROLLER, EPISODE {self.current_episode}] Training complete.")
        else:
            print(
                f"[CONTROLLER, EPISODE {self.current_episode}] "
                f"Not enough experiences to train "
                f"(have {len(self.agent.memory)}, need {self.agent.batch_size})"
            )
        # Log episode metrics
        print(f"[CONTROLLER, EPISODE {self.current_episode}] "
              f"Experiences: {len(self.step_experiences)}, "
              f"Episodic Reward: {episodic_reward:.3f}, "
              f"Epsilon: {self.agent.epsilon:.6f}")

        # Checkpoint every N episodes
        # if self.current_episode % 10 == 0:
        #     self.save_checkpoint()

        # Reset episode-level tracking
        self.step_experiences = []
        self.step_rewards = []
        self.episode_waiting_times = []
        self.current_chunk = 0
        self.current_step = 0
        self.current_episode += 1
        if self.current_episode >= self.expected_episodes:
            print("[CONTROLLER] ✅ All training episodes finished.")
            self.training_done.set()


        print(f"{'='*60}\n")


    def save_checkpoint(self):
        """Save model and metrics."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_dir = Path("training_logs")
        save_dir.mkdir(exist_ok=True)

        # Save model
        model_path = save_dir / f"model_ep{self.current_episode}_{timestamp}.keras"
        self.agent.model.save(model_path)
        print(f"[CONTROLLER] ✅ Model checkpoint saved at {model_path}")

        # Save metrics
        metrics_path = save_dir / f"metrics_ep{self.current_episode}_{timestamp}.csv"
        pd.DataFrame({
            "epsilon": self.agent.epsilon_curve,
            "reward": self.agent.rewards,
            "loss": self.agent.loss,
            "val_loss": self.agent.val_loss,
            "latency": self.agent.latencies,
            "deviation": self.agent.deviations,
        }).to_csv(metrics_path, index=False)
        print(f"[CONTROLLER] 📊 Metrics logged to {metrics_path}")


    # ---- Controller loop ----
    def send_to_server(self, chunk):
        """
        Main control loop implementing step/episodic training.

        Architecture:
        - STEP = processing one of the requests in a chunk
        - EPISODE = processing 3 chunks (chunk = list of requests)
        - Training happens ONCE per episode
        """
        try:
            print("[CONTROLLER] Starting main control loop with step/episodic rewards.")
            print(
                f"[CONTROLLER] Processing chunk {self.current_chunk + 1}/"
                f"{self.chunks_per_episode}"
            )

            # Validate chunk
            if chunk is None:
                raise ValueError("[CONTROLLER, !] Received None chunk")

            if not hasattr(chunk, "__iter__"):
                raise TypeError(f"[CONTROLLER, !] Chunk is not iterable: {type(chunk)}")

            for request in chunk:
                try:
                    with self.lock:
                        # Process this chunk as one STEP
                        self.process_step(request)
                except Exception as e:
                    # Fail-soft: skip bad request, continue episode
                    print(
                        f"[CONTROLLER, !] Error processing request "
                        f"{repr(request)}: {type(e).__name__} - {e}"
                    )
                    continue

            self.current_chunk += 1

            # Check if episode is complete
            if self.current_chunk >= self.chunks_per_episode:
                try:
                    self.finalize_episode()
                except Exception as e:
                    # Episode finalization is critical but must not crash controller
                    print(
                        f"[CONTROLLER, !] Error finalizing episode: "
                        f"{type(e).__name__} - {e}"
                    )

        except Exception as e:
            # Catch-all to protect controller thread
            print(
                f"[CONTROLLER, !] Fatal error in send_to_server: "
                f"{type(e).__name__} - {e}"
            )


    def run(self):
        # ---------------- start receiver ----------------
        self.receiver_queue.run_async()
        print("[CONTROLLER] Receiver started in background thread.")
        time.sleep(1.0)  # let it bind
