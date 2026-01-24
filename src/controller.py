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
        Step reward computed PER REQUEST:
        
        Reward function:
        R = 1 + log( e^((1 - cm)(1 - cu)(1 - gm)(1 - gu)) - 1 )

        Where:
        cm = RAM utilization = RAM_used / Total_RAM_available
        cu = CPU core utilization = Total_core_utilization_% / (Number_of_cores * 100)
        gm = GPU memory utilization = GPU_memory_used / Total_GPU_memory_available
        gu = GPU core utilization = GPU_core_utilization
        """
        # cm = request.cm
        # cu = request.cu
        # gm = request.gm
        # gu = request.gu

        # core = (1 - cm) * (1 - cu) * (1 - gm) * (1 - gu)

        # # numerical stability
        # reward = 1.0 + np.log(np.exp(core) - 1.0 + 1e-8)

        # print(
        #     f"[STEP REWARD] cm={cm:.3f}, cu={cu:.3f}, gm={gm:.3f}, gu={gu:.3f} → R={reward:.4f}"
        # )
        
        reward = 0.6  # Placeholder for actual reward calculation

        return np.array([reward, reward, reward, reward, reward]) # Placeholder for per-server rewards


    def compute_episodic_reward(self):
        """
        Compute episodic reward as per new architecture:
        R_episode = Σ(γ^i * R_step^(i+1)) + ω - W_avg

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
        print(f"\n[CONTROLLER, STEP {self.current_step}] Processing request {request.request_id}.")
        
        load = self.find_free_servers()
        num_free = sum(l != 1e9 for l in load)
        
        if num_free == 0:
            print("[CONTROLLER, QUEUE] All servers busy. Retrying shortly.")
            time.sleep(0.1)
            return

        # update request state
        request.load = load
        request.total_delay = []
        combined_strs = []

        for i in range(self.num_servers):
            delay, combined_str = self.server_list[i].compute_request_time(request)
            request.total_delay.append(delay)
            combined_strs.append(combined_str)
        
        # Update request state to add data from servers
        for i in range(len(combined_strs)):
            server_index = i + 1
            request.populate_request_from_csv(server_index, combined_strs[i])
        
        request.step_reward_list = self.compute_step_reward(request)
        
        # agent action
        action_subset, action_index = self.agent.get_action(request)
        
        # assign request
        for i in action_subset:
            self.server_list[i].schedule_request(request)
        
        # compute reward
        final_step_reward_list = self.compute_step_reward(request, action_subset)
        
        combined_step_reward = np.mean(final_step_reward_list)
        
        print(f"[CONTROLLER, STEP {self.current_step}] Completed. Step reward: {combined_step_reward:.3f}")
        
        # store experience
        self.step_experiences.append({
            "state": request,
            "action": action_index,
            "reward": combined_step_reward,
            "next_state": request  # environment is partially observable anyway
        })
        
        self.step_rewards.append(combined_step_reward)
        self.agent.epsilon_curve = np.append(
            self.agent.epsilon_curve, self.agent.epsilon
        )
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
        print("[CONTROLLER] Starting main control loop with step/episodic rewards.")
        print(
            f"[CONTROLLER] Processing chunk {self.current_chunk + 1}/"
            f"{self.chunks_per_episode}"
        )
        for request in chunk:
            with self.lock:
                # Process this chunk as one STEP
                self.process_step(request)
        self.current_chunk += 1
        
        # Check if episode is complete
        if self.current_chunk >= self.chunks_per_episode:
            self.finalize_episode()

    def run(self):
        # ---------------- start receiver ----------------
        self.receiver_queue.run_async()
        print("[CONTROLLER] Receiver started in background thread.")
        time.sleep(1.0)  # let it bind
