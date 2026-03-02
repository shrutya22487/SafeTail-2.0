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
        
        # Values for D1 and D2 based on the combination type of the request (e.g., "s" or "d" or "p").
        self.deadlines = np.asarray([ [100,400], [30,200] ])  # in ms

        # ---------------- Episode / Step tracking ----------------
        self.chunks_per_episode = chunks_per_episode
        self.steps_per_episode = steps_per_episode  # Use if required.
        self.current_chunk = 0
        self.current_step = 0
        self.current_episode = 0
        self.training_done = threading.Event()
        self.expected_episodes = constants.no_of_episodes
        self.step_experiences = []
        self.step_rewards = []
        self.episode_start_time = None
        self.average_P_T_values = []  # To track P(T) values for satisfaction calculation
        self.average_waiting_times = []  # To track average waiting time for each request in the queue per episode
        # Tracking for request completion counts (for use in episodic reward shaping)
        self.request_s_done = 0
        self.request_s_total = 0
        self.request_d_done = 0
        self.request_d_total = 0
        self.request_p_done = 0
        self.request_p_total = 0

        # ---------------- Plotting Configuration ----------------
        self.plot_every_n_episodes = 20  # Generate plots every N episodes
        self.plot_dir = Path(constants.training_log_folder + "/plots")
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # ---------------- Agent ----------------
        self.agent = DQNAgent(
            states=constants.nS,
            actions=constants.nA,
            alpha=constants.alpha,
            reward_gamma=constants.discount_rate,
            epsilon=1.0,
            epsilon_min=constants.epsilon_min,
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

        # ---------------- Tracking for plots ----------------
        self.episode_latencies = []  # Track latencies per episode
        self.episode_deviations = []  # Track deviations per episode

        latency_log_path = Path(constants.training_log_folder + "/plots") / ".." / "latency_log.txt"
        latency_log_path = latency_log_path.resolve()
        with open(latency_log_path, "w") as f:
            f.write("LATENCY LOG\n")
            f.write("=" * 100 + "\n")
            f.write(f"{'[EP | STEP]':<20} {'Req':<12} {'Type':<6} {'Servers':<15} "
                    f"{'Latency(ms)':<14} {'QueueWait(ms)':<16} {'Total(ms)':<12} "
                    f"{'D1':<8} {'D2':<8} {'Satisfaction':<12} Arrival\n")
            f.write("=" * 100 + "\n")
    # ------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------
    def log_latency_to_file(self, request, observed_latency, action_subset, episode, step):
        """
        Append latency + request details to a structured text log file.
        """
        log_path = self.plot_dir.parent / "latency_log.txt"

        try:
            combination = getattr(request, 'combination', ['?'])
            request_id = getattr(request, 'request_id', 'unknown')
            arrival = getattr(request, 'arrival_time', 'N/A')
            queue_wait = getattr(request, 'queue_waiting_time', -1.0)

            # Deadline thresholds for reference
            D1, D2 = self.deadlines[0] if combination == "s" else self.deadlines[1]

            # Satisfaction label
            if observed_latency <= D1:
                satisfaction = "FULL"
            elif observed_latency <= D2:
                satisfaction = "PARTIAL"
            else:
                satisfaction = "MISSED"

            with open(log_path, "a") as f:
                f.write(
                    f"[EP {episode:04d} | STEP {step:04d}] "
                    f"Req={request_id} | "
                    f"Type={combination} | "
                    f"Servers={list(action_subset)} | "
                    f"Latency={observed_latency:.3f} ms | "
                    f"QueueWait={queue_wait:.3f} ms | "
                    f"Total={observed_latency + queue_wait:.3f} ms | "
                    f"D1={D1} ms | D2={D2} ms | "
                    f"Satisfaction={satisfaction} | "
                    f"Arrival={arrival}\n"
                )
        except Exception as e:
            print(f"[CONTROLLER] ⚠️ Failed to write latency log: {type(e).__name__} - {e}")

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
                    print(f"[CONTROLLER, ASSIGN] Req {request.request_id} → Server {i + 1} ({proc:.3f}s)")
                else:
                    print(f"[CONTROLLER, BUSY] Server {i + 1} full for Req {request.request_id}")

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

        # Decide which servers to evaluate (0-based)
        if action_subset is None:
            server_indices = range(num_servers)
        else:
            server_indices = action_subset
            
            try:
            
                ########################################################################################################
                # Track P(T) values for satisfaction calculation, to be used in episodic reward. 
                # Calculated once per request after processing is complete 
                # (i.e. after action is taken and request is scheduled on selected servers).
                # 
                # For each request with completion time T:
                #
                #     P(T) = 1                      if T <= D1        (fully satisfied)
                #     P(T) = (T - D1) / (D2 - D1)   if D1 < T <= D2   (linearly decreasing satisfaction)
                #     P(T) = 0                      if T > D2         (not satisfied)
                #
                # where:
                #     D1 = soft deadline
                #     D2 = hard deadline
                #     T = minimum observed completion time for this request among all the servers it was assigned to
                #         + total waiting time in the queue before processing starts
                #######################################################################################################
                
                # D1 = request.deadline[0]  # soft deadline in ms
                # D2 = request.deadline[1]  # hard deadline in ms
                
                # request combination type
                combination = request.combination[0]
                
                if(combination == "s"):
                    D1, D2  = self.deadlines[0]  # soft and hard deadlines for "s" type requests
                else:
                    D1, D2 = self.deadlines[1]   # soft and hard deadlines for "d" or "p" type requests
                
                
                # minimum observed completion time for this request among all the servers it was assigned to
                delays = request.total_processing_delay
                valid_delays = delays[delays >= 0]
                # keep only valid (processed) delays (removing -1s)
                min_observed_completion_time = ( np.min(valid_delays) if valid_delays.size > 0 else 0.0 )

                # total waiting time in the queue before processing starts
                total_queue_waiting_time = request.queue_waiting_time
                
                # Total time = completion time + queue waiting time (in ms)
                T = min_observed_completion_time + total_queue_waiting_time
                
                print(f"[CONTROLLER, P(T) CALC] Req {request.request_id}: T={T:.2f} ms, D1={D1} ms, D2={D2} ms")
                if T <= D1:
                    self.average_P_T_values.append(1)
                elif D1 < T <= D2:
                    P_T = (T - D1) / (D2 - D1)
                    self.average_P_T_values.append(P_T)
                else:                    
                    self.average_P_T_values.append(0)
                    
                #######################################################################################################
                
                # Track request completion counts for different combination types (for use in episodic reward shaping)
                if(combination == "s"):
                    self.request_s_total += 1
                    # if T <= D2:
                    #     self.request_s_done += 1
                elif(combination == "d"):
                    self.request_d_total += 1
                    # if T <= D2:
                    #     self.request_d_done += 1
                elif(combination == "p"):
                    self.request_p_total += 1
                    # if T <= D2:
                    #     self.request_p_done += 1
                        
                #######################################################################################################
            
            except Exception as e:
                print(f"[CONTROLLER, !] Failed to compute P(T) for satisfaction tracking: {type(e).__name__} - {e}")
                print(request.deadline)
                self.average_P_T_values.append(0)  # Default to 0 if we can't compute it

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

                cpu_cores_utilised = np.sum(d["cpu_usage"]) / 100.0
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

                reward = np.log(product + 1.0)
                # print(f"[CONTROLLER]...[STEP REWARD] Server {server_idx}: Reward={reward:.6f}")

                # print()
                # print()

                # Final sanity check
                if not np.isfinite(reward):
                    raise ValueError(
                        f"[CONTROLLER]...[STEP REWARD][ERROR] Non-finite reward computed for server {server_idx}")

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

        #############################################################
        # Discounted sum of step rewards
        #############################################################
        discounted_step_rewards = sum(
            (gamma ** i) * r for i, r in enumerate(self.step_rewards)
        )
        
        ##############################################################################################
        # Degree of satisfaction ω
        ##############################################################################################
        # Degree of satisfaction ω is computed using a deadline-based piecewise function P(T)
        #
        # The overall satisfaction ω is the average satisfaction over all requests:
        #
        # omega (ω) = (sum of P(T) over all requests) / (total number of expected requests in episode)
        ##############################################################################################
        
        # sum of P(T) over all requests
        average_P_T = sum(self.average_P_T_values)
        
        # total number of expected requests in episode
        total_requests_in_episode = constants.total_no_request / constants.no_of_episodes
        
        # average satisfaction ω for this episode      
        omega = average_P_T / total_requests_in_episode if total_requests_in_episode > 0 else 0.0
        
        
        #################################################################################
        # Average waiting time across the episode
        #################################################################################
        
        avg_waiting_time = sum(self.average_waiting_times) / len(self.average_waiting_times) if self.average_waiting_times else 0.0
        
        # sum of (Percentage of s done) * D1 + (Percentage of d done) * D1 + (Percentage of p done) * D1
        wait_time_denominator = 0.0
        if self.request_s_total > 0:
            wait_time_denominator += (self.request_s_done/self.request_s_total)*100
        if self.request_d_total > 0:
            wait_time_denominator += (self.request_d_done/self.request_d_total)*30
        if self.request_p_total > 0:
            wait_time_denominator += (self.request_p_done/self.request_p_total)*30

        #################################################################################

        # Episodic reward
        episodic_reward = (discounted_step_rewards + omega - avg_waiting_time/wait_time_denominator) if wait_time_denominator > 0 else (discounted_step_rewards + omega)

        print(f"[CONTROLLER, EPISODE REWARD] Discounted steps: {discounted_step_rewards:.3f}, "
              f"Satisfaction: {omega:.3f}, Avg wait: {avg_waiting_time:.3f} ms, "
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
            num_free = sum(l != -1 for l in load)
        except Exception as e:
            print(f"[CONTROLLER, !] Failed to find free servers: {type(e).__name__} - {e}")
            return

        if num_free == 0:
            print("[CONTROLLER, QUEUE] All servers busy. Retrying shortly.")
            time.sleep(0.1)
            # try once more
            self.process_step(request)
            return

        # update request state
        try:
            request.load = np.array(load)
            request_total_delay = []
            combined_strs = []
        except Exception as e:
            print(f"[CONTROLLER, !] Failed to update request state: {type(e).__name__} - {e}")
            return

        for i in range(self.num_servers):
            try:
                delay, combined_str = self.server_list[i].compute_request_time(request)
                request_total_delay.append(delay)
                combined_strs.append(combined_str)
            except Exception as e:
                print(
                    f"[CONTROLLER, !] Error computing request time "
                    f"for server {i + 1}: {type(e).__name__} - {e}"
                )
                request_total_delay.append(float(-1))  # Use -1 to indicate failure to compute delay for this server
                combined_strs.append(None)
        
        # Store estimated total_delay in request for each server for prediction of RL model
        request.total_processing_delay = np.array(request_total_delay)
        request_total_delay = [float(-1)] * self.num_servers # reset to later use for tracking observed latency after action is taken (i.e. when request is scheduled and completed)

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

        # track the waiting time in the queue before request starts processing
        try:
            request.queue_waiting_time = time.time() * 1000.0 - request.arrival_time  # in ms
            self.average_waiting_times.append(request.queue_waiting_time) 
        except Exception as e:
            print(f"[CONTROLLER, !] Failed to compute waiting time: {type(e).__name__} - {e}")
            request.queue_waiting_time = 0.0  
        l = []
        # assign request
        for i in action_subset:
            try:
                _, _, processing_time, combined_str = self.server_list[i].schedule_request(request)
                # Update observed processing time for this server (in ms)
                request_total_delay[i] = float(processing_time) * 1000.0
                l.append([request_total_delay[i], combined_str])
                
            except Exception as e:
                print(
                    f"[CONTROLLER, !] Failed to schedule request on server {i}: "
                    f"{type(e).__name__} - {e}"
                )
                print(processing_time)

        # Store actual total_delay in request for each server for reward calculation and tracking
        request.total_processing_delay = np.array(request_total_delay)
        
        # compute reward and track latency metrics
        try:
            final_step_reward_list = self.compute_step_reward(request, action_subset)
            combined_step_reward = np.mean(final_step_reward_list)

            # Track observed latency (minimum among selected servers)
            if len(action_subset) > 0 and hasattr(request, 'total_processing_delay'):
                observed_latency = sorted(l)[0][0]
                request.combination = sorted(l)[0][1]
                self.episode_latencies.append(observed_latency)

                # Track deviation from median
                deviation = abs(observed_latency - self.agent.median_computation_delay)
                self.episode_deviations.append(deviation)
                self.log_latency_to_file(
                    request=request,
                    observed_latency=observed_latency,
                    action_subset=action_subset,
                    episode=self.current_episode,
                    step=self.current_step
                )

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
        4. Generate plots (periodically)
        """
        print(f"\n{'=' * 60}")
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

        # Transfer episode metrics to agent for plotting
        if len(self.episode_latencies) > 0:
            self.agent.latencies = np.append(self.agent.latencies, self.episode_latencies)
        if len(self.episode_deviations) > 0:
            self.agent.deviations = np.append(self.agent.deviations, self.episode_deviations)

        episode_end_time = time.time()
        episode_duration = episode_end_time - self.episode_start_time

        print(
            f"[CONTROLLER, EPISODE {self.current_episode}] "
            f"Time taken: {episode_duration:.2f} seconds"
        )
        self.episode_start_time = None

        # ============================================================
        # PLOTTING: Generate plots periodically and at the end
        # ============================================================
        should_plot = (
                (self.current_episode % self.plot_every_n_episodes == 0) or  # Every N episodes
                (self.current_episode + 1 >= self.expected_episodes)  # Final episode
        )


        if should_plot:
            self.generate_plots()

        # Checkpoint every N episodes
        # if self.current_episode % 10 == 0:
        #     self.save_checkpoint()

        # Reset episode-level tracking
        self.step_experiences = []
        self.step_rewards = []
        self.episode_waiting_times = []
        self.episode_latencies = []  # Clear latency tracking
        self.episode_deviations = []  # Clear deviation tracking
        self.current_chunk = 0
        self.current_step = 0
        self.current_episode += 1
        self.average_P_T_values = []  # Clear P(T) tracking for next episode
        self.average_waiting_times = []  # Clear waiting time tracking for next episode
        self.request_s_done = 0
        self.request_s_total = 0
        self.request_d_done = 0
        self.request_d_total = 0
        self.request_p_done = 0
        self.request_p_total = 0
        if self.current_episode >= self.expected_episodes:
            print("[CONTROLLER] ✅ All training episodes finished.")
            # Generate final comprehensive plots
            self.generate_final_plots()
            self.training_done.set()

        print(f"{'=' * 60}\n")

    def generate_plots(self):
        """
        Generate all training plots and save metrics summary.
        Called periodically during training.
        """
        episode_str = f"ep{self.current_episode:04d}"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_dir = self.plot_dir / episode_str

        try:
            # 1. Save comprehensive metrics plot
            print(f"[CONTROLLER] 📊 Generating plots for episode {self.current_episode}...")
            self.agent.plot_all_metrics(
                save_dir=save_dir,
                show_plot=False  # Don't block execution with plots
            )

            # 2. Save focused loss plot
            self.agent.plot_training_loss(
                save_path=save_dir / f"loss_{timestamp}.png",
                show_plot=False
            )

            # 3. Save metrics summary text file
            self.agent.save_metrics_summary(
                save_dir / f"metrics_summary_{timestamp}.txt"
            )

            print(f"[CONTROLLER] ✅ Plots saved to {save_dir}")

        except Exception as e:
            print(f"[CONTROLLER] ⚠️ Failed to generate plots: {type(e).__name__} - {e}")

    def generate_final_plots(self):
        """
        Generate final comprehensive plots at the end of training.
        These are higher quality and include all data.
        """
        print("\n" + "=" * 60)
        print("[CONTROLLER] 🎨 Generating final training visualizations...")
        print("=" * 60)

        final_dir = self.plot_dir / "final"
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        try:
            # 1. Comprehensive metrics dashboard
            self.agent.plot_all_metrics(
                save_dir=final_dir,
                show_plot=False
            )

            # 2. High-res loss curves
            self.agent.plot_training_loss(
                save_path=final_dir / f"final_loss_{timestamp}.png",
                show_plot=False
            )

            # 3. Detailed metrics summary
            self.agent.save_metrics_summary(
                final_dir / f"final_summary_{timestamp}.txt"
            )

            # 4. Export training data to CSV for external analysis
            self.export_training_data(final_dir / f"training_data_{timestamp}.csv")

            print(f"[CONTROLLER] ✅ Final plots saved to {final_dir}")
            print("=" * 60 + "\n")

        except Exception as e:
            print(f"[CONTROLLER] ⚠️ Failed to generate final plots: {type(e).__name__} - {e}")

    def export_training_data(self, filepath):
        """
        Export all training metrics to CSV for external analysis.
        """
        try:
            # Determine the maximum length
            max_len = max(
                len(self.agent.loss),
                len(self.agent.val_loss),
                len(self.agent.epsilon_curve),
                len(self.agent.rewards),
                len(self.agent.latencies),
                len(self.agent.deviations),
                len(self.agent.episode_access_rate),
                len(self.agent.exploit_or_explore),
                len(self.agent.prediction_times)
            )

            # Helper to pad arrays
            def pad_to_length(arr, length, fill_value=np.nan):
                if len(arr) < length:
                    if isinstance(arr, np.ndarray):
                        return np.concatenate([arr, np.full(length - len(arr), fill_value)])
                    else:
                        return list(arr) + [fill_value] * (length - len(arr))
                return arr

            # Create DataFrame
            data = {
                "episode": range(max_len),
                "loss": pad_to_length(self.agent.loss, max_len),
                "val_loss": pad_to_length(self.agent.val_loss, max_len),
                "epsilon": pad_to_length(self.agent.epsilon_curve, max_len),
                "reward": pad_to_length(self.agent.rewards, max_len),
                "latency": pad_to_length(self.agent.latencies, max_len),
                "deviation": pad_to_length(self.agent.deviations, max_len),
                "access_rate": pad_to_length(self.agent.episode_access_rate, max_len),
                "strategy": pad_to_length(self.agent.exploit_or_explore, max_len, fill_value=""),
                "prediction_time": pad_to_length(self.agent.prediction_times, max_len),
            }

            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            print(f"[CONTROLLER] 💾 Training data exported to {filepath}")

        except Exception as e:
            print(f"[CONTROLLER] ⚠️ Failed to export training data: {type(e).__name__} - {e}")

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
        - EPISODE = processing N chunks (chunk = list of requests)
        - Training happens ONCE per episode
        """
        try:
            print("[CONTROLLER] Starting main control loop with step/episodic rewards.")
            if self.current_chunk == 0:
                self.episode_start_time = time.time()
            print(
                f"[CONTROLLER] Processing chunk {self.current_chunk + 1}/"
                f"{self.chunks_per_episode}"
            )

            # Validate chunk
            if chunk is None:
                self.episode_start_time = None
                raise ValueError("[CONTROLLER, !] Received None chunk")

            if not hasattr(chunk, "__iter__"):
                self.episode_start_time = None
                raise TypeError(f"[CONTROLLER, !] Chunk is not iterable: {type(chunk)}")

            for request in chunk:
                try:
                    with self.lock:
                        # Process this chunk as one STEP
                        
                        self.process_step(request)
                        
                        # request combination type
                        combination = request.combination[0]
                        # Track request completion counts for different combination types (for use in episodic reward shaping)
                        if(combination == "s"):
                            self.request_s_done += 1
                        elif(combination == "d"):
                            self.request_d_done += 1
                        elif(combination == "p"):
                            self.request_p_done += 1
                            
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
                    self.episode_start_time = None
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