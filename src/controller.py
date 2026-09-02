import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

import constants
import servers
from agent import DQNAgent, request_to_state_array
from policy_registry import PolicyContext, subset_to_index  # [SAFETAIL][SEAM] stdlib-only, always present

MAX_CONCURRENT_REQUESTS = servers.MAX_CONCURRENT_REQUESTS

# [SAFETAIL][REPLAY][AUDIT][D-04] When SAFETAIL_AUDIT_REPLAY=1, every stored
# transition is also appended here with the realised per-server delays of the
# same step, so tools/audit_replay.py (gate G2) can assert the state carries no
# outcome. Off by default -- zero cost in normal runs.
AUDIT_TRANSITIONS: list = []
_AUDIT_REPLAY = os.environ.get("SAFETAIL_AUDIT_REPLAY", "0") not in ("0", "", "false")


class Controller:
    def __init__(self, num_servers=5, steps_per_episode=10, chunks_per_episode=3):
        self.access_rate_log = []  # [(episode, avg_access_rate)]
        self.testing_phase_active = False
        self.testing_request_limit = 3000
        self.testing_request_count = 0
        self.testing_rewards = []
        self.testing_latencies = []

        self.num_servers = num_servers
        self.server_list = [servers.Server(i + 1) for i in range(num_servers)]

        self.lock = threading.Lock()
        self.receiver_queue = None  # attached externally

        # Values for D1 and D2 based on the combination type of the request (e.g., "s" or "d" or "p").
        self.deadlines = np.asarray(constants.deadlines)  # in ms

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

        # [SAFETAIL][CONTROLLER][D-21] requests dropped after saturation retries
        self.dropped_requests = 0

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

        if constants.testing_phase_active:
            self.agent.model = load_model(constants.saved_model_path)
            print("model loaded")
            self.agent.epsilon = 0.0

        # ---------------- Tracking for plots ----------------
        self.episode_latencies = []  # Track latencies per episode
        self.episode_deviations = []  # Track deviations per episode

        # ---------------- Post-epsilon-min phase ----------------
        self.epsilon_min_reached = False  # flag: epsilon has hit its floor
        self.post_epsilon_steps = 0  # steps counted after epsilon_min reached
        self.post_epsilon_steps_target = constants.post_epsilon_steps  # run this many steps before saving + testing
        self.testing_phase_active = constants.testing_phase_active  # flag: we are now in the testing phase

        # ── BASELINE MODE ────────────────────────────────────────────────────────
        # Change this string to match whichever run you are doing.
        # Options: "safetail" | "minload_1" | "minload_2" | "minload_3"
        #          | "minprop_1" | "minprop_2" | "minprop_3"
        #          | "rand_1" | "rand_2" | "rand_3"
        self.BASELINE_MODE = constants.BASELINE_MODE
        # ─────────────────────────────────────────────────────────────────────

        # [SAFETAIL][SEAM][M-02] Optional external policy (plan.md 8.3).
        # With baselines/ absent, constants.POLICY stays "native" and this is None;
        # the DQN / heuristic paths below are unconditional and survive deletion.
        self.policy = None
        self._pending_policy_ctx = None
        _pol = getattr(constants, "POLICY", "native")
        if _pol not in (None, "", "native"):
            from policy_registry import get  # stdlib-only module, always importable
            self.policy = get(_pol)()         # KeyError here is a loud, correct failure
            print(f"[SAFETAIL][SEAM] external policy active: {_pol} -> {type(self.policy).__name__}")
        base_log_dir = Path(constants.training_log_folder)
        base_log_dir.mkdir(parents=True, exist_ok=True)

        self.latency_log_path = base_log_dir / f"latency_log.csv"
        self.request_access_log_path = base_log_dir / f"request_wise_access_log.csv"
        self.latency_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.request_access_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.step_reward_log_path = base_log_dir / "step_rewards.csv"

        self.episode_reward_log_path = base_log_dir / "episode_rewards.csv"

        with open(self.episode_reward_log_path, "w") as f:
            f.write("episode,episodic_reward\n")

        with open(self.step_reward_log_path, "w") as f:
            f.write("episode,step,reward\n")

        with open(self.request_access_log_path, "w") as f:
            f.write("request_id,request_type,access_rate\n")
        header = (
            "request_id,request_type,contention_str,computation_delay,propagation_delay,"
            "transmission_delay,queueing_delay,total_latency,message_size_kb,end_to_end_latency\n"
        )
        with open(self.latency_log_path, "w") as f:
            f.write(header)
        if self.testing_phase_active:
            self.load_existing_plot_history()

            # boundary marker for plots
            self.agent.testing_start_index = len(self.agent.rewards)

    def log_request_access_rate_with_type(self, request_id, request_type, access_rate):
        try:
            row = f"{request_id},{request_type},{access_rate:.6f}\n"
            with open(self.request_access_log_path, "a") as f:
                f.write(row)
        except Exception as e:
            print(f"[CONTROLLER] Failed to log request access rate: {e}")

    # ------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------
    def log_latency_to_csv(
            self,
            request,
            computation_delay,
            propagation_delay,
            transmission_delay,
            queueing_delay,
            total_latency,
    ):
        """
        Logs latency components per request into CSV.
        """

        try:
            request_id = getattr(request, "request_id", "unknown")
            # [SAFETAIL][FIX][D-19] log the immutable type letter, never the
            # contention string. `combination` is no longer mutated (D-19), but
            # slice defensively so this column can only ever be s|d|p|?.
            request_type = str(getattr(request, "combination", "?"))[:1] or "?"
            contention = str(getattr(request, "contention_str", "") or "")
            row = (
                f"{request_id},"
                f"{request_type},"
                f"{contention},"
                f"{computation_delay:.6f},"
                f"{propagation_delay:.6f},"
                f"{transmission_delay:.6f},"
                f"{queueing_delay:.6f},"
                f"{total_latency:.6f},"
                # [SAFETAIL][METRIC][FIX][D-11][D-34] payload size (drives
                # transmission after B5) and END-TO-END latency = service
                # (== total_latency) + the queue wait the reward penalises.
                # Plot one deliberately; never plot a quantity the reward
                # ignores. Decision 13.1(2).
                f"{float(getattr(request, 'message_size', 0) or 0):.1f},"
                f"{(float(total_latency) + float(queueing_delay)):.6f}\n"
            )
            with open(self.latency_log_path, "a") as f:
                f.write(row)

        except Exception as e:
            print(f"[CONTROLLER] ⚠️ CSV logging failed: {type(e).__name__} - {e}")

    def _safe_read_csv(self, path):
        try:
            p = Path(path)
            if p.exists() and p.stat().st_size > 0:
                return pd.read_csv(p)
        except Exception as e:
            print(f"[CONTROLLER] Failed reading {path}: {e}")
        return pd.DataFrame()

    def load_existing_plot_history(self):
        """
        Load previous results and set testing-start indices correctly.
        """

        base_dir = Path(constants.original_training_log_folder)

        reward_csv = base_dir / "episode_rewards.csv"
        access_csv = base_dir / f"request_wise_access_log.csv"
        latency_csv = base_dir / f"latency_log.csv"
        # ---------------- Rewards ----------------
        if reward_csv.exists():
            df = pd.read_csv(reward_csv)
            self.agent.rewards = df["episodic_reward"].dropna().values

        # ---------------- Access -----------------
        if access_csv.exists():
            df = pd.read_csv(access_csv)
            self.agent.episode_access_rate = df["access_rate"].dropna().values

        # ---------------- Latency ----------------
        if latency_csv.exists():
            df = pd.read_csv(latency_csv)
            self.agent.latencies = df["total_latency"].dropna().values
            self.agent.deviations = abs(
                self.agent.latencies - constants.median_computation_delay * 1000
            )

        # =====================================================
        # IMPORTANT:
        # Since old history = training history,
        # testing starts AFTER loaded dataset
        # =====================================================
        self.agent.testing_start_reward_index = len(self.agent.rewards)
        self.agent.testing_start_latency_index = len(self.agent.latencies)
        self.agent.testing_start_access_index = len(self.agent.episode_access_rate)
        self.agent.testing_start_epsilon_index = len(self.agent.epsilon_curve)
        self.agent.testing_start_loss_index = len(self.agent.loss)
        self.agent.testing_start_deviation_index = len(self.agent.deviations)

        print("[CONTROLLER] Loaded plot history:")
        print("Rewards :", len(self.agent.rewards))
        print("Latency :", len(self.agent.latencies))
        print("Access  :", len(self.agent.episode_access_rate))

    def find_free_servers(self):
        now = time.time()
        load = [s.check_server_availability(now) for s in self.server_list]
        return np.array(load)

    def _build_policy_context(self, request, load, other_latencies):
        """
        [SAFETAIL][SEAM] Assemble the frozen, pre-action PolicyContext (plan.md 8.4).

        `load[i]` is check_server_availability(): current active-request count when
        the server has room, else -1 (full). free_slots is derived from it.
        Per-server estimates come from the SINGLE phase-2 draw already stored on
        `request.total_processing_delay` / `other_latencies` (guards against the
        D-18 double-draw for policies that use the seam).
        """
        beta = self.num_servers
        load = list(load)
        free_slots = [(-1 if l == -1 else int(MAX_CONCURRENT_REQUESTS - l)) for l in load]

        est_delay = []
        est_components = []
        tpd = getattr(request, "total_processing_delay", None)
        for i in range(beta):
            comp = prop = trans = 0.0
            if i < len(other_latencies) and other_latencies[i]:
                comp = float(other_latencies[i].get("computation", 0.0))
                prop = float(other_latencies[i].get("propagation", 0.0))
                trans = float(other_latencies[i].get("transmission", 0.0))
            est_components.append((comp, prop, trans))
            if tpd is not None and i < len(tpd):
                est_delay.append(float(tpd[i]))
            else:
                est_delay.append(comp + prop + trans)

        static, dynamic = [], []
        for i, srv in enumerate(self.server_list):
            d = request.server_dicts[i] if i < len(request.server_dicts) else {}
            static.append({
                "total_ram": d.get("total_ram", 0.0),
                "total_cpu_cores": d.get("total_cpu_cores", 0),
                "total_gpu_memory": d.get("total_gpu_memory", 0.0),
                "has_gpu": bool(d.get("total_gpu_memory", 0.0)),
            })
            dynamic.append({
                "active_requests": int(getattr(srv, "num_requests", 0)),
                "ram_usage": d.get("ram_usage", 0.0),
                "gpu_usage": d.get("gpu_usage", 0),
            })

        rtype = "?"
        try:
            rtype = str(request.combination)[0]
        except Exception:
            pass
        try:
            if str(request.combination)[0] == "s":
                deadline = tuple(self.deadlines[0])
            else:
                deadline = tuple(self.deadlines[1])
        except Exception:
            deadline = (float("nan"), float("nan"))

        return PolicyContext(
            request_id=int(getattr(request, "request_id", -1)),
            request_type=rtype,
            deadline=(float(deadline[0]), float(deadline[1])),
            message_size=float(getattr(request, "message_size", 0.0)),
            bandwidth=float(getattr(request, "bandwidth", 0.0)),
            beta=beta,
            free_slots=free_slots,
            est_delay=est_delay,
            est_components=est_components,
            server_static=static,
            server_dynamic=dynamic,
            arrival_time=float(getattr(request, "arrival_time", 0.0)),
            episode_index=int(self.current_episode),
            step_index=int(self.current_step),
        )

    # ── Baseline server selection helpers ────────────────────────────────────

    def _select_minload_servers(self, x):
        """Return indices of the x servers with the fewest active requests."""
        loads = [(s.num_requests, i) for i, s in enumerate(self.server_list)]
        loads.sort(key=lambda t: t[0])
        return [i for _, i in loads[:x]]

    def _select_minprop_servers(self, x):
        """Return indices of the x servers with the lowest propagation delay."""
        prop_delays = [(s._get_propogation_delay(), i) for i, s in enumerate(self.server_list)]
        prop_delays.sort(key=lambda t: t[0])
        return [i for _, i in prop_delays[:x]]

    def _select_rand_servers(self, x):
        """Return indices of x randomly selected servers."""
        import random
        indices = list(range(self.num_servers))
        return random.sample(indices, min(x, self.num_servers))

    # ─────────────────────────────────────────────────────────────────────────

    def get_queue_lengths(self):
        # [SAFETAIL][DEAD][D-29] orphan -- never called. Also misnamed: there are
        # no queues (D-24); this returns active-request counts. Kept as a
        # read-only accessor for tooling.
        return np.array([len(s.active_requests) for s in self.server_list])

    # [SAFETAIL][FIX][D-28] `assign_request(request, indices)` DELETED here.
    # It unpacked 3 values from schedule_request's 7-tuple (`ok, finish, proc =
    # ...`) and would raise on any call. It was unreachable dead code, and a
    # broken orphan invites someone to "fix" it into the live path. The live
    # scheduling loop is in process_step(). `dispatch_to_agent` (a one-line
    # wrapper around agent.get_action) is removed with it -- also unused.

    # ------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------
    def compute_step_reward(self, request, action_subset=None):
        """
        Step reward computed PER REQUEST (per server).

        Implemented formula (see S-03 -- this is the ONE authoritative statement):

            headroom_i = geometric_mean( available (1 - u) factors )
            R_i        = log(1 + headroom_i)            in [0, log 2]

        Utilisation factors u:
            cm = ram_usage / total_ram
            cu = (sum cpu_core_% / 100) / total_cpu_cores
            gm = gpu_memory / total_gpu_memory      } only when the server HAS a GPU
            gu = gpu_usage / 100                    }   (D-16)

        [SAFETAIL][REWARD][FIX][D-16] CPU-only servers (no GPU columns in their
        dataset CSV -- servers 3 and 4) previously got (1-0)*(1-0)=1 for the two
        GPU factors, structurally inflating their headroom. Now the GPU factors
        are DROPPED for such servers and the product is renormalised via a
        geometric mean so a 2-factor and a 4-factor server are on one scale.

        [SAFETAIL][REWARD][FIX][D-27] docstring now states the formula that is
        actually computed (was: "1 + log(exp(...)-1)").

        NOTE: this reward is still monotone non-decreasing in |A| once the
        controller collapses it with a mean -- pricing redundancy is D-07/M-04,
        fixed in B3, not here.

        If action_subset is None -> compute for all servers, else only those.
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

                if (combination == "s"):
                    D1, D2 = self.deadlines[0]  # soft and hard deadlines for "s" type requests
                else:
                    D1, D2 = self.deadlines[1]  # soft and hard deadlines for "d" or "p" type requests

                # minimum observed completion time for this request among all the servers it was assigned to
                delays = request.total_processing_delay
                valid_delays = delays[delays >= 0]
                # keep only valid (processed) delays (removing -1s)
                min_observed_completion_time = (np.min(valid_delays) if valid_delays.size > 0 else 0.0)

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

                # [SAFETAIL][REWARD][FIX][D-20] completion counts are DEADLINE-
                # CONDITIONAL again. Previously *_total was set here but *_done was
                # incremented unconditionally in send_to_server(), so the
                # completion ratio in the episodic wait-time denominator was
                # identically 1.0 by construction. `_done` is now gated on T<=D2
                # here, at the one place T is known, and the unconditional
                # increments in send_to_server() are removed.
                _met_deadline = (T <= D2)
                if combination == "s":
                    self.request_s_total += 1
                    self.request_s_done += int(_met_deadline)
                elif combination == "d":
                    self.request_d_total += 1
                    self.request_d_done += int(_met_deadline)
                elif combination == "p":
                    self.request_p_total += 1
                    self.request_p_done += int(_met_deadline)

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

                cpu_cores_utilised = np.sum(d["cpu_usage"]) / 100.0
                cu = cpu_cores_utilised / max(float(d["total_cpu_cores"]), 1e-8)

                # [SAFETAIL][REWARD][FIX][D-16] GPU factors only for GPU servers.
                has_gpu = float(d.get("total_gpu_memory", 0.0) or 0.0) > 0.0
                factors = [1.0 - cm, 1.0 - cu]
                if has_gpu:
                    total_gpu_mem_mb = float(d["total_gpu_memory"]) * GB_TO_MB
                    gm = d["gpu_memory"] / max(total_gpu_mem_mb, 1e-8)
                    gu = float(d["gpu_usage"]) / 100.0
                    factors += [1.0 - gm, 1.0 - gu]

                # clip each factor into [0,1] (utilisation can momentarily exceed
                # capacity in the trace) then take the geometric mean so servers
                # with 2 vs 4 factors are comparable.
                factors = [min(1.0, max(0.0, f)) for f in factors]
                headroom = float(np.prod(factors)) ** (1.0 / len(factors))

                # ---- Reward ----  R in [0, log 2]  (S-03)
                reward = float(np.log1p(headroom))

                if not np.isfinite(reward):
                    raise ValueError(
                        f"[CONTROLLER][REWARD][INVARIANT][D-17] non-finite reward for server {server_idx}")

                rewards[i] = reward

            except Exception as e:
                # [SAFETAIL][REWARD][FIX][D-17] zero THIS server's slot (was
                # rewards[server_idx-1], which corrupted a neighbour and left
                # server 0 at its initialised value). server_idx is validated
                # 0 <= server_idx < num_servers above.
                rewards[server_idx] = 0.0
                print(
                    f"[CONTROLLER][REWARD][D-17] step-reward failure on server {server_idx}: "
                    f"{type(e).__name__} - {e}"
                )

        return rewards

    def _collapse_step_reward(self, per_server_rewards, action_subset):
        """
        [SAFETAIL][REWARD][FIX][D-07][M-04][S-02] Collapse the per-server headroom
        rewards to one scalar AND price redundancy.

            R_step = (1/|A|) * sum_{i in A} log(1 + headroom_i)
                     - c_red * (|A| - 1) / (beta - 1)

        Two changes vs the old `np.mean(per_server_rewards)`:
          * the mean is over the SELECTED servers |A|, not a constant 6 (the
            phantom slot from server_dicts length 6 -- W-02 -- also drops out);
          * an explicit cost, linear in the redundancy count, subtracts from the
            reward. c_red = 0 reproduces the old (un-priced) behaviour.

        S-02: this is NOT HED's `Sum(l_i - l_bar) - W_step` term (that was
        sign-inverted for load balancing). BTP correctly replaced the load term
        with the headroom product but dropped every |A|-sensitive quantity;
        `c_red` restores redundancy pricing without the HED sign bug.
        """
        idx = [int(i) for i in action_subset] if action_subset is not None else \
              list(range(len(per_server_rewards)))
        idx = [i for i in idx if 0 <= i < len(per_server_rewards)]
        if not idx:
            return 0.0
        mean_headroom = float(np.mean([per_server_rewards[i] for i in idx]))
        k = len(idx)
        beta = max(2, self.num_servers)
        c_red = float(getattr(constants, "C_RED", 0.0))
        penalty = c_red * (k - 1) / (beta - 1)
        return mean_headroom - penalty

    def _apply_tau_term(self, headroom_reward, request, action_subset, observed_latency_ms, mode):
        """
        [SAFETAIL][REWARD][FIX][B4][M-03] Fold the SafeTail 1.0 tau-referenced
        5-case reward into the step reward.

        `observed_latency_ms` is min realised latency over A, in ms; tau (per
        request type, from constants.TAU_BY_TYPE) is in seconds.
        """
        try:
            from rewards import tau_reward_5case
            letter = str(getattr(request, "combination", "d"))[:1]
            tau = constants.TAU_BY_TYPE.get(letter, constants.TAU_BY_TYPE.get("d", 0.05))
            obs_s = float(observed_latency_ms) / 1000.0
            r_tau, oob = tau_reward_5case(
                obs_s, tau, len(action_subset), self.num_servers, float(constants.alpha))
            if oob:
                self._tau_out_of_band = getattr(self, "_tau_out_of_band", 0) + 1
            return r_tau if mode == "tau" else (headroom_reward + r_tau)
        except Exception as e:
            print(f"[SAFETAIL][REWARD][B4] tau term failed: {type(e).__name__} - {e}")
            return headroom_reward

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

        avg_waiting_time = sum(self.average_waiting_times) / len(
            self.average_waiting_times) if self.average_waiting_times else 0.0

        # sum of (Percentage of s done) * D1 + (Percentage of d done) * D1 + (Percentage of p done) * D1
        wait_time_denominator = 0.0
        if self.request_s_total > 0:
            wait_time_denominator += (self.request_s_done / self.request_s_total) * self.deadlines[0][0]
        if self.request_d_total > 0:
            wait_time_denominator += (self.request_d_done / self.request_d_total) * self.deadlines[1][0]
        if self.request_p_total > 0:
            wait_time_denominator += (self.request_p_done / self.request_p_total) * self.deadlines[1][0]

        #################################################################################

        # Episodic reward (raw)
        # NOTE: avg_waiting_time is in ms — normalise to seconds before dividing
        # to prevent the wait penalty from dominating and exploding Q-targets.
        avg_waiting_time_s = avg_waiting_time / 1000.0  # ms → s
        episodic_reward_raw = (
            discounted_step_rewards + omega - avg_waiting_time_s / wait_time_denominator
            if wait_time_denominator > 0
            else discounted_step_rewards + omega
        )

        # Hard-clip to [-10, 10] so Q-targets never diverge to 10^16
        episodic_reward = float(np.clip(episodic_reward_raw, -10.0, 10.0))

        print(f"[CONTROLLER, EPISODE REWARD] Discounted steps: {discounted_step_rewards:.3f}, "
              f"Satisfaction: {omega:.3f}, Avg wait: {avg_waiting_time:.3f} ms, "
              f"R_raw: {episodic_reward_raw:.3f}, R_clipped: {episodic_reward:.3f}")

        return episodic_reward

    # [SAFETAIL][CONTROLLER][FIX][D-21] saturation retry: bounded, exp backoff,
    # explicit drop + counter. Was unbounded recursion (stack overflow under
    # sustained saturation).
    SATURATION_MAX_RETRIES = 6
    SATURATION_BACKOFF_BASE = 0.05   # s; 0.05, 0.1, 0.2, ... capped
    SATURATION_BACKOFF_CAP = 0.8

    def process_step(self, request):

        try:
            print(f"\n[CONTROLLER, STEP {self.current_step}] Processing request {request.request_id}.")
        except Exception:
            print(f"\n[CONTROLLER, STEP {self.current_step}] Processing request <unknown>.")

        num_free = 0
        load = None
        for attempt in range(self.SATURATION_MAX_RETRIES + 1):
            try:
                load = self.find_free_servers()
                num_free = sum(l != -1 for l in load)
            except Exception as e:
                print(f"[CONTROLLER, !] Failed to find free servers: {type(e).__name__} - {e}")
                return
            if num_free > 0:
                break
            if attempt == self.SATURATION_MAX_RETRIES:
                self.dropped_requests += 1
                print(
                    f"[SAFETAIL][CONTROLLER][SCHED][D-21] all servers saturated after "
                    f"{self.SATURATION_MAX_RETRIES} retries; DROPPING request "
                    f"{getattr(request, 'request_id', '?')} (total dropped={self.dropped_requests})"
                )
                return
            backoff = min(self.SATURATION_BACKOFF_BASE * (2 ** attempt), self.SATURATION_BACKOFF_CAP)
            # [SAFETAIL][CONTROLLER][FIX][D-23] this backoff IS the request's only
            # genuine wait (Erlang-B loss system, no real queue). Accumulate it as
            # a simulated quantity; do not rely on wall-clock later.
            try:
                request._saturation_wait_s = getattr(request, "_saturation_wait_s", 0.0) + backoff
            except Exception:
                pass
            print(f"[SAFETAIL][CONTROLLER][SCHED][D-21] all servers busy, "
                  f"retry {attempt + 1}/{self.SATURATION_MAX_RETRIES} in {backoff:.2f}s")
            time.sleep(backoff)

        try:
            request.load = np.array(load)

            request_total_delay = []
            combined_strs = []
            other_latencies = []
        except Exception as e:
            print(f"[CONTROLLER, !] Failed to update request state: {type(e).__name__} - {e}")
            return

        for i in range(self.num_servers):
            try:
                delay, combined_str, comp, prop, trans = \
                    self.server_list[i].compute_request_time(request)

                request_total_delay.append(delay)
                combined_strs.append(combined_str)
                other_latencies.append({
                    'computation': comp,
                    'propagation': prop,
                    "transmission": trans
                })
            except Exception as e:
                print(
                    f"[CONTROLLER, !] Error computing request time "
                    f"for server {i + 1}: {type(e).__name__} - {e}"
                )
                request_total_delay.append(float(-1))  # Use -1 to indicate failure to compute delay for this server
                combined_strs.append(None)
                other_latencies.append({})

        # Store estimated total_delay in request for each server for prediction of RL model
        request.total_processing_delay = np.array(request_total_delay)
        request_total_delay = [float(
            -1)] * self.num_servers  # reset to later use for tracking observed latency after action is taken (i.e. when request is scheduled and completed)

        # Update request state to add dataset from servers
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

        # ── Server selection: driven automatically by self.BASELINE_MODE ────────
        try:
            if self.policy is not None:
                # [SAFETAIL][SEAM][M-02] External policy path. Native branches below
                # are untouched and remain the only path when baselines/ is absent.
                ctx = self._build_policy_context(request, load, other_latencies)
                self._pending_policy_ctx = ctx
                subset = [int(i) for i in self.policy.select(ctx)]
                if not subset or not all(0 <= i < self.num_servers for i in subset):
                    raise ValueError(
                        f"[SAFETAIL][SEAM][INVARIANT] policy {self.policy.name!r} returned "
                        f"invalid subset {subset} for beta={self.num_servers}"
                    )
                action_subset = subset
                action_index = subset_to_index(subset, self.num_servers)
                try:
                    self.log_request_access_rate_with_type(
                        request_id=request.request_id,
                        request_type=str(request.combination),
                        access_rate=len(action_subset) / self.num_servers,
                    )
                except Exception as e:
                    print(f"[CONTROLLER] Access rate logging failed: {e}")

            elif self.BASELINE_MODE == "safetail":
                original_request_type = request.combination
                action_subset, action_index = self.agent.get_action(request)
                # [SAFETAIL][CONTROLLER][FIX][B6][D-09] --match-k: cap SafeTail's
                # subset to a target mean K so the headline comparison is
                # replication-budget controlled. Keeps the K servers with the
                # lowest phase-2 estimate; alternates floor/ceil(MATCH_K) so the
                # realised mean lands on the fractional target.
                mk = getattr(constants, "MATCH_K", None)
                if mk:
                    self._mk_steps = getattr(self, "_mk_steps", 0) + 1
                    self._mk_selected = getattr(self, "_mk_selected", 0)
                    # target keeps the running mean pinned to mk
                    target = int(round(float(mk) * self._mk_steps - self._mk_selected))
                    target = max(1, min(target, len(action_subset)))
                    if len(action_subset) > target:
                        tpd = getattr(request, "total_processing_delay", None)
                        order = sorted(action_subset,
                                       key=lambda i: (tpd[i] if tpd is not None and i < len(tpd) else 0.0))
                        action_subset = sorted(order[:target])
                        action_index = subset_to_index(action_subset, self.num_servers)
                    self._mk_selected += len(action_subset)
                # -------- ACCESS RATE PER REQUEST --------
                try:
                    access_rate = len(action_subset) / self.num_servers
                    self.log_request_access_rate_with_type(
                        request_id=request.request_id,
                        request_type=original_request_type,
                        access_rate=access_rate
                    )
                except Exception as e:
                    print(f"[CONTROLLER] Access rate logging failed: {e}")

            else:
                # [SAFETAIL][CONTROLLER][FIX][B6][D-09] generic {family}_{K}
                # dispatch. K now ranges 1..beta (was hardcoded 1..3), so the
                # baseline comparison can be run budget-matched to SafeTail's
                # mean K instead of capped below it.
                fam, _, k_str = self.BASELINE_MODE.partition("_")
                selectors = {
                    "minload": self._select_minload_servers,
                    "minprop": self._select_minprop_servers,
                    "rand": self._select_rand_servers,
                }
                if fam not in selectors or not k_str.isdigit():
                    raise ValueError(f"Unknown BASELINE_MODE: '{self.BASELINE_MODE}'")
                k = max(1, min(int(k_str), self.num_servers))
                action_subset = list(selectors[fam](k))
                action_index = subset_to_index(action_subset, self.num_servers) if action_subset else 0
        # ─────────────────────────────────────────────────────────────────────
        except Exception as e:
            print(f"[CONTROLLER, !] Agent failed to produce action: {type(e).__name__} - {e}")
            return

        # [SAFETAIL][REPLAY][FIX][D-04] Snapshot the state the agent ACTED ON,
        # as a flat array, BEFORE schedule_request() mutates the request with
        # realised delays / contention string / queue_waiting_time. The old code
        # stored the live Request object and only flattened it in
        # finalize_episode(), by which time it held the OUTCOME of its own action.
        s_t_arr = None
        _learning = (self.BASELINE_MODE == "safetail" and self.policy is None
                     and not self.testing_phase_active)
        if _learning:
            try:
                s_t_arr = request_to_state_array(request, self.agent.remove_nan_in_state)
            except Exception as e:
                print(f"[SAFETAIL][REPLAY][D-04] s_t snapshot failed: {type(e).__name__} - {e}")

        # [SAFETAIL][CONTROLLER][FIX][D-23] SIMULATED queue wait, not wall-clock.
        # Was `time.time()*1000 - arrival_time`: CSV lookups + regressor inference
        # + a TensorFlow forward pass + lock contention, i.e. a function of how
        # fast the test machine is, and it fed BOTH the reward penalty and T in
        # P(T). The servers reject when full (M/M/c/c, D-24), so the only genuine
        # wait is the D-21 saturation backoff this request incurred, plus a small
        # fixed controller-dispatch cost.
        try:
            if getattr(constants, "LEGACY_QUEUE_WAIT", False):
                # [SAFETAIL][LEGACY][D-23] pre-fix wall-clock elapsed
                sim_wait_ms = time.time() * 1000.0 - request.arrival_time
            else:
                sim_wait_ms = (getattr(request, "_saturation_wait_s", 0.0) * 1000.0
                               + float(constants.DISPATCH_COST_MS))
            request.queue_waiting_time = sim_wait_ms
            self.average_waiting_times.append(request.queue_waiting_time)
        except Exception as e:
            print(f"[CONTROLLER, !] Failed to compute waiting time: {type(e).__name__} - {e}")
            request.queue_waiting_time = 0.0
        l = []
        # assign request
        for i in action_subset:
            try:
                _, finish_time, processing_time, combined_str, computation_delay_for_node, propagation_delay_for_node, tramission_delay_for_node = \
                    self.server_list[i].schedule_request(request)
                # Update observed processing time for this server (in ms)
                request_total_delay[i] = float(processing_time) * 1000.0
                l.append([request_total_delay[i], combined_str, computation_delay_for_node, propagation_delay_for_node,
                          tramission_delay_for_node])

            except Exception as e:
                print(
                    f"[CONTROLLER, !] Failed to schedule request on server {i}: "
                    f"{type(e).__name__} - {e}"
                )

        request.total_processing_delay = np.array(request_total_delay)

        # compute reward and track latency metrics
        try:
            final_step_reward_list = self.compute_step_reward(request, action_subset)
            headroom_reward = self._collapse_step_reward(final_step_reward_list, action_subset)
            combined_step_reward = headroom_reward   # default; may be adjusted by REWARD_MODE below

            # Track observed latency (minimum among selected servers)
            if len(action_subset) > 0 and hasattr(request, 'total_processing_delay'):
                observed_latency = min(request.total_processing_delay[i] for i in action_subset
                                       if i < len(request.total_processing_delay))
            if len(action_subset) > 0 and hasattr(request, 'total_processing_delay'):
                sorted_list = sorted(l)[0]
                observed_latency = sorted_list[0]
                request.contention_str = sorted_list[1]  # [SAFETAIL][FIX][D-19] not .combination
                self.episode_latencies.append(observed_latency)

                # [SAFETAIL][REWARD][FIX][B4][M-03] optional tau-referenced
                # tail-latency term. REWARD_MODE in {headroom, tau, headroom+tau};
                # "headroom" (default) reproduces the pre-B4 reward exactly.
                mode = getattr(constants, "REWARD_MODE", "headroom")
                if mode in ("tau", "headroom+tau"):
                    combined_step_reward = self._apply_tau_term(
                        headroom_reward, request, action_subset, observed_latency, mode)
                # l.append([request_total_delay[i], combined_str, computation_delay_for_node, propagation_delay_for_node, tramission_delay_for_node])

                # Track deviation from median
                deviation = abs(observed_latency - self.agent.median_computation_delay)
                self.episode_deviations.append(deviation)
                self.log_latency_to_csv(
                    request=request,
                    total_latency=sorted_list[0],
                    computation_delay=sorted_list[2],
                    propagation_delay=sorted_list[3],
                    transmission_delay=sorted_list[4],
                    queueing_delay=getattr(request, 'queue_waiting_time', -1.0),
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

        self.current_step += 1
        try:
            with open(self.step_reward_log_path, "a") as f:
                f.write(f"{self.current_episode},{self.current_step},{combined_step_reward:.6f}\n")
        except Exception as e:
            print(f"[CONTROLLER] Failed to log step reward: {e}")
        if self.policy is not None:
            # [SAFETAIL][SEAM][M-02] Feed the realised reward back; keep episodic
            # bookkeeping alive so latency/episode logs still populate.
            try:
                self.policy.observe(self._pending_policy_ctx, action_subset, float(combined_step_reward))
            except Exception as e:
                print(f"[CONTROLLER, !] policy.observe failed: {type(e).__name__} - {e}")
            try:
                self.step_rewards.append(combined_step_reward)
            except Exception:
                pass

        elif self.BASELINE_MODE == "safetail":
            # [SAFETAIL][REPLAY][FIX][D-04][D-05] store ARRAYS, not the live
            # Request. s_t was snapshotted before scheduling (estimates only);
            # s_{t+1} is snapshotted now, post-action (realised delays), so the
            # two are genuinely different (D-05) and s_t carries no outcome (D-04).
            try:
                if not self.testing_phase_active:
                    s_tp1_arr = None
                    try:
                        s_tp1_arr = request_to_state_array(request, self.agent.remove_nan_in_state)
                    except Exception as e:
                        print(f"[SAFETAIL][REPLAY][D-04] s_t+1 snapshot failed: {type(e).__name__} - {e}")
                    self.step_experiences.append({
                        "state_arr": s_t_arr,
                        "action": action_index,
                        "reward": float(combined_step_reward),   # per-step credit (D-06)
                        "next_state_arr": s_tp1_arr,
                    })
                    if _AUDIT_REPLAY and s_t_arr is not None:
                        try:
                            AUDIT_TRANSITIONS.append({
                                "episode": int(self.current_episode),
                                "s_t": np.asarray(s_t_arr, dtype=float).copy(),
                                "s_tp1": (None if s_tp1_arr is None
                                          else np.asarray(s_tp1_arr, dtype=float).copy()),
                                "reward": float(combined_step_reward),
                                "realised_delay": np.asarray(
                                    getattr(request, "total_processing_delay", []), dtype=float).copy(),
                            })
                        except Exception:
                            pass
            except Exception as e:
                print(f"[CONTROLLER, !] Failed to store experience: {type(e).__name__} - {e}")

            try:
                self.step_rewards.append(combined_step_reward)
                self.agent.epsilon_curve = np.append(
                    self.agent.epsilon_curve, self.agent.epsilon
                )
            except Exception as e:
                print(f"[CONTROLLER, !] Failed to update reward/epsilon tracking: {type(e).__name__} - {e}")

            # ---- Post-epsilon-min countdown ----
            if not self.epsilon_min_reached and self.agent.epsilon <= self.agent.epsilon_min:
                self.epsilon_min_reached = True
                print(
                    f"[CONTROLLER] Epsilon has reached its minimum value "
                    f"({self.agent.epsilon_min}). "
                    f"Running {self.post_epsilon_steps_target} more steps, "
                    f"then saving model and entering testing phase."
                )

            if self.epsilon_min_reached and not self.testing_phase_active:
                self.post_epsilon_steps += 1
                print(
                    f"[CONTROLLER] Post-epsilon-min step "
                    f"{self.post_epsilon_steps}/{self.post_epsilon_steps_target}"
                )
                if self.post_epsilon_steps >= self.post_epsilon_steps_target:
                    self._save_and_enter_testing()

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
        try:
            with open(self.episode_reward_log_path, "a") as f:
                f.write(f"{self.current_episode},{episodic_reward:.6f}\n")
        except Exception as e:
            print(f"[CONTROLLER] Failed to log episodic reward: {e}")
        print(
            f"[EPISODE {self.current_episode}] "
            f"Steps={len(self.step_rewards)}, "
            f"Reward={episodic_reward:.4f}"
        )

        if self.policy is not None:
            try:
                self.policy.finish_episode(int(self.current_episode))
            except Exception as e:
                print(f"[CONTROLLER, !] policy.finish_episode failed: {type(e).__name__} - {e}")
            self.agent.rewards = np.append(self.agent.rewards, episodic_reward)

        elif self.BASELINE_MODE == "safetail":

            if self.testing_phase_active:
                # continue reward plots during testing
                self.agent.rewards = np.append(self.agent.rewards, episodic_reward)

                print(
                    f"[TEST EPISODE {self.current_episode}] "
                    f"Reward={episodic_reward:.3f}"
                )

            else:
                # [SAFETAIL][REPLAY][FIX][D-06] keep the PER-STEP reward and add
                # the episodic signal as a broadcast bonus R_ep/N -- do NOT
                # overwrite r_t with episodic_reward (which is what discarded all
                # per-step credit).
                n_steps = max(1, len(self.step_experiences))
                ep_bonus = float(episodic_reward) / n_steps
                stored = 0
                for exp in self.step_experiences:
                    if exp.get("state_arr") is None or exp.get("next_state_arr") is None:
                        continue
                    self.agent.store_arrays(
                        exp["state_arr"], exp["action"],
                        exp["reward"] + ep_bonus, exp["next_state_arr"],
                    )
                    stored += 1
                print(f"[SAFETAIL][REPLAY][D-06] stored {stored}/{len(self.step_experiences)} "
                      f"transitions (per-step r + R_ep/{n_steps}={ep_bonus:.4f})")

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
        # -------- ACCESS RATE LOGGING --------
        if len(self.agent.episode_access_rate) > 0:
            avg_access_rate = np.mean(self.agent.episode_access_rate)
        else:
            avg_access_rate = 0.0

        self.access_rate_log.append((self.current_episode, avg_access_rate))

        try:
            df = pd.DataFrame(self.access_rate_log, columns=["episode", "access_rate"])
            csv_path = Path(constants.training_log_folder) / "access_rate_log.csv"
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"[CONTROLLER] Failed to save access rate CSV: {e}")
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

    def _save_and_enter_testing(self):
        """
        Save trained model and switch to testing phase.
        Also store separate testing-start indices for each metric.
        """

        print("[CONTROLLER] Saving model and entering testing phase...")

        save_dir = Path(constants.training_log_folder) / "saved_models"
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / f"model_post_eps_min_{time.strftime('%Y%m%d_%H%M%S')}.keras"
        self.agent.model.save(model_path)

        print(f"[CONTROLLER] Model saved -> {model_path}")

        # =====================================================
        # IMPORTANT: Separate indices for each metric
        # =====================================================
        self.agent.testing_start_reward_index = len(self.agent.rewards)
        self.agent.testing_start_latency_index = len(self.agent.latencies)
        self.agent.testing_start_access_index = len(self.agent.episode_access_rate)
        self.agent.testing_start_epsilon_index = len(self.agent.epsilon_curve)
        self.agent.testing_start_loss_index = len(self.agent.loss)
        self.agent.testing_start_deviation_index = len(self.agent.deviations)

        print("[CONTROLLER] Testing indices stored:")
        print("Reward   :", self.agent.testing_start_reward_index)
        print("Latency  :", self.agent.testing_start_latency_index)
        print("Access   :", self.agent.testing_start_access_index)
        print("Epsilon  :", self.agent.testing_start_epsilon_index)
        print("Loss     :", self.agent.testing_start_loss_index)
        print("Deviation:", self.agent.testing_start_deviation_index)

        # testing mode
        self.testing_phase_active = True
        self.agent.epsilon = 0.0
        self.agent.memory.clear()

        print("[CONTROLLER] Testing phase started.")

    def generate_testing_plots(self):
        # [SAFETAIL][DEAD][D-29][M-14] empty stub -- testing-phase results are not
        # plotted at all. Implement or drop the split's plotting claim (B9/M-14).
        pass

    def generate_plots(self):
        """
        Generate all training plots and save metrics summary.
        Called periodically during training.
        """
        if getattr(constants, "SMOKE", False):
            return  # [SAFETAIL][PLOT][smoke] plan.md 9.4: smoke runs produce no plots
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
        These are higher quality and include all dataset.
        """
        if getattr(constants, "SMOKE", False):
            return  # [SAFETAIL][PLOT][smoke] plan.md 9.4
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

            # # 3. Detailed metrics summary
            # self.agent.save_metrics_summary(
            #     final_dir / f"final_summary_{timestamp}.txt"
            # )

            # # 4. Export training dataset to CSV for external analysis
            # self.export_training_data(final_dir / f"training_data_{timestamp}.csv")

            print(f"[CONTROLLER] ✅ Final plots saved to {final_dir}")
            print("=" * 60 + "\n")

        except Exception as e:
            print(f"[CONTROLLER] ⚠️ Failed to generate final plots: {type(e).__name__} - {e}")

    def export_training_data(self, filepath):
        """[SAFETAIL][DEAD][D-29][M-14] orphan; body is entirely commented out."""
        # try:
        #     # Determine the maximum length
        #     max_len = max(
        #         len(self.agent.loss),
        #         len(self.agent.val_loss),
        #         len(self.agent.epsilon_curve),
        #         len(self.agent.rewards),
        #         len(self.agent.latencies),
        #         len(self.agent.deviations),
        #         len(self.agent.episode_access_rate),
        #         len(self.agent.exploit_or_explore),
        #         len(self.agent.prediction_times)
        #     )

        #     # Helper to pad arrays
        #     def pad_to_length(arr, length, fill_value=np.nan):
        #         if len(arr) < length:
        #             if isinstance(arr, np.ndarray):
        #                 return np.concatenate([arr, np.full(length - len(arr), fill_value)])
        #             else:
        #                 return list(arr) + [fill_value] * (length - len(arr))
        #         return arr

        #     # Create DataFrame
        #     dataset = {
        #         "episode": range(max_len),
        #         "loss": pad_to_length(self.agent.loss, max_len),
        #         "val_loss": pad_to_length(self.agent.val_loss, max_len),
        #         "epsilon": pad_to_length(self.agent.epsilon_curve, max_len),
        #         "reward": pad_to_length(self.agent.rewards, max_len),
        #         "latency": pad_to_length(self.agent.latencies, max_len),
        #         "deviation": pad_to_length(self.agent.deviations, max_len),
        #         "access_rate": pad_to_length(self.agent.episode_access_rate, max_len),
        #         "strategy": pad_to_length(self.agent.exploit_or_explore, max_len, fill_value=""),
        #         "prediction_time": pad_to_length(self.agent.prediction_times, max_len),
        #     }

        #     df = pd.DataFrame(dataset)
        #     df.to_csv(filepath, index=False)
        #     print(f"[CONTROLLER] 💾 Training dataset exported to {filepath}")

        # except Exception as e:
        #     print(f"[CONTROLLER] ⚠️ Failed to export training dataset: {type(e).__name__} - {e}")
        pass

    def save_checkpoint(self):
        """[SAFETAIL][DEAD][D-29] orphan -- only referenced from a commented-out call."""
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
                        self.process_step(request)
                        # [SAFETAIL][REWARD][FIX][D-20] the unconditional
                        # request_*_done increments that used to live here are
                        # gone -- completion is now counted deadline-conditionally
                        # inside compute_step_reward() where T is known.

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
