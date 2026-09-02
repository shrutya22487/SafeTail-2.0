import os


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "", "false", "no")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip())
    except (TypeError, ValueError):
        return default


############################### RUN CONTROL (plan.md 9.4 / B7) ###########

# [SAFETAIL][MAIN][smoke] Short deterministic run for gates G2/G3/G4 and CI.
# Enabled by `--smoke` (main.py sets SAFETAIL_SMOKE=1) or the env var directly.
SMOKE = _env_flag("SAFETAIL_SMOKE")

# [SAFETAIL][MAIN][B7] Master seed threaded to random / numpy / tensorflow.
# `None` (env unset / "none") keeps legacy non-deterministic behaviour.
_seed_raw = os.environ.get("SAFETAIL_SEED", "").strip().lower()
SEED = None if _seed_raw in ("", "none") else _env_int("SAFETAIL_SEED", 0)

# [SAFETAIL][SEAM] External-policy selector (plan.md 8.3). "native" => built-in
# DQN / heuristic path; any other value is resolved via src/policy_registry.get().
POLICY = os.environ.get("POLICY", "native").strip() or "native"

# Smoke-scale overrides: ~20 episodes, ~300 requests, no plots.
SMOKE_CHUNKS = _env_int("SAFETAIL_SMOKE_CHUNKS", 60)   # 60 * chunk_size(5) = 300 requests
SMOKE_EPISODES = _env_int("SAFETAIL_SMOKE_EPISODES", 20)

# [SAFETAIL][REGRESSOR][D-02c] When False (default), a regressor that fails to
# load or a missing contention trace row is a HARD error -- the collapse is no
# longer silent. When True, degradation is allowed but logged [DEGRADED][D-02c]
# and counted in the run manifest; a run with any [DEGRADED] count > 0 is not
# publishable (gate G5).
ALLOW_DEGRADED_PREDICTORS = _env_flag("SAFETAIL_ALLOW_DEGRADED_PREDICTORS")

# [SAFETAIL][REWARD][B3][M-04][S-02] Redundancy price. The step reward gets an
# explicit cost  c_red * (|A|-1)/(beta-1)  subtracted, so replicating to more
# servers is no longer free (root cause of D-07 over-replication). c_red = 0.0
# reproduces the pre-B3 (un-priced) reward exactly.
# Source: "chosen, see plan.md B3". Unit: reward points (reward is in [0, log2]).
# SWEEP for the publishable K-vs-c_red ablation (plan.md B3 accept / figure F8):
C_RED = float(os.environ.get("SAFETAIL_C_RED", "0.0"))
C_RED_SWEEP = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]

# [SAFETAIL][CONTROLLER][B6][D-09] Budget-controlled evaluation. When set (e.g.
# 3.0), SafeTail's per-request subset is CAPPED (keeping the lowest-estimate
# servers) so its running mean |A| never exceeds this target -- the headline
# comparison then runs at a stated replication budget instead of SafeTail
# running free over all 31 subsets at mean K ~ 3.55 while the baselines are
# capped at K in {1,2,3}. The realised mean K is reported alongside (plan 9.3).
# None => SafeTail unconstrained.
_mk = os.environ.get("SAFETAIL_MATCH_K", "").strip()
MATCH_K = float(_mk) if _mk else None

# [SAFETAIL][REWARD][B4][M-03] Which step reward the 2.0 agent optimises:
#   "headroom"       -- BTP's resource-headroom product + B3 c_red cost (default;
#                       reproduces the pre-B4 reward exactly)
#   "tau"            -- SafeTail 1.0's tau-referenced 5-case tail-latency reward
#                       ONLY (src/rewards.tau_reward_5case)
#   "headroom+tau"   -- their sum
# M-03: tail latency -- the stated objective of ST/HED/BTP -- otherwise never
# appears in the 2.0 reward.
REWARD_MODE = os.environ.get("SAFETAIL_REWARD_MODE", "headroom").strip() or "headroom"

# tau per request type (seconds). Median service latency (comp+prop+trans) per
# type from results/reference_v0/safetail_training_logs/latency_log.csv
# (d 34.5 ms, p 35.5 ms, s 51.8 ms). Same source as baselines/safetail_v1
# config; a single global tau is meaningless (S-14). plan.md 13.1(1).
TAU_BY_TYPE = {
    "s": float(os.environ.get("SAFETAIL_TAU_S", "0.0518")),
    "d": float(os.environ.get("SAFETAIL_TAU_D", "0.0345")),
    "p": float(os.environ.get("SAFETAIL_TAU_P", "0.0355")),
}

############################### HYPERPARAMETERS ##########################
median_computation_delay = 0.05
# [SAFETAIL][DEAD][D-33] total_no_request only SIZES the pre-generated request
# pool. The run stops after `no_of_burst` bursts -> ~15,225 requests actually
# processed, not 500,000 (BTP Table 4.2 overstates by ~33x).
total_no_request = 500000
chunk_size = 5
no_of_chunk = int(total_no_request / chunk_size)
# [SAFETAIL][DEAD][D-30] episode_size is vestigial: the controller's episode
# length is `chunks_per_episode = 3`, not this. It only feeds `no_of_episodes`
# below, whose value (25,000) is never reached (~1,000 episodes actually run).
episode_size = 4
no_of_burst = 1000
no_of_episodes = int(no_of_chunk / episode_size)  # see D-30 note above
learning_rate = 1e-6
gamma_decay = 0.002
epsilon_min = 0.1
min_burst = 2
max_burst = 4
min_interval = 0.2
max_interval = 0.8
jitter = 0.02
lr_decay_rate = 0.999
lr_min = 1e-5
epochs = 1
# [SAFETAIL][DEAD][D-30] max_load is unused; servers.py hardcodes
# MAX_CONCURRENT_REQUESTS = 4.
max_load = 5
beta = 5  # Number of edge servers.
alpha = 0.005  # Reward scaling factor.
discount_rate = 0.9
batch_size = 128
# [SAFETAIL][DEAD][D-30][D-35] nS is dead: the encoder input is shape=(None,1)
# (variable-length, per S-10 -- hardware heterogeneity, not temporal). The agent
# stores `states=constants.nS` but never uses it. Kept only so the existing
# DQNAgent(states=constants.nS) call still resolves.
nS = 1 * beta + 1
nA = 2 ** beta - 1  # Number of possible actions (subsets of servers).
post_epsilon_steps = 8000

############################### TRAINING LOGS ##########################

training_log_folder = os.environ.get("TRAINING_LOG_FOLDER", "training_logs_1")

# references the results where the results were stored when first run was performed, WHILE TESTING PHASE IS RUNNING,
# leave same as `training_log_folder` if training for first time
original_training_log_folder = os.environ.get("TRAINING_LOG_FOLDER", "training_logs_1")

############################### DEADLINES ##########################

# Deadlines in ms: [[D1_s, D2_s], [D1_dp, D2_dp]]
# "s" (speech) type:      D1=100ms (soft), D2=400ms (hard)
# "d"/"p" type:           D1=30ms  (soft), D2=200ms (hard)

DEADLINE_SCALE = 1  # change accordingly to make deadlines 80% or 150% of original

ORIGINAL_DEADLINES = [[100, 400], [30, 200]]
deadlines = [[value * DEADLINE_SCALE for value in pair] for pair in ORIGINAL_DEADLINES]

############################### TESTING ##########################

# true when need to run testing phase
testing_phase_active = False

# which model to use while testing
saved_model_path = ""

############################### ADDRESS CONFIGURATION ##########################

receiver_host = "127.0.0.1"
receiver_port = int(os.environ.get("RECEIVER_PORT", 6001))

############################### BASELINE COMPARISON ##########################

BASELINE_MODE = os.environ.get("BASELINE_MODE", "safetail")
# Options:
#   "safetail"                      -- the DQN policy (or an external one via POLICY)
#   "{minload,minprop,rand}_{K}"    -- fixed-K redundant dispatcher; [SAFETAIL][B6]
#                                      K now ranges 1..beta (was hardcoded 1..3, D-09).
#                                      e.g. minload_4, minprop_5, rand_2
# NOTE the _{K} suffix means the REPLICATION BUDGET, not a seed. Seeds are
# SAFETAIL_SEED (plan.md B7). Results dirs are {policy}_{k}_{seed}_{gitsha7}.
