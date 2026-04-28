import os

############################### HYPERPARAMETERS ##########################
median_computation_delay = 0.05
total_no_request = 500000
chunk_size = 5
no_of_chunk = int(total_no_request / chunk_size)
episode_size = 4
no_of_burst = 1000
no_of_episodes = int(no_of_chunk / episode_size)
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
max_load = 5
beta = 5  # Number of edge servers.
alpha = 0.005  # Reward scaling factor.
discount_rate = 0.9
batch_size = 128
nS = 1 * beta + 1  # Number of state features per step.
nA = 2 ** beta - 1  # Number of possible actions (subsets of servers).
post_epsilon_steps = 8000

############################### TRAINING LOGS ##########################

training_log_folder = os.environ.get("TRAINING_LOG_FOLDER", "training_logs_1_testing")

# references the results where the results were stored when first run was performed, WHILE TESTING PHASE IS RUNNING...
original_training_log_folder = os.environ.get("TRAINING_LOG_FOLDER", "training_logs_1")

############################### DEADLINES ##########################

# Deadlines in ms: [[D1_s, D2_s], [D1_dp, D2_dp]]
# "s" (speech) type:      D1=100ms (soft), D2=400ms (hard)
# "d"/"p" type:           D1=30ms  (soft), D2=200ms (hard)

DEADLINE_SCALE = 1  # change accordingly to make dealines 80% or 150% of original

ORIGINAL_DEADLINES = [[100, 400], [30, 200]]
deadlines = [[value * DEADLINE_SCALE for value in pair] for pair in ORIGINAL_DEADLINES]

############################### TESTING ##########################

# true when need to run testing phase
testing_phase_active = True

# which model to use while testing
saved_model_path = ""

############################### ADDRESS CONFIGURATION ##########################

receiver_host = "127.0.0.8"
receiver_port = int(os.environ.get("RECEIVER_PORT", 6008))

############################### BASELINE COMPARISON ##########################

BASELINE_MODE = os.environ.get("BASELINE_MODE", "safetail")
# Options: "safetail"
# | "minload_1" | "minload_2" | "minload_3"
# | "minprop_1" | "minprop_2" | "minprop_3"
# | "rand_1"    | "rand_2"    | "rand_3"
