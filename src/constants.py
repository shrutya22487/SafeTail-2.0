max_bandwidth = 20
median_computation_delay = 0.05
total_no_request = 500000
chunk_size = 5
no_of_chunk= int(total_no_request/chunk_size)
episode_size = 4
no_of_burst = 1000
no_of_episodes = int(no_of_chunk/episode_size)
learning_rate =  1e-6
gamma_decay = 0.002
epsilon_min = 0.1
min_burst = 2
max_burst = 4
min_interval = 0.2
max_interval = 0.8
jitter = 0.02
lr_decay_rate = 0.999
lr_min = 1e-5
training_log_folder = "training_logs_2"
epochs = 1
no_of_sensors = 1
max_load = 5
beta = 5 # Number of edge servers.
alpha = 0.005 # Reward scaling factor.
discount_rate = 0.95
batch_size = 128
nS = 1*beta + 1 # Number of state features per step.
nA = 2**beta - 1 # Number of possible actions (subsets of servers).
exploit_or_explore = []
epsilon_curve=[]
episode_access_rate = []
avg_waiting_time = []
latencies = []
deviations = []
rewards = []
action = []
load_arr = []

epsilon_min_reached = False
post_epsilon_steps = 0
post_epsilon_steps_target = 8000
testing_phase_active = False

import os

receiver_host = "127.0.0.1"
receiver_port = int(os.environ.get("RECEIVER_PORT", 6000))

BASELINE_MODE = os.environ.get("BASELINE_MODE", "safetail")
# Options: "safetail"
# | "minload_1" | "minload_2" | "minload_3"
# | "minprop_1" | "minprop_2" | "minprop_3"
# | "rand_1"    | "rand_2"    | "rand_3"