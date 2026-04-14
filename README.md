# SafeTail 2.0

SafeTail 2.0 is a **Deep Reinforcement Learning-based workload scheduler for heterogeneous edge computing**. It uses a **DQN agent** to decide which subset of edge servers should handle each incoming request, optimising for low latency and high resource efficiency.

It extends [SafeTail 1.0](https://arxiv.org/html/2408.17171v1) with RL-based scheduling, MLP latency prediction, deadline-aware rewards, and automated training analytics.

---

## How It Works

```
Users → SenderBursts ──TCP──► Receiver ──► Controller ──► Edge Servers (1–5)
                                               │
                                           DQN Agent
                                        (picks server subset)
```

1. **SenderBursts** generates requests in configurable bursts and sends them over TCP.
2. **Receiver** deserialises incoming chunks and passes them to the Controller.
3. **Controller** queries all servers for estimated latency, feeds the request state into the DQN, and schedules it on the selected server subset.
4. After every episode (N chunks), the Controller computes a reward and trains the DQN via experience replay.

### Request Types

| Type | Description | Soft deadline (D1) | Hard deadline (D2) |
|------|-------------|--------------------|--------------------|
| `s`  | Speech      | 100 ms             | 400 ms             |
| `d`  | Detect      | 30 ms              | 200 ms             |
| `p`  | Predict     | 30 ms              | 200 ms             |

### Reward Structure

- **Step reward** — `log((1−cpu)(1−ram)(1−gpu_mem)(1−gpu_util) + 1)` — rewards low resource utilisation on chosen servers.
- **Episodic reward** — `Σ(γⁱ · R_step) + ω − W_avg` — adds deadline satisfaction (ω) and penalises average queue wait time.

### Training Phases

1. **Training** — epsilon decays from 1.0 → `epsilon_min`, experience replay trains the DQN after every episode.
2. **Post-epsilon-min** — runs `post_epsilon_steps_target` more steps at minimum epsilon, then saves the model.
3. **Testing** — loads the saved model, sets epsilon to 0, no further training.

---

## Project Structure

```
SafeTail-2.0/
├── run.py                        # Entry point — launch training on the server
├── src/
│   ├── constants.py              # All configuration lives here
│   ├── main.py                   # Core training loop
│   ├── watchdog.py               # Auto-restarts training on declining reward
│   ├── controller.py             # RL scheduling + reward + training orchestration
│   ├── agent.py                  # DQN agent (Encoder + DQN, experience replay)
│   ├── servers.py                # Edge server simulation + latency prediction
│   ├── user.py                   # Request object
│   ├── receiver.py               # TCP socket server
│   ├── sender_bursts.py          # Burst request generator
│   └── server{1-5}_regressor/    # Per-server MLP latency predictors
├── data/
│   ├── server{1-5}.csv           # Server hardware + processing time profiles
│   └── propagation_delays.pkl    # Propagation delay distributions
└── training_logs_*/              # Generated during training (plots, CSVs, model)
```

---

## Configuration

**Everything is configured in `src/constants.py`.** Key parameters:

### Training
| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_no_request` | `500000` | Total requests to send |
| `chunk_size` | `5` | Requests per chunk |
| `episode_size` | `4` | Chunks per episode |
| `beta` | `5` | Number of edge servers |
| `learning_rate` | `1e-6` | DQN learning rate |
| `epsilon_min` | `0.1` | Minimum exploration rate |
| `gamma_decay` | `0.002` | Epsilon decay per episode |
| `discount_rate` | `0.9` | Reward discount factor (γ) |
| `batch_size` | `128` | Experience replay batch size |

### Deadlines
```python
DEADLINE_SCALE   = 1              # Multiply deadlines by this (e.g. 0.8 = tighter, 1.5 = looser)
ORIGINAL_DEADLINES = [[[100, 400], [30, 200]]]   # [[D1_s, D2_s], [D1_dp, D2_dp]] in ms
```

### Phases
```python
testing_phase_active = False      # False = training, True = testing (loads saved model)
saved_model_path     = "..."      # Path to model used during testing phase
```

### Baseline Mode
```python
BASELINE_MODE = "safetail"        # Which scheduling strategy to use
```

| Value | Strategy |
|-------|----------|
| `"safetail"` | DQN agent (default) |
| `"minload_1/2/3"` | Pick 1/2/3 servers with lowest current load |
| `"minprop_1/2/3"` | Pick 1/2/3 servers with lowest propagation delay |
| `"rand_1/2/3"` | Pick 1/2/3 random servers |

### Run Settings
```python
log_file_prefix = "log"           # Log files: log_1.txt, log_2.txt, ...
use_watchdog    = False           # True → auto-restart if reward declines
```

### Watchdog Tuning
```python
watchdog_min_episodes_before_check = 30    # Warm-up episodes before any restart
watchdog_window                    = 15    # Slope computed over last N episodes
watchdog_slope_threshold           = -0.005 # Slope below this = bad check
watchdog_consecutive_bad_checks    = 10    # Bad checks in a row before restart
watchdog_check_interval_sec        = 30    # Seconds between checks
watchdog_max_restarts              = 10    # Give up after this many restarts
```

---

## Running

### On a Server (recommended)

```bash
python run.py
```

That's it. `run.py` reads all settings from `constants.py`, picks the next available log file (`log_1.txt`, `log_2.txt`, ...), detaches from the terminal (no need for `nohup`), and prints the PID and how to monitor/kill it:

```
[RUN] Script  : main.py
[RUN] Log file: log_1.txt
[RUN] PID     : 12345  (saved to log_1.txt.pid)
[RUN] Monitor : tail -f log_1.txt
[RUN] Kill    : kill $(cat log_1.txt.pid)
```

### Mode 1 — Standard Training

In `constants.py`:
```python
testing_phase_active = False
use_watchdog         = False
BASELINE_MODE        = "safetail"
```

```bash
python run.py
```

### Mode 2 — Training with Auto-Restart (Watchdog)

If reward keeps declining, the watchdog kills and restarts training automatically.

In `constants.py`:
```python
testing_phase_active = False
use_watchdog         = True
```

```bash
python run.py
```

The watchdog monitors the reward CSV every `watchdog_check_interval_sec` seconds. If the slope of the last `watchdog_window` episodes stays below `watchdog_slope_threshold` for `watchdog_consecutive_bad_checks` consecutive checks, it kills and restarts the process. All restarts append to the **same log file**.

### Mode 3 — Testing a Saved Model

In `constants.py`:
```python
testing_phase_active = True
saved_model_path     = "./training_logs_1/post_epsilon_min_save/model_post_eps_min_YYYYMMDD_HHMMSS.keras"
```

```bash
python run.py
```

Epsilon is set to 0 (pure exploitation). Training history is loaded from `training_logs_1/` so reward/latency plots are continuous across training → testing.

### Mode 4 — Baseline Comparison

In `constants.py`:
```python
BASELINE_MODE = "minload_2"       # or minprop_1, rand_3, etc.
log_file_prefix = "minload_2"     # keep logs organised by mode
```

```bash
python run.py
```

### Running Locally (without detaching)

```bash
cd src
python main.py
```

---

## Monitoring & Stopping

```bash
# Follow logs live
tail -f log_1.txt

# Check if process is still running
ps -p $(cat log_1.txt.pid)

# Stop everything (watchdog + main.py) for the latest run
python stop.py

# Stop a specific run
python stop.py log_2.txt
```

> **Note:** `kill $(cat log_1.txt.pid)` only kills the watchdog — `stop.py` kills both the watchdog and the main.py it spawned.

Training plots are generated every 20 episodes and saved to:
```
src/training_logs_*/plots/ep<N>/all_metrics.png
```

Reward and latency CSVs are at:
```
src/training_logs_*/episode_rewards.csv
src/training_logs_*/<BASELINE_MODE>_latency_log.csv
```

---

## Contributors

- [Shrutya Chawla](https://github.com/shrutya22487/)
- [Shamik Sinha](https://github.com/theshamiksinha)
- [Shivankar Singh](https://github.com/BingoBoy479)
- [Jyoti Shokhanda](https://github.com/Jyotishokhanda)
- [Arani Bhattacharya](https://github.com/arani89)
