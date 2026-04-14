#!/usr/bin/env python3
"""
watchdog.py

Launches main.py and monitors the episodic reward at fixed checkpoints.

Logic:
  - Record the average reward over the first COMPARISON_WINDOW episodes as the baseline.
  - At every CHECKPOINT_INTERVAL episodes (200, 400, 600, ...) compute the average
    reward over the last COMPARISON_WINDOW episodes.
  - If that average has not improved by at least MIN_IMPROVEMENT over the previous
    checkpoint, kill the process and restart.

Tune the parameters in the Watchdog section of constants.py.
"""

import os
import sys
import signal
import subprocess
import time
import numpy as np
import pandas as pd
from pathlib import Path
import constants

CHECKPOINT_INTERVAL  = constants.watchdog_checkpoint_interval
MIN_IMPROVEMENT      = constants.watchdog_min_improvement
COMPARISON_WINDOW    = constants.watchdog_comparison_window
CHECK_INTERVAL_SEC   = constants.watchdog_check_interval_sec
MAX_RESTARTS         = constants.watchdog_max_restarts

SRC_DIR = Path(__file__).resolve().parent


def _reward_csv_path() -> Path:
    return SRC_DIR / constants.training_log_folder / "episode_rewards.csv"


def _window_avg(rewards: np.ndarray, end_episode: int) -> float:
    """Average reward over the COMPARISON_WINDOW episodes ending at end_episode."""
    start = max(0, end_episode - COMPARISON_WINDOW)
    return float(np.mean(rewards[start:end_episode]))


def _start() -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=str(SRC_DIR),
        env=os.environ.copy(),
        preexec_fn=os.setsid,
    )
    print(f"\n[WATCHDOG] Started main.py  (PID={proc.pid})\n")
    return proc


def _kill(proc: subprocess.Popen) -> None:
    if proc is None or proc.poll() is not None:
        return
    pgid = None
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        print(f"[WATCHDOG] Sent SIGTERM to process group {pgid}")
    except ProcessLookupError:
        return
    except Exception as e:
        print(f"[WATCHDOG] SIGTERM failed ({e}); falling back to proc.terminate()")
        proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        print("[WATCHDOG] Process did not exit after 15 s — sending SIGKILL")
        try:
            if pgid:
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass
        proc.wait()


def _wait_for_csv(proc: subprocess.Popen, csv_path: Path) -> bool:
    print(f"[WATCHDOG] Waiting for reward CSV: {csv_path}")
    while not csv_path.exists():
        if proc.poll() is not None:
            print(f"[WATCHDOG] Process exited before CSV appeared.")
            return False
        time.sleep(3)
    print("[WATCHDOG] Reward CSV found. Monitoring started.")
    return True


def _monitor(proc: subprocess.Popen, csv_path: Path) -> str:
    """
    Returns "restart", "done", or "crashed".

    Checkpoints: CHECKPOINT_INTERVAL, 2*CHECKPOINT_INTERVAL, 3*CHECKPOINT_INTERVAL, ...
    At each checkpoint, compare avg reward in the last COMPARISON_WINDOW episodes
    against the avg at the previous checkpoint. Restart if improvement < MIN_IMPROVEMENT.
    """
    next_checkpoint   = CHECKPOINT_INTERVAL   # next episode count to evaluate
    prev_avg          = None                  # avg reward at the previous checkpoint

    print(f"[WATCHDOG] First check at episode {next_checkpoint}, "
          f"then every {CHECKPOINT_INTERVAL} episodes.")
    print(f"[WATCHDOG] MIN_IMPROVEMENT={MIN_IMPROVEMENT}, "
          f"COMPARISON_WINDOW={COMPARISON_WINDOW}")

    while True:
        time.sleep(CHECK_INTERVAL_SEC)

        # ── Process still alive? ─────────────────────────────────────────────
        ret = proc.poll()
        if ret is not None:
            return "done" if ret == 0 else "crashed"

        # ── Read CSV ─────────────────────────────────────────────────────────
        try:
            df = pd.read_csv(csv_path)
            rewards = df["episodic_reward"].dropna().values
        except Exception as e:
            print(f"[WATCHDOG] Could not read CSV ({e}). Skipping.")
            continue

        n = len(rewards)
        print(f"[WATCHDOG] Episodes so far: {n}  |  next checkpoint: {next_checkpoint}")

        if n < next_checkpoint:
            continue

        # ── We've hit the checkpoint ─────────────────────────────────────────
        current_avg = _window_avg(rewards, next_checkpoint)
        print(f"[WATCHDOG] Checkpoint ep {next_checkpoint}: "
              f"avg reward (last {COMPARISON_WINDOW} eps) = {current_avg:.4f}")

        if prev_avg is None:
            # First checkpoint — this becomes the baseline, no restart decision yet
            print(f"[WATCHDOG] Baseline set to {current_avg:.4f}. "
                  f"Next check at episode {next_checkpoint + CHECKPOINT_INTERVAL}.")
        else:
            improvement = current_avg - prev_avg
            print(f"[WATCHDOG] vs previous checkpoint ({prev_avg:.4f}): "
                  f"improvement = {improvement:.4f}  (need >= {MIN_IMPROVEMENT})")
            if improvement < MIN_IMPROVEMENT:
                print(f"[WATCHDOG] Not enough improvement. Triggering restart.")
                return "restart"
            else:
                print(f"[WATCHDOG] OK. Next check at episode {next_checkpoint + CHECKPOINT_INTERVAL}.")

        prev_avg       = current_avg
        next_checkpoint += CHECKPOINT_INTERVAL


def main():
    print("=" * 60)
    print(" SafeTail Watchdog")
    print(f" CHECKPOINT_INTERVAL : {CHECKPOINT_INTERVAL}")
    print(f" MIN_IMPROVEMENT     : {MIN_IMPROVEMENT}")
    print(f" COMPARISON_WINDOW   : {COMPARISON_WINDOW}")
    print(f" CHECK_INTERVAL_SEC  : {CHECK_INTERVAL_SEC}")
    print(f" MAX_RESTARTS        : {MAX_RESTARTS}")
    print("=" * 60)

    restarts = 0

    while restarts <= MAX_RESTARTS:
        csv_path = _reward_csv_path()
        proc     = _start()

        if not _wait_for_csv(proc, csv_path):
            restarts += 1
            print(f"[WATCHDOG] Restart #{restarts}/{MAX_RESTARTS} in 5 s...\n")
            time.sleep(5)
            continue

        outcome = _monitor(proc, csv_path)

        if outcome == "done":
            print("[WATCHDOG] Training completed successfully. Exiting.")
            return

        _kill(proc)
        restarts += 1
        reason = "no improvement at checkpoint" if outcome == "restart" else "crash"
        print(f"[WATCHDOG] Restart #{restarts}/{MAX_RESTARTS} — {reason}. "
              f"Waiting 5 s...\n")
        time.sleep(5)

    print(f"[WATCHDOG] Reached maximum restarts ({MAX_RESTARTS}). Giving up.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[WATCHDOG] Interrupted by user. Exiting.")
