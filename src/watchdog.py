#!/usr/bin/env python3
"""
watchdog.py

Launches main.py and monitors the episodic reward curve.
If the reward is consistently declining after a warm-up period,
it kills the process and restarts from scratch.

Usage:
    python watchdog.py

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

MIN_EPISODES_BEFORE_CHECK = constants.watchdog_min_episodes_before_check
WINDOW                    = constants.watchdog_window
SLOPE_THRESHOLD           = constants.watchdog_slope_threshold
CONSECUTIVE_BAD_CHECKS    = constants.watchdog_consecutive_bad_checks
CHECK_INTERVAL_SEC        = constants.watchdog_check_interval_sec
MAX_RESTARTS              = constants.watchdog_max_restarts

SRC_DIR = Path(__file__).resolve().parent


def _reward_csv_path() -> Path:
    return SRC_DIR / constants.training_log_folder / "episode_rewards.csv"


def _slope(values: np.ndarray) -> float:
    """Linear regression slope of a 1-D array."""
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def _start() -> subprocess.Popen:
    """Launch main.py in a new process group so we can kill the whole group."""
    proc = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=str(SRC_DIR),
        env=os.environ.copy(),
        preexec_fn=os.setsid,   # new process group
    )
    print(f"\n[WATCHDOG] Started main.py  (PID={proc.pid})\n")
    return proc


def _kill(proc: subprocess.Popen) -> None:
    """Send SIGTERM to the entire process group, then SIGKILL if it hangs."""
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
    """
    Block until the reward CSV appears or the process exits.
    Returns True if the CSV appeared, False if the process exited first.
    """
    print(f"[WATCHDOG] Waiting for reward CSV: {csv_path}")
    while not csv_path.exists():
        if proc.poll() is not None:
            print(f"[WATCHDOG] Process exited (code={proc.returncode}) before CSV appeared.")
            return False
        time.sleep(3)
    print("[WATCHDOG] Reward CSV found. Monitoring started.")
    return True


def _monitor(proc: subprocess.Popen, csv_path: Path) -> str:
    """
    Continuously check the reward CSV until one of:
      - "restart"  : reward is consistently decreasing → caller should restart
      - "done"     : process exited cleanly (training finished)
      - "crashed"  : process exited with non-zero code

    Returns one of the three strings above.
    """
    bad_streak = 0

    while True:
        time.sleep(CHECK_INTERVAL_SEC)

        # ── Has the process exited? ──────────────────────────────────────────
        ret = proc.poll()
        if ret is not None:
            if ret == 0:
                print(f"[WATCHDOG] Training finished cleanly (code=0).")
                return "done"
            else:
                print(f"[WATCHDOG] Process crashed (code={ret}).")
                return "crashed"

        # ── Read reward CSV ──────────────────────────────────────────────────
        try:
            df = pd.read_csv(csv_path)
            rewards = df["episodic_reward"].dropna().values
        except Exception as e:
            print(f"[WATCHDOG] Could not read CSV ({e}). Skipping check.")
            continue

        n = len(rewards)
        print(f"[WATCHDOG] Episodes recorded: {n}  |  bad_streak: {bad_streak}/{CONSECUTIVE_BAD_CHECKS}")

        if n < MIN_EPISODES_BEFORE_CHECK:
            print(f"[WATCHDOG] Warm-up phase ({n}/{MIN_EPISODES_BEFORE_CHECK} episodes). Waiting...")
            continue

        # ── Trend check ──────────────────────────────────────────────────────
        recent = rewards[-WINDOW:]
        slope = _slope(recent)
        recent_mean = float(np.mean(recent))
        print(f"[WATCHDOG] Last {WINDOW} episodes — mean: {recent_mean:.4f}  slope: {slope:.6f}")

        if slope < SLOPE_THRESHOLD:
            bad_streak += 1
            print(f"[WATCHDOG] Reward declining  (slope={slope:.6f} < threshold={SLOPE_THRESHOLD})  "
                  f"bad_streak={bad_streak}/{CONSECUTIVE_BAD_CHECKS}")
            if bad_streak >= CONSECUTIVE_BAD_CHECKS:
                print("[WATCHDOG] Reward consistently decreasing. Triggering restart.")
                return "restart"
        else:
            if bad_streak > 0:
                print(f"[WATCHDOG] Slope recovered ({slope:.6f}). Resetting bad streak.")
            bad_streak = 0


def main():
    print("=" * 60)
    print(" SafeTail Watchdog")
    print(f" MIN_EPISODES_BEFORE_CHECK : {MIN_EPISODES_BEFORE_CHECK}")
    print(f" WINDOW                    : {WINDOW}")
    print(f" SLOPE_THRESHOLD           : {SLOPE_THRESHOLD}")
    print(f" CONSECUTIVE_BAD_CHECKS    : {CONSECUTIVE_BAD_CHECKS}")
    print(f" CHECK_INTERVAL_SEC        : {CHECK_INTERVAL_SEC}")
    print(f" MAX_RESTARTS              : {MAX_RESTARTS}")
    print("=" * 60)

    restarts = 0

    while restarts <= MAX_RESTARTS:
        csv_path = _reward_csv_path()

        proc = _start()

        csv_found = _wait_for_csv(proc, csv_path)
        if not csv_found:
            # Process died before producing any output — count as a crash/restart
            restarts += 1
            print(f"[WATCHDOG] Restart #{restarts}/{MAX_RESTARTS} in 5 s...\n")
            time.sleep(5)
            continue

        outcome = _monitor(proc, csv_path)

        if outcome == "done":
            print("[WATCHDOG] Training completed successfully. Exiting.")
            return

        if outcome in ("restart", "crashed"):
            _kill(proc)
            restarts += 1
            reason = "declining reward" if outcome == "restart" else "crash"
            print(f"[WATCHDOG] Restart #{restarts}/{MAX_RESTARTS} due to {reason}. "
                  f"Waiting 5 s before restarting...\n")
            time.sleep(5)

    print(f"[WATCHDOG] Reached maximum restarts ({MAX_RESTARTS}). Giving up.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[WATCHDOG] Interrupted by user. Exiting.")
