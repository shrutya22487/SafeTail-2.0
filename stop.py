#!/usr/bin/env python3
"""
stop.py  —  stop all SafeTail processes for a given run

Usage:
    python stop.py              # stops the most recently started run
    python stop.py log_2.txt    # stops the run whose PID was saved in log_2.txt.pid
"""

import os
import sys
import signal
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR  = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
import constants


def _kill(pid: int, label: str) -> bool:
    """Send SIGTERM to a PID. Returns True if the process existed."""
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"[STOP] Sent SIGTERM to {label} (PID={pid})")
        return True
    except ProcessLookupError:
        print(f"[STOP] {label} (PID={pid}) is already gone.")
        return False
    except Exception as e:
        print(f"[STOP] Could not kill {label} (PID={pid}): {e}")
        return False


def _children(pid: int) -> list[int]:
    """Return direct child PIDs of the given process (works on Linux/macOS)."""
    try:
        out = subprocess.run(
            ["pgrep", "-P", str(pid)],
            capture_output=True, text=True
        ).stdout.strip()
        return [int(p) for p in out.split() if p.strip()]
    except Exception:
        return []


def _latest_pid_file() -> Path | None:
    """Return the most recently modified .pid file matching the log prefix."""
    files = sorted(
        ROOT_DIR.glob(f"{constants.log_file_prefix}_*.txt.pid"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    return files[0] if files else None


def main():
    # ── Resolve PID file ─────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        # accept "log_2.txt" or "log_2.txt.pid"
        pid_file = ROOT_DIR / (arg if arg.endswith(".pid") else arg + ".pid")
    else:
        pid_file = _latest_pid_file()
        if pid_file is None:
            print("[STOP] No .pid files found — nothing to stop.")
            return

    if not pid_file.exists():
        print(f"[STOP] PID file not found: {pid_file}")
        return

    parent_pid = int(pid_file.read_text().strip())
    print(f"[STOP] Using {pid_file.name}  (PID={parent_pid})")

    # ── Kill children first (main.py spawned by watchdog) ────────────────────
    children = _children(parent_pid)
    if children:
        for child in children:
            _kill(child, "child process (main.py)")
    else:
        print("[STOP] No child processes found (running main.py directly, or already dead).")

    # ── Kill the parent (watchdog or main.py) ────────────────────────────────
    _kill(parent_pid, "parent process (watchdog/main.py)")

    # ── Clean up PID file ────────────────────────────────────────────────────
    try:
        pid_file.unlink()
        print(f"[STOP] Removed {pid_file.name}")
    except Exception:
        pass

    print("[STOP] Done.")


if __name__ == "__main__":
    main()
