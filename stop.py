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
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR  = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
import constants

WAIT_TIMEOUT = 10   # seconds to wait for SIGTERM before sending SIGKILL


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but not ours to signal — treat as alive


def _kill_wait(pid: int, label: str) -> None:
    """SIGTERM a PID, wait for it to die, SIGKILL if it doesn't."""
    if not _is_alive(pid):
        print(f"[STOP] {label} (PID={pid}) already gone.")
        return

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"[STOP] Sent SIGTERM to {label} (PID={pid})")
    except Exception as e:
        print(f"[STOP] Could not SIGTERM {label} (PID={pid}): {e}")
        return

    # Wait for clean exit
    deadline = time.time() + WAIT_TIMEOUT
    while time.time() < deadline:
        if not _is_alive(pid):
            print(f"[STOP] {label} (PID={pid}) exited cleanly.")
            return
        time.sleep(0.5)

    # Force kill
    try:
        os.kill(pid, signal.SIGKILL)
        print(f"[STOP] {label} (PID={pid}) did not exit — sent SIGKILL.")
    except Exception:
        pass


def _children_of(pid: int) -> list[int]:
    """
    Find direct child PIDs using /proc (more reliable than pgrep on all distros).
    Falls back to pgrep if /proc is unavailable.
    """
    children = []

    # /proc-based (Linux)
    proc = Path("/proc")
    if proc.exists():
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                status = (entry / "status").read_text()
                for line in status.splitlines():
                    if line.startswith("PPid:") and int(line.split()[1]) == pid:
                        children.append(int(entry.name))
                        break
            except Exception:
                pass
        return children

    # pgrep fallback (macOS / other)
    try:
        out = subprocess.run(
            ["pgrep", "-P", str(pid)],
            capture_output=True, text=True
        ).stdout.strip()
        return [int(p) for p in out.split() if p.strip()]
    except Exception:
        return []


def _kill_by_port(port: int) -> None:
    """
    Last-resort: kill whatever process is still holding the receiver port.
    Uses fuser (Linux) or lsof (macOS).
    """
    pids = []

    # fuser (Linux)
    try:
        out = subprocess.run(
            ["fuser", f"{port}/tcp"],
            capture_output=True, text=True
        ).stdout.strip()
        pids = [int(p) for p in out.split() if p.strip()]
    except Exception:
        pass

    # lsof fallback (macOS)
    if not pids:
        try:
            out = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}"],
                capture_output=True, text=True
            ).stdout.strip()
            pids = [int(p) for p in out.split() if p.strip()]
        except Exception:
            pass

    if pids:
        for pid in pids:
            _kill_wait(pid, f"process holding port {port}")
    else:
        print(f"[STOP] No process found holding port {constants.receiver_port}.")


def _latest_pid_file() -> Path | None:
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
        pid_file = ROOT_DIR / (arg if arg.endswith(".pid") else arg + ".pid")
    else:
        pid_file = _latest_pid_file()
        if pid_file is None:
            print("[STOP] No .pid files found — falling back to port-based kill.")
            _kill_by_port(constants.receiver_port)
            return

    if not pid_file.exists():
        print(f"[STOP] PID file not found: {pid_file} — falling back to port-based kill.")
        _kill_by_port(constants.receiver_port)
        return

    parent_pid = int(pid_file.read_text().strip())
    print(f"[STOP] Using {pid_file.name}  (PID={parent_pid})")

    # ── Kill children first (main.py spawned by watchdog) ────────────────────
    children = _children_of(parent_pid)
    if children:
        print(f"[STOP] Found child processes: {children}")
        for child in children:
            _kill_wait(child, "child process (main.py)")
    else:
        print("[STOP] No child processes found.")

    # ── Kill parent (watchdog or main.py) ────────────────────────────────────
    _kill_wait(parent_pid, "parent process (watchdog/main.py)")

    # ── Final safety net: kill anything still holding the port ───────────────
    time.sleep(1)
    _kill_by_port(constants.receiver_port)

    # ── Clean up PID file ────────────────────────────────────────────────────
    try:
        pid_file.unlink()
        print(f"[STOP] Removed {pid_file.name}")
    except Exception:
        pass

    print("[STOP] Done.")


if __name__ == "__main__":
    main()
