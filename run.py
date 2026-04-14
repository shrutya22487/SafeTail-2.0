#!/usr/bin/env python3
"""
run.py  —  run from the project root

    python run.py

Configure everything in src/constants.py:
    log_file_prefix  — log files will be named <prefix>_1.txt, <prefix>_2.txt, ...
    use_watchdog     — True to run watchdog.py, False to run main.py directly

The process is detached from the terminal (nohup-equivalent) so it keeps
running after you disconnect from the server.  The PID is saved to
<log_file>.pid so you can kill it later with:
    kill $(cat log_1.txt.pid)
"""

import subprocess
import sys
from pathlib import Path

SRC_DIR  = Path(__file__).resolve().parent / "src"
ROOT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(SRC_DIR))
import constants


def next_log_path() -> Path:
    """Return the first <prefix>_N.txt that does not already exist."""
    i = 1
    while True:
        p = ROOT_DIR / f"{constants.log_file_prefix}_{i}.txt"
        if not p.exists():
            return p
        i += 1


def main():
    log_path = next_log_path()
    script   = "watchdog.py" if constants.use_watchdog else "main.py"

    print(f"[RUN] Script  : {script}")
    print(f"[RUN] Log file: {log_path.name}")
    print(f"[RUN] Python  : {sys.executable}")

    with open(log_path, "a") as log_file:
        proc = subprocess.Popen(
            [sys.executable, script],
            cwd=str(SRC_DIR),
            stdout=log_file,
            stderr=log_file,
            stdin=subprocess.DEVNULL,
            start_new_session=True,     # detach from terminal (nohup equivalent)
        )

    pid_file = log_path.with_suffix(log_path.suffix + ".pid")
    pid_file.write_text(str(proc.pid))

    print(f"[RUN] PID     : {proc.pid}  (saved to {pid_file.name})")
    print(f"[RUN] Monitor : tail -f {log_path.name}")
    print(f"[RUN] Kill    : kill $(cat {pid_file.name})")


if __name__ == "__main__":
    main()
