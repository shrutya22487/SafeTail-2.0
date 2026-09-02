import argparse
import inspect
import logging
import random
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# [SAFETAIL][MAIN][FIX][D-36] Windows consoles default to cp1252; src/ is full of
# emoji print()s that raise UnicodeEncodeError there. Force UTF-8 on the streams
# before anything prints. (Real fix -- routing prints through _safetail_log -- is B9.)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import constants
import user
from _seeding import seed_everything
from controller import Controller
from receiver import Receiver
from sender_bursts import SenderBursts

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---- load server CSVs once (NOT per request) ----
BASE_DIR = Path(__file__).resolve().parent.parent  # go from src/ -> project root
DATA_DIR = BASE_DIR / "dataset"

SERVER_CSVS = [
    DATA_DIR / "server1.csv",
    DATA_DIR / "server2.csv",
    DATA_DIR / "server3.csv",
    DATA_DIR / "server4.csv",
    DATA_DIR / "server5.csv",
]


def print_constants(module):
    print("\n" + "=" * 50)
    print(f"  CONSTANTS ({module.__name__})")
    print("=" * 50)
    for name, value in inspect.getmembers(module):
        # Skip built-ins, modules, and callables
        if not name.startswith("_") and not inspect.ismodule(value) and not callable(value):
            print(f"  {name:<30} = {value}")
    print("=" * 50 + "\n")


print_constants(constants)

SERVER_DFS = []
for p in SERVER_CSVS:
    if not p.exists():
        raise FileNotFoundError(f"Missing CSV: {p}")
    SERVER_DFS.append(pd.read_csv(p))


# --- request factory that uses the actual Request signature ---
def request_factory(i: int):
    """
    Pure Request factory.

    Responsibilities:
    - create Request object
    - set ids, combination, arrival time
    - initialize base arrays ONLY

    Non-responsibilities:
    - NO server CSV access
    - NO filling server dicts
    - NO filling server NP arrays
    - NO computation

    This function MUST NOT crash the sender.
    """

    # -------- safest defaults --------
    server_count = 5
    # added the deadlines these are in milli- second ms

    deadlines = np.asarray([
        [100, 400],
        [30, 200]
    ])

    try:
        # ---------------- combination ----------------
        try:
            combination = random.choice(["s", "p", "d"])
            if (combination == "s"):
                deadline = deadlines[0]
            else:
                deadline = deadlines[1]

        except Exception:
            # absolute fallback
            combination = "s"
            deadline = deadlines[0]

        # [SAFETAIL][MAIN][FIX][D-34] real per-request payload-size variation by
        # type (was the literal 1024 for every request, so transmission delay --
        # ~45% of the reported metric -- carried no policy-relevant signal).
        try:
            lo, hi = constants.MESSAGE_SIZE_KB_BY_TYPE.get(combination, (512, 1536))
            message_size = int(random.randint(lo, hi))
        except Exception:
            message_size = 1024

        # ---------------- construct request ----------------
        req = user.Request(
            request_id=int(i),
            process_id=int(i),
            combination=combination,
            message_size=message_size,
            bandwidth=constants.DEFAULT_BANDWIDTH_MBPS,
            load=np.zeros(server_count, dtype=int),
            deadline=deadline
        )

        return req

    except Exception as e:
        # ==================================================
        # HARD FAILURE: Request constructor failed
        # ==================================================
        try:
            logger.error(
                "[request_factory] Failed to construct Request. "
                f"i={i}, error={e}",
                exc_info=True
            )
        except Exception:
            pass  # logging must never break execution

        # ==================================================
        # FALLBACK 1: minimal valid Request
        # ==================================================
        try:
            return user.Request(
                request_id=int(i),
                process_id=int(i),
                combination="s",
                message_size=np.zeros(server_count, dtype=float),
                bandwidth=np.zeros(server_count, dtype=float),
                load=np.zeros(server_count, dtype=int),
                deadline=deadlines[0]
            )
        except Exception:
            # ==================================================
            # FALLBACK 2: bare minimum namespace (last resort)
            # ==================================================
            try:
                return SimpleNamespace(
                    request_id=int(i),
                    process_id=int(i),
                    combination="s",
                    arrival_time=time.time() * 1000.0,
                    deadline=deadlines[0]
                )
            except Exception:
                # ==================================================
                # FALLBACK 3: absolute last resort
                # ==================================================
                return None


def run_direct(seed: int | None, n_chunks: int, n_episodes: int,
               log_folder: str = "tools/out/smoke_logs", label: str = "run"):
    """
    [SAFETAIL][MAIN][RUN] Socket-free, seeded, in-process run.

    Bypasses the TCP sender/receiver entirely -- chunks are generated here and
    handed straight to Controller.send_to_server(). That removes thread/socket
    nondeterminism AND the ~500 s of inter-burst sleeps a full socket run spends,
    so a 15k-request run finishes in minutes instead of half an hour.

    `--smoke` (plan.md 9.4) is this with small numbers: ~20 episodes / ~300
    requests / no plots, used by gates G2/G3/G4.
    """
    seed_rep = seed_everything(seed)
    print(f"[SAFETAIL][MAIN][{label}] seeding -> {seed_rep}")

    constants.training_log_folder = log_folder
    constants.original_training_log_folder = log_folder
    Path(log_folder).mkdir(parents=True, exist_ok=True)

    controller = Controller(num_servers=constants.beta)
    controller.expected_episodes = n_episodes
    controller.plot_every_n_episodes = 10 ** 12  # we make our own figures (tools/make_figures.py)

    print(f"[SAFETAIL][MAIN][{label}] feeding {n_chunks} chunks "
          f"({n_chunks * constants.chunk_size} requests), target {n_episodes} episodes -> {log_folder}")

    rid = 0
    t0 = time.time()
    for c in range(n_chunks):
        reqs = [request_factory(rid + k) for k in range(constants.chunk_size)]
        rid += constants.chunk_size
        chunk = np.array([r for r in reqs if r is not None], dtype=object)
        now_ms = time.time() * 1000.0
        for r in chunk:
            r.arrival_time = now_ms
        controller.send_to_server(chunk)
        if controller.training_done.is_set():
            break
        if c and c % 200 == 0:
            el = time.time() - t0
            print(f"[SAFETAIL][MAIN][{label}] chunk {c}/{n_chunks} "
                  f"ep {controller.current_episode} | {el:.0f}s elapsed, "
                  f"ETA {el / c * (n_chunks - c):.0f}s", flush=True)

    dt = time.time() - t0
    print(f"[SAFETAIL][MAIN][{label}] done: {controller.current_episode} episodes, "
          f"{rid} requests, {dt:.1f}s wall, dropped={controller.dropped_requests}")
    return controller


def run_smoke(seed: int | None, log_folder: str = "tools/out/smoke_logs"):
    """plan.md 9.4 smoke: run_direct at smoke scale."""
    return run_direct(seed, constants.SMOKE_CHUNKS, constants.SMOKE_EPISODES,
                      log_folder=log_folder, label="smoke")


def main():
    parser = argparse.ArgumentParser(description="SafeTail 2.0 (heterogeneous) runner")
    parser.add_argument("--smoke", action="store_true",
                        help="short deterministic socket-free run (plan.md 9.4)")
    parser.add_argument("--run", action="store_true",
                        help="full socket-free run; size it with --chunks/--episodes")
    parser.add_argument("--chunks", type=int, default=3045,
                        help="chunks to feed (chunk_size=5) -- 3045 => 15,225 requests, matching reference_v0")
    parser.add_argument("--episodes", type=int, default=1015,
                        help="episode budget (chunks_per_episode=3)")
    parser.add_argument("--out", type=str, default=None, help="log folder for this run")
    parser.add_argument("--label", type=str, default="run", help="label used in progress lines")
    parser.add_argument("--seed", type=int, default=None,
                        help="master seed (overrides SAFETAIL_SEED / constants.SEED)")
    args, _ = parser.parse_known_args()

    smoke = args.smoke or constants.SMOKE
    seed = args.seed if args.seed is not None else constants.SEED

    if args.run:
        return run_direct(seed, args.chunks, args.episodes,
                          log_folder=args.out or f"results/{args.label}", label=args.label)

    if smoke:
        return run_smoke(seed, log_folder=args.out or "tools/out/smoke_logs")

    if seed is not None:
        print(f"[SAFETAIL][MAIN] seeding -> {seed_everything(seed)}")

    # --------------- setup controller ---------------
    controller = Controller(num_servers=constants.beta)  # Change number of servers accordingly
    # ---------------- setup receiver ----------------
    receiver = Receiver(persist_chunks=True, process_time_per_chunk=0.2, controller=controller)

    controller.run()

    # ---------------- create sender ----------------
    sender = SenderBursts(
        arr=None,
        sample_count=constants.total_no_request,  # total requests
        chunk_size=constants.chunk_size,  # requests per chunk
        bursts=constants.no_of_burst,
        min_burst=constants.min_burst,
        max_burst=constants.max_burst,
        min_interval=constants.min_interval,
        max_interval=constants.max_interval,
        jitter=constants.jitter,
        request_factory=request_factory,
        host=constants.receiver_host,
        port=constants.receiver_port,
    )

    # ---------------- run sender (blocking) ----------------
    try:
        print("[MAIN] Starting sender bursts...")
        stats = sender.run()
        print("[MAIN] Sender finished. Stats:", stats)
    except KeyboardInterrupt:
        print("[MAIN] Sender interrupted.")
    finally:
        print("[MAIN] Waiting for training to finish...")

        try:
            controller.training_done.wait()  # BLOCKS safely
        except KeyboardInterrupt:
            print("\n[MAIN] Ctrl+C received early.")
        finally:
            print("[MAIN] Stopping receiver...")
            receiver.stop()
            receiver._thread.join(timeout=5)
            print("[MAIN] Exiting cleanly.")


if __name__ == "__main__":
    main()
