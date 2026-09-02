"""
[SAFETAIL][MAIN][B7] One place that seeds every RNG the pipeline touches.

plan.md B7: `constants.SEED`, threaded to `random`, `np.random`, and `tf.random`.
Call `seed_everything(constants.SEED)` once at process start (main.py). A value of
None is a no-op (legacy non-deterministic behaviour) so existing runs are
unaffected until a seed is explicitly requested.

Full determinism of the socket/thread harness is NOT claimed -- wall-clock
`queue_waiting_time` (D-23) and thread scheduling still vary. This seeds the
algorithmic RNGs so that, in --smoke mode (which bypasses the sockets, see
main.run_smoke), a run is reproducible enough for gates G2/G3.
"""
from __future__ import annotations

import os
import random


def seed_everything(seed: int | None) -> dict:
    """Seed random / numpy / tensorflow. Returns a small report for the manifest."""
    report: dict[str, object] = {"seed": seed}
    if seed is None:
        report["applied"] = False
        return report

    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
        report["numpy"] = True
    except Exception as exc:  # pragma: no cover
        report["numpy"] = f"failed: {exc}"

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            pass
        report["tensorflow"] = True
    except Exception as exc:  # pragma: no cover
        report["tensorflow"] = f"failed: {exc}"

    report["applied"] = True
    return report
