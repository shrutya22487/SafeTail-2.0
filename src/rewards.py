"""
[SAFETAIL][REWARD] Reward primitives, factored out of controller.py (plan.md 10.4).

Currently holds the tau-referenced tail-latency reward (M-03 / B4). The headroom
product + redundancy-cost collapse still lives in Controller (compute_step_reward
/ _collapse_step_reward) because it needs the per-server request dicts; move it
here too if controller.py is split further.

`baselines/safetail_v1/reward_v1.py` re-exports `tau_reward_5case` -- src -> baselines
is the ALLOWED import direction (baselines never imported by src).
"""
from __future__ import annotations

import math

# 1.0's `abs(obs_latency - tau) < 1000` guard band.
TAU_BAND = 1000.0


def tau_reward_5case(obs_latency: float, tau: float, action_size: int,
                     beta: int, alpha: float) -> tuple[float, bool]:
    """
    SafeTail 1.0's tau-referenced 5-case asymmetric penalty (ST Def. 4.3 / Eq. 5,
    plan.md 14.1). This is the ONLY place tail latency enters a SafeTail reward
    (M-03: the 2.0 headroom reward never references latency at all).

        lam   = obs_latency - tau
        gamma = |A| - 1                        # redundant-server count
        delta = alpha * exp(-lam) if lam < 0 else alpha * exp(+lam)

        lam == 0                           ->  0
        lam  > 0 and (beta - gamma) == 1   ->  0
        lam  > 0 and (beta - gamma)  > 1   -> -exp(beta - gamma - 1) * delta
        lam  < 0                           -> -exp(gamma) * delta

    R <= 0 always. Late => penalty scaled by how FEW servers were used; early =>
    penalty scaled by how MANY. This IS the redundancy pricing 2.0 lacks
    (M-04/D-07) -- B3's c_red is the 2.0-native analogue.

    Returns (reward, out_of_band). out_of_band marks the V1-BUG-01 case where 1.0
    returned None into the replay buffer; here it is 0.0 + this flag.
    """
    lam = float(obs_latency) - float(tau)
    if abs(lam) >= TAU_BAND:
        return 0.0, True

    gamma = action_size - 1
    delta = alpha * math.exp(-lam) if lam < 0 else alpha * math.exp(lam)

    if lam == 0:
        r = 0.0
    elif lam > 0 and (beta - gamma) == 1:
        r = 0.0
    elif lam > 0 and (beta - gamma) > 1:
        r = -math.exp(beta - gamma - 1) * delta
    elif lam < 0:
        r = -math.exp(gamma) * delta
    else:
        r = 0.0
    return float(r), False
