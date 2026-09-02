"""
[SAFETAIL][SEAM] Policy plug-in point. Core infrastructure.

plan.md section 8.3. This module is the ENTIRE surface that external policies
(baselines/) are allowed to touch inside src/. It:

  * defines the `Policy` protocol every pluggable scheduler implements,
  * defines `PolicyContext` -- the frozen, pre-action view of the environment a
    policy is entitled to see (plan.md section 8.4),
  * holds a name -> factory registry,
  * provides subset <-> action-index helpers matching agent.get_subsets().

Hard constraints (enforced by gate G4, plan.md section 11):
  * stdlib only. No third-party imports. No import of anything under baselines/.
  * must remain importable and functional with baselines/ deleted.
  * src/ must never import baselines/. The dependency direction is one-way:
    baselines/register.py imports THIS module and calls register(); src/ only
    ever calls get()/available().

If nothing registers a policy, `constants.POLICY` stays "native" and the
controller runs its built-in DQN / heuristic path unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Protocol, Sequence, runtime_checkable


# --------------------------------------------------------------------------- #
# Context object -- the read-only, pre-action environment view (plan.md 8.4)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PolicyContext:
    """
    Everything a scheduling policy is allowed to see BEFORE it acts, and nothing
    that is only known after. Keeping post-action fields out of here makes the
    D-04 class of state/outcome leak structurally impossible for any policy that
    uses the seam.

    Index convention: every per-server sequence is length `beta`, 0-based, in
    server order 0..beta-1 (i.e. server{1..5} -> index 0..4).
    """

    request_id: int
    request_type: str                      # original letter in {s,d,p} (S-14, D-19)
    deadline: tuple[float, float]          # (D1, D2) in ms
    message_size: float
    bandwidth: float
    beta: int

    free_slots: Sequence[int]             # per server: free slots, -1 == full
    est_delay: Sequence[float]            # per server: phase-2 total est delay (s)
    est_components: Sequence[tuple[float, float, float]]   # (comp, prop, trans) s
    server_static: Sequence[Dict[str, Any]]   # per server: capacities (RAM/cores/GPU)
    server_dynamic: Sequence[Dict[str, Any]]  # per server: current utilisation

    arrival_time: float                   # ms
    episode_index: int = 0
    step_index: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Policy protocol
# --------------------------------------------------------------------------- #
@runtime_checkable
class Policy(Protocol):
    name: str

    def select(self, ctx: PolicyContext) -> Sequence[int]:
        """Return a non-empty sequence of 0-based server indices to dispatch to."""

    def observe(self, ctx: PolicyContext, action: Sequence[int], reward: float) -> None:
        """Feed back the realised reward. No-op for stateless policies."""

    def finish_episode(self, episode_index: int) -> None:
        """Episode boundary hook (train/decay/checkpoint). No-op if not needed."""


class BasePolicy:
    """Convenience base: stateless no-op hooks. Baselines may subclass or not."""

    name = "base"

    def select(self, ctx: PolicyContext) -> Sequence[int]:  # pragma: no cover
        raise NotImplementedError

    def observe(self, ctx: PolicyContext, action: Sequence[int], reward: float) -> None:
        return None

    def finish_episode(self, episode_index: int) -> None:
        return None


# --------------------------------------------------------------------------- #
# Subset <-> action-index helpers (must match agent.get_subsets ordering)
# --------------------------------------------------------------------------- #
def _all_subsets(beta: int) -> list[list[int]]:
    """Non-empty subsets of range(beta) in agent.get_subsets() bitmask order."""
    out: list[list[int]] = []
    for mask in range(2 ** beta):
        subset = [k for k in range(beta) if mask & (1 << k)]
        out.append(subset)
    return out[1:]  # drop the empty set, exactly as agent.get_subsets does


def subset_to_index(subset: Sequence[int], beta: int) -> int:
    """Inverse of agent.get_subsets(): 0-based server indices -> action index."""
    s = sorted(int(i) for i in subset)
    if not s or s[0] < 0 or s[-1] >= beta:
        raise ValueError(
            f"[SAFETAIL][SEAM][INVARIANT] subset {list(subset)} out of range for beta={beta}"
        )
    mask = 0
    for i in s:
        mask |= (1 << i)
    return mask - 1  # subsets[1:] shifts every index down by one


def index_to_subset(action_index: int, beta: int) -> list[int]:
    subsets = _all_subsets(beta)
    if not (0 <= action_index < len(subsets)):
        raise ValueError(
            f"[SAFETAIL][SEAM][INVARIANT] action_index {action_index} out of range "
            f"[0,{len(subsets)}) for beta={beta}"
        )
    return subsets[action_index]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_REGISTRY: Dict[str, Callable[[], Policy]] = {}


def register(name: str, factory: Callable[[], Policy]) -> None:
    """Register a zero-arg factory producing a Policy. Idempotent re-register
    with the same object is allowed; a conflicting re-register raises."""
    if not callable(factory):
        raise TypeError(f"[SAFETAIL][SEAM] factory for {name!r} is not callable")
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not factory:
        raise KeyError(
            f"[SAFETAIL][SEAM] policy {name!r} already registered to a different factory"
        )
    _REGISTRY[name] = factory


def get(name: str) -> Callable[[], Policy]:
    """Return the factory for `name`, or raise KeyError listing known names."""
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(
            f"[SAFETAIL][SEAM] unknown policy {name!r}. "
            f"Known policies: {known}. "
            f"Did baselines/register.py run? src/ never imports baselines/, so the "
            f"entry point (baselines/run_baseline.py) must import it before main()."
        ) from None


def available() -> list[str]:
    return sorted(_REGISTRY)


def clear() -> None:
    """Test hook only."""
    _REGISTRY.clear()
