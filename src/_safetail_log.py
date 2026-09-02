"""
[SAFETAIL][MAIN][SEAM] Central logging + debug-tag helper.

This is the debuggability contract from plan.md section 10.1:

    given a symptom -> grep finds the code.

Every non-obvious log line and code block carries a bracketed tag:

    [SAFETAIL][<COMPONENT>][<KIND>][<ID>] message

    COMPONENT : MAIN SENDER RECEIVER CONTROLLER SERVER AGENT REGRESSOR
                REWARD REPLAY SEAM POLICY PLOT AUDIT
    KIND      : TRACE STEP EPISODE STATE ACTION REWARD SCHED DEGRADED
                INVARIANT FIX SEAM PERF
    ID        : the defect / feature id -- D-04, M-02, V1-BUG-01, ...
                omit only for routine tracing.

Usage:

    from _safetail_log import get_logger, tag
    log = get_logger("CONTROLLER")
    log.debug(tag("STATE", "D-04", "s_t dim=%d"), s.size)
    log.warning(tag("DEGRADED", "D-02c", "server=%d task=%s load failed"), i, t)

Design notes:
  * stdlib `logging` only -- no third-party imports, importable everywhere.
  * level is read from env SAFETAIL_LOG_LEVEL (default INFO; smoke mode sets DEBUG).
  * TRACE-kind lines are gated behind SAFETAIL_TRACE=1 so per-request spam is
    off by default (plan.md 10.1).
  * `degraded_count` is a process-global counter. §10.3 requires that any run
    whose manifest reports a [DEGRADED] count > 0 is not publishable; call
    `note_degraded()` at every degradation site and dump `degraded_report()`
    into the run manifest.
"""
from __future__ import annotations

import logging
import os
import sys
import threading

_VALID_COMPONENTS = {
    "MAIN", "SENDER", "RECEIVER", "CONTROLLER", "SERVER", "AGENT", "REGRESSOR",
    "REWARD", "REPLAY", "SEAM", "POLICY", "PLOT", "AUDIT",
}
_VALID_KINDS = {
    "TRACE", "STEP", "EPISODE", "STATE", "ACTION", "REWARD", "SCHED",
    "DEGRADED", "INVARIANT", "FIX", "SEAM", "PERF",
}

_TRACE_ENABLED = os.environ.get("SAFETAIL_TRACE", "0") not in ("0", "", "false", "False")
_LEVEL = os.environ.get("SAFETAIL_LOG_LEVEL", "INFO").upper()

_configured = False
_lock = threading.Lock()

# --- degradation ledger (plan.md 10.3, gate G5) -----------------------------
_degraded: dict[str, int] = {}
_degraded_lock = threading.Lock()


def _configure_root() -> None:
    global _configured
    with _lock:
        if _configured:
            return
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        root = logging.getLogger("safetail")
        root.handlers[:] = [handler]
        root.setLevel(getattr(logging, _LEVEL, logging.INFO))
        root.propagate = False
        _configured = True


def get_logger(component: str) -> logging.Logger:
    """Return the logger for a COMPONENT (validated against the taxonomy)."""
    comp = component.upper()
    if comp not in _VALID_COMPONENTS:
        raise ValueError(
            f"[SAFETAIL][MAIN][INVARIANT] unknown log component {component!r}; "
            f"allowed: {sorted(_VALID_COMPONENTS)}"
        )
    _configure_root()
    return logging.getLogger(f"safetail.{comp}")


def tag(kind: str, ident: str | None, message: str) -> str:
    """
    Build a tagged message string. `ident` is the D-xx / M-xx / S-xx / V1-* id,
    or None for routine tracing.
    """
    k = kind.upper()
    if k not in _VALID_KINDS:
        raise ValueError(
            f"[SAFETAIL][MAIN][INVARIANT] unknown log kind {kind!r}; "
            f"allowed: {sorted(_VALID_KINDS)}"
        )
    parts = ["[SAFETAIL]", f"[{k}]"]
    if ident:
        parts.append(f"[{ident}]")
    return "".join(parts) + " " + message


def trace_enabled() -> bool:
    return _TRACE_ENABLED


def note_degraded(key: str, n: int = 1) -> None:
    """
    Record a degradation event (a swallowed error that changed a number, an
    explicitly-enabled fallback, ...). `key` should be the defect id, e.g.
    'D-02c'. plan.md 10.3: these MUST be surfaced in the run manifest and a run
    with any count > 0 is not publishable.
    """
    with _degraded_lock:
        _degraded[key] = _degraded.get(key, 0) + n


def degraded_report() -> dict[str, int]:
    """Snapshot of the degradation ledger, for the run manifest (gate G5)."""
    with _degraded_lock:
        return dict(_degraded)


def reset_degraded() -> None:
    with _degraded_lock:
        _degraded.clear()
