import pickle
import random
import time
from pathlib import Path
from typing import Any, Union

import pandas as pd

import constants
import user
from regressors import load_all  # [SAFETAIL][REGRESSOR][FIX][D-02] one parameterised predictor

MAX_CONCURRENT_REQUESTS = 4


class Server:
    def __init__(self, server_index):
        base_dir = Path(__file__).resolve().parent  # Points to src/

        # Load propagation delays
        propagation_file = base_dir.parent / "dataset" / "propagation_delays.pkl"
        if not propagation_file.exists():
            raise FileNotFoundError(f"Propagation delays file not found: {propagation_file}")

        with open(propagation_file, 'rb') as f:
            self.propagation_delays = pickle.load(f)

        # Load server dataset path
        self.server_data_path = base_dir.parent / "dataset" / f"server{server_index}.csv"
        if not self.server_data_path.exists():
            raise FileNotFoundError(f"Server dataset CSV not found: {self.server_data_path}")

        self.server_data = pd.read_csv(self.server_data_path)
        self.server_data.columns = [c.strip() for c in self.server_data.columns]

        self.server_index = server_index
        self.num_requests = 0
        self.requests = []
        self.active_requests = []

        self._load_predictors(server_index)

    def _load_predictors(self, server_index: int):
        """
        [SAFETAIL][REGRESSOR][FIX][D-02][D-02b][D-02c]
        Load this server's OWN {detect,speech,predict} trace predictors.

        Was `_load_predictors_from_regressor_folder`: it `sys.path.insert`ed each
        `src/server{i}_regressor/` dir and `import_module("detect_predictor")` --
        identical module names meant `sys.modules` returned server 1's class for
        every server (D-02b), and every failure fell through to a contention-free
        single-letter CSV lookup (D-02c). Now there is one module and the server
        index is an argument.

        A load failure RAISES unless constants.ALLOW_DEGRADED_PREDICTORS is True,
        in which case it is logged with [DEGRADED][D-02c] and counted in the run
        manifest (a run with any [DEGRADED] count > 0 is not publishable -- G5).
        """
        allow = bool(getattr(constants, "ALLOW_DEGRADED_PREDICTORS", False))
        self.predictors = load_all(server_index, allow_degraded=allow)

    def print_active_requests(self):
        print(f"Server {self.server_index} Active Requests:")
        for ar in self.active_requests:
            start = ar['start_time']
            finish = ar['finish_time']
            proc = ar['proc_time']
            print(
                f" Combination: {ar['request'].combination}, Start: {start:.2f}, Finish: {finish:.2f}, Proc Time: {proc:.6f}")

    def _get_propogation_delay(self):
        return random.choice(self.propagation_delays[self.server_index - 1])

    def _get_transmission_delay(self, message_size_kb=None, bandwidth_mbps=None):
        """
        [SAFETAIL][SERVER][FIX][D-12][D-34] Transmission delay as a function of
        payload size and link bandwidth, not the old
        `random.choice([18.5,19.2,20,21.5,22])/1000` (5 fixed values, independent
        of server, message_size and bandwidth -- ~45% of the reported metric was
        this coin flip).

        Model (SafeTail 1.0 / ST IV): t = 8*KB/up_kbps + 8*KB/dn_kbps, up = dn =
        bandwidth, plus a small per-server multiplicative link jitter.
        """
        if getattr(constants, "LEGACY_TRANSMISSION", False):
            # [SAFETAIL][LEGACY][D-12][D-34] the pre-fix 5-value coin flip,
            # independent of server / payload size / bandwidth.
            return random.choice([18.5, 19.2, 20, 21.5, 22]) / 1000

        ms = float(message_size_kb) if message_size_kb else float(constants.DEFAULT_MESSAGE_SIZE_KB)
        bw = float(bandwidth_mbps) if bandwidth_mbps and float(bandwidth_mbps) > 0 \
            else float(constants.DEFAULT_BANDWIDTH_MBPS)
        bits_kb = 8.0 * ms
        link_kbps = bw * 1000.0
        base = bits_kb / link_kbps + bits_kb / link_kbps          # uplink + downlink, seconds
        j = random.uniform(-constants.LINK_JITTER_FRAC, constants.LINK_JITTER_FRAC)
        return max(1e-6, base * (1.0 + j))

    # back-compat alias (old misspelled name, no args)
    def _get_tramission_delay(self):
        return self._get_transmission_delay()

    def _collect_visible_types(self):
        sorted_active = sorted(self.active_requests, key=lambda ar: ar['start_time'])
        types = []
        for ar in sorted_active:
            try:
                comb = getattr(ar['request'], 'combination', '')
                comb_str = str(comb).strip() if comb is not None else ''
                types.append(comb_str[0] if comb_str else '')
            except Exception:
                types.append('')
        return types

    def _csv_lookup(self, combined_str: str):
        """
        [SAFETAIL][SERVER][FIX][D-02c] CSV fallback keyed on the FULL contention
        string, never on the bare letter. Returns the trace's Total Processing
        Time for that exact contention row, or None if absent. The old code
        matched `Combination == letter`, i.e. it silently returned the
        CONTENTION-FREE latency whenever a predictor failed.
        """
        key = str(combined_str).strip().lower()
        mask = self.server_data['Combination'].astype(str).str.strip().str.lower() == key
        if mask.any():
            t = self.server_data[mask].iloc[0].get("Total Processing Time (sec)", None)
            if t is not None and not pd.isna(t):
                return float(t)
        return None

    def _predict_using_letter(self, letter: str, combined_str: str) -> float:
        if not letter:
            return 0.0
        allow = bool(getattr(constants, "ALLOW_DEGRADED_PREDICTORS", False))

        predictor = self.predictors.get(letter.lower())
        if predictor is None:
            # only reachable when ALLOW_DEGRADED_PREDICTORS made load_all() tolerant
            val = self._csv_lookup(combined_str)
            if val is not None:
                return val
            raise RuntimeError(
                f"[SAFETAIL][SERVER][DEGRADED][D-02c] server={self.server_index} letter={letter!r}: "
                f"no predictor and no trace row for contention string {combined_str!r}"
            )

        try:
            return float(predictor.predict_from_combination(combined_str))
        except Exception as exc:
            val = self._csv_lookup(combined_str)
            if val is not None:
                if allow:
                    print(f"[SAFETAIL][SERVER][DEGRADED][D-02c] server={self.server_index} "
                          f"predictor raised for {combined_str!r}; using trace row. ({exc})")
                    try:
                        from _safetail_log import note_degraded
                        note_degraded("D-02c")
                    except Exception:
                        pass
                    return val
            raise RuntimeError(
                f"[SAFETAIL][SERVER][DEGRADED][D-02c] server={self.server_index} letter={letter!r} "
                f"combined={combined_str!r}: predictor failed and no trace row"
            ) from exc

    def _choose_first_letter_for_regressor(self, request):
        existing_types = self._collect_visible_types()
        try:
            new_comb = str(getattr(request, 'combination', '')).strip()
            new_letter = new_comb[0] if new_comb else ''
        except Exception:
            new_letter = ''

        if new_letter == '':
            return '', ''

        rev_existing = "".join([t for t in reversed(existing_types) if t])
        combined_string = new_letter + rev_existing
        first_letter = combined_string[0] if combined_string else ''
        return first_letter, combined_string

    # ---------- Public API ----------
    def compute_request_time(self, request: user.Request, reuse: bool = False) -> tuple[
        Union[float, Any], str, float, Union[float, Any], Union[float, Any]]:
        """
        Compute total delay for `request` WITHOUT scheduling it.

        [SAFETAIL][SERVER][FIX][D-18] `reuse=True` returns the estimate this
        server already produced for this request during phase 2, instead of
        drawing FRESH propagation / transmission samples. Previously phase 2
        (what the policy sees) and phase 7 (schedule_request, what it is graded
        on) each drew their own samples, so the agent was scored on numbers it
        never saw. The controller calls this with reuse=False in phase 2 and the
        scheduling path calls it with reuse=True.
        """
        cache = getattr(request, "_est_by_server", None)
        if cache is None:
            cache = request._est_by_server = {}
        if getattr(constants, "LEGACY_DOUBLE_DRAW", False):
            reuse = False  # [SAFETAIL][LEGACY][D-18] pre-fix: phase 2 and phase 7 each draw
        if reuse and self.server_index in cache:
            c = cache[self.server_index]
            return (c["total"], c["combined_str"], c["comp"], c["prop"], c["trans"])

        # 1) propagation
        propagation_delay_for_node = self._get_propogation_delay()

        # 2) transmission -- now f(payload size, link bandwidth)  (D-12, D-34)
        try:
            transmission_delay_for_node = self._get_transmission_delay(
                message_size_kb=getattr(request, "message_size", None),
                bandwidth_mbps=getattr(request, "bandwidth", None),
            )
        except Exception as e:
            print(f"[SERVER]    [ERROR] Unexpected error in transmission delay calculation: {e}")
            transmission_delay_for_node = float('inf')

        # 3) computation: determine first letter and use predictor
        first_letter, combined_str = self._choose_first_letter_for_regressor(request)
        computation_delay_for_node = self._predict_using_letter(first_letter, combined_str)

        total_delay = propagation_delay_for_node + transmission_delay_for_node + computation_delay_for_node

        cache[self.server_index] = {
            "total": total_delay, "combined_str": combined_str,
            "comp": computation_delay_for_node,
            "prop": propagation_delay_for_node,
            "trans": transmission_delay_for_node,
        }
        return (
            total_delay,
            combined_str,
            computation_delay_for_node,
            propagation_delay_for_node,
            transmission_delay_for_node,
        )

    def schedule_request(self, request: user.Request, current_time: float = None, do_sleep: bool = False):
        """
        Compute time, schedule request if capacity, return (success, finish_time_or_reason, proc_time).
        """
        if current_time is None:
            current_time = time.time()

        self.update_active_requests(current_time=current_time)
        if self.num_requests >= MAX_CONCURRENT_REQUESTS:
            (total_delay,
             combined_str,
             computation_delay_for_node,
             propagation_delay_for_node,
             transmission_delay_for_node) = self.compute_request_time(request, reuse=True)
            return False, "server full", total_delay, combined_str, computation_delay_for_node, propagation_delay_for_node, transmission_delay_for_node

        (total_delay,
         combined_str,
         computation_delay_for_node,
         propagation_delay_for_node,
         transmission_delay_for_node) = self.compute_request_time(request, reuse=True)  # [D-18]
        start_time = current_time
        finish_time = start_time + total_delay

        # [SAFETAIL][SERVER][FIX][D-19] record contention string WITHOUT clobbering
        # the request-type letter.
        request.contention_str = combined_str

        self.requests.append(request)
        self.active_requests.append({
            'request': request,
            'start_time': start_time,
            'proc_time': total_delay,
            'finish_time': finish_time
        })
        self.num_requests += 1

        if do_sleep and total_delay > 0:
            time.sleep(total_delay)
            self.update_active_requests(current_time=time.time())

        return True, finish_time, total_delay, combined_str, computation_delay_for_node, propagation_delay_for_node, transmission_delay_for_node

    # ---------- Remaining helpers ----------
    def update_active_requests(self, current_time: float = None):
        if current_time is None:
            current_time = time.time()
        freed = 0
        remaining = []
        for ar in self.active_requests:
            if ar['finish_time'] <= current_time:
                freed += 1
                try:
                    self.requests.remove(ar['request'])
                except ValueError:
                    pass
                self.num_requests = max(0, self.num_requests - 1)
            else:
                remaining.append(ar)
        self.active_requests = remaining
        return freed

    def check_server_availability(self, current_time: float = None):
        if current_time is None:
            current_time = time.time()
        self.update_active_requests(current_time=current_time)

        if self.num_requests < MAX_CONCURRENT_REQUESTS:
            return self.num_requests
        else:
            return -1

    def time_until_next_free(self, current_time: float = None):
        if current_time is None:
            current_time = time.time()
        self.update_active_requests(current_time=current_time)
        if self.num_requests < MAX_CONCURRENT_REQUESTS:
            return 0.0
        visible_active = len(self.active_requests)
        hidden_count = max(0, self.num_requests - visible_active)
        if hidden_count > 0:
            return None
        earliest = min(ar['finish_time'] for ar in self.active_requests)
        return max(0.0, earliest - current_time)
