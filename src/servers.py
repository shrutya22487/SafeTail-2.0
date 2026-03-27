import importlib
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Union

import pandas as pd

import user

# fixed bandwidth (kbps) used for both uplink and downlink when request doesn't specify bandwidth/load
FIXED_BANDWIDTH_KBPS = 1000.0

MAX_CONCURRENT_REQUESTS = 4


class Server:
    def __init__(self, server_index):
        base_dir = Path(__file__).resolve().parent  # Points to src/

        # Load propagation delays
        propagation_file = base_dir.parent / "data" / "propagation_delays.pkl"
        if not propagation_file.exists():
            raise FileNotFoundError(f"Propagation delays file not found: {propagation_file}")

        with open(propagation_file, 'rb') as f:
            self.propagation_delays = pickle.load(f)

        # Load server data path
        self.server_data_path = base_dir.parent / "data" / f"server{server_index}.csv"
        if not self.server_data_path.exists():
            raise FileNotFoundError(f"Server data CSV not found: {self.server_data_path}")

        self.server_data = pd.read_csv(self.server_data_path)
        self.server_data.columns = [c.strip() for c in self.server_data.columns]

        self.server_index = server_index
        self.num_requests = 0
        self.requests = []
        self.active_requests = []

        # Attempt to import predictor classes from server{index}_regressor folder only
        self._load_predictors_from_regressor_folder(base_dir, server_index)

    def _load_predictors_from_regressor_folder(self, base_dir: Path, server_index: int):
        self.predictors = {'d': None, 's': None, 'p': None}
        reg_folder = base_dir / f"server{self.server_index}_regressor"
        if not reg_folder.exists() or not reg_folder.is_dir():
            return

        sys.path.insert(0, str(reg_folder))
        try:
            # detect
            try:
                detect_mod = importlib.import_module("detect_predictor")
                DetectCls = getattr(detect_mod, "DetectPredictor", None)
                if DetectCls:
                    self.predictors['d'] = DetectCls()
            except Exception:
                self.predictors['d'] = None

            # speech
            try:
                speech_mod = importlib.import_module("speech_predictor")
                SpeechCls = getattr(speech_mod, "SpeechPredictor", None)
                if SpeechCls:
                    self.predictors['s'] = SpeechCls()
            except Exception:
                self.predictors['s'] = None

            # predict
            try:
                predict_mod = importlib.import_module("predict_predictor")
                PredictCls = getattr(predict_mod, "PredictPredictor", None)
                if PredictCls:
                    self.predictors['p'] = PredictCls()
            except Exception:
                self.predictors['p'] = None

        finally:
            # keep path inserted for the runtime; remove if you prefer cleanup
            pass

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

    def _get_tramission_delay(self, message_size_bytes, upload_bandwidth_kbps, download_bandwidth_kbps):
        return random.choice([18.5, 19.2, 20, 21.5, 22]) / 1000

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

    def _predict_using_letter(self, letter: str, combined_str: str) -> float:
        if not letter:
            return 0.0

        predictor = self.predictors.get(letter.lower())
        if predictor is None:
            # fallback to CSV value if available
            try:
                mask = self.server_data['Combination'].astype(str).str.strip().str.lower() == letter.lower()
                if mask.any():
                    row = self.server_data[mask].iloc[0]
                    t = row.get("Total Processing Time (sec)", None)
                    if t is not None and not pd.isna(t):
                        return float(t)
            except Exception:
                pass
            return 0.0

        try:
            # print(f"\n[SERVER]    Using predictor for letter '{letter}' with combined string '{combined_str}'\n")
            return float(predictor.predict_from_combination(combined_str))
        except Exception:
            # fallback to CSV
            try:
                mask = self.server_data['Combination'].astype(str).str.strip().str.lower() == letter.lower()
                if mask.any():
                    row = self.server_data[mask].iloc[0]
                    t = row.get("Total Processing Time (sec)", None)
                    if t is not None and not pd.isna(t):
                        return float(t)
            except Exception:
                pass
            return 0.0

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
    def compute_request_time(self, request: user.Request) -> tuple[
        Union[float, Any], str, float, Union[float, Any], Union[float, Any]]:
        """
        Compute and return total delay for `request` WITHOUT scheduling it.
        Transmission uses FIXED_BANDWIDTH_KBPS when request does not provide load/bandwidth.
        """
        # 1) propagation
        propagation_delay_for_node = self._get_propogation_delay()

        # 2) transmission (use fixed bandwidth; request may not have load or bandwidth)
        try:
            # If request has a bandwidth attribute and load (deprecated), we ignore it per new requirement.
            upl = FIXED_BANDWIDTH_KBPS
            down = FIXED_BANDWIDTH_KBPS
            tramission_delay_for_node = self._get_tramission_delay(
                getattr(request, 'message_size', 0),
                upl,
                down
            )
        except Exception as e:
            print(f"[SERVER]    [ERROR] Unexpected error in transmission delay calculation: {e}")
            tramission_delay_for_node = float('inf')

        # 3) computation: determine first letter and use predictor
        first_letter, combined_str = self._choose_first_letter_for_regressor(request)
        computation_delay_for_node = self._predict_using_letter(first_letter, combined_str)

        # print("propagation_delay_for_node: ", propagation_delay_for_node *1000)
        # print("transmission_delay_for_node: ", tramission_delay_for_node*1000)
        # print("computation_delay_for_node: ", computation_delay_for_node*1000)
        # print("computation_delay_for_node: ", computation_delay_for_node*1000)

        total_delay = propagation_delay_for_node + tramission_delay_for_node + computation_delay_for_node
        return (
            total_delay,
            combined_str,
            computation_delay_for_node,
            propagation_delay_for_node,
            tramission_delay_for_node,
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
             tramission_delay_for_node) = self.compute_request_time(request)
            return False, "server full", total_delay, combined_str, computation_delay_for_node, propagation_delay_for_node,tramission_delay_for_node

        (total_delay,
         combined_str,
         computation_delay_for_node,
         propagation_delay_for_node,
         tramission_delay_for_node) = self.compute_request_time(request)
        start_time = current_time
        finish_time = start_time + total_delay

        request.combination = combined_str

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

        return True, finish_time, total_delay, combined_str, computation_delay_for_node, propagation_delay_for_node, tramission_delay_for_node

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

        if (self.num_requests < MAX_CONCURRENT_REQUESTS):
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
