import random
import numpy as np
import pickle
import pandas as pd
import time
import user
from pathlib import Path
import computation_delay_regressor
import sys


TIME_SCALE = 10.0  # factor to speed up time in simulation

# TODO(@shrutya22487): change the server indexing, time scaling etc if needed
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
        self.server_index = server_index
        self.num_requests = 0
        self.requests = []
        self.active_requests = []

    def print_active_requests(self):
        print(f"Server {self.server_index} Active Requests:")
        for ar in self.active_requests:
            req = ar['request']
            start = ar['start_time']
            finish = ar['finish_time']
            proc = ar['proc_time']
            print(f"  Request Process ID: {req.process_id}, Start: {start:.2f}, Finish: {finish:.2f}, Proc Time: {proc:.2f}")

    def _get_propogation_delay(self):
        return random.choice(self.propagation_delays[self.server_index - 1])
    
    def _get_tramission_delay(self, message_size, upload_bandwidth, download_bandwidth):
        message_size = 8 * message_size
        uplink = upload_bandwidth * 1000
        downlink = download_bandwidth * 1000
        return (message_size / uplink) + (message_size / downlink)

    def _get_computation_delay(self, process_row_num):
        return computation_delay_regressor.predict_rows(row_num=process_row_num, server_index=self.server_index)

    def _cpu_mean_from_str(self, cpu_str):
        """
        cpu_str is expected like "2.5, 2.5, 40.4, 18.7, ..."
        returns mean of parsed floats, or 0.0 if parsing fails / empty.
        """
        if not isinstance(cpu_str, str) or cpu_str.strip() == "":
            return 0.0
        try:
            parts = [p.strip() for p in cpu_str.split(',') if p.strip() != '']
            nums = [float(p) for p in parts]
            if len(nums) == 0:
                return 0.0
            return float(np.mean(nums))
        except Exception:
            return 0.0

    def _choose_combination_row_for_request(self, request):
        """
        Inspect currently visible active requests, build the 'combination' string
        (existing requests in start_time order, then the new request's type),
        then find matching rows in self.server_data and pick the row with the
        maximum (mean CPU usage + GPU usage) score. Returns the selected
        row index (integer) or None if fallback required.
        """
        if len(self.active_requests) == 0:
            return request.process_id
        # 1) get types of currently visible active requests in order of start_time
        existing_types = []
        # Sort visible actives by start_time to ensure deterministic ordering
        sorted_active = sorted(self.active_requests, key=lambda ar: ar['start_time'])
        for ar in sorted_active:
            try:
                proc_id = ar['request'].process_id
                # ensure proc_id index exists
                if 0 <= int(proc_id) < len(self.server_data):
                    comb = str(self.server_data.iloc[int(proc_id)]['Combination'])
                    # if Combination is a multi-character string, treat its first char as the "type"
                    # but from your description, single-combination processes are single-letter like "s" or "p".
                    # We'll append the whole combination entry for safety, but typical rows are single-letter.
                    # However we only want the 'type' that corresponding to the process - which should be a single char.
                    existing_types.append(comb)
                else:
                    existing_types.append("")  # unknown
            except Exception:
                existing_types.append("")

        # 2) determine the new request type (from its process_id row)
        new_type = ""
        try:
            new_pid = int(request.process_id)
            if 0 <= new_pid < len(self.server_data):
                new_type = str(self.server_data.iloc[new_pid]['Combination'])
            else:
                new_type = ""
        except Exception:
            new_type = ""

        # Build the combined string: concatenate existing types in order then the new_type
        # Example: existing 'p' and new 's' => combined 'p' + 's' -> 'ps'
        # If existing entries are themselves multi-letter combos, we concatenate them as-is.
        combined_string = "".join(existing_types) + new_type

        # If combined_string is empty or yields no matches, fall back to original single process row
        if combined_string == "":
            return None

        # 3) find matching rows in the server data by 'Combination'
        # Make sure column exists
        if 'Combination' not in self.server_data.columns:
            return None

        matches = self.server_data[self.server_data['Combination'] == combined_string]
        if matches.empty:
            # no direct match found
            return None

        # 4) compute score for each candidate row and pick the row index with maximum score
        # Score = mean_cpu_usage + gpu_usage (treat NaN/parse errors as 0)
        best_idx = None
        best_score = -np.inf
        for idx, row in matches.iterrows():
            cpu_mean = self._cpu_mean_from_str(row.get('CPU Usage Per Core', ""))
            gpu = row.get('GPU Usage (%)', 0.0)
            try:
                if not np.isfinite(gpu):
                    gpu = 0.0
            except Exception:
                gpu = 0.0
            score = float(cpu_mean) + float(gpu)
            if score > best_score:
                best_score = score
                best_idx = int(idx)

        return best_idx

    def get_delays(self, request: user.Request):
        """
        Compute propagation + transmission + computation delays (orig, unscaled).

        New behavior:
          - When a new request arrives, we attempt to pick a combined-row
            from the server CSV based on currently-running visible processes
            + the incoming request. If a combined-row exists, we pass its
            row index to the computation_delay regressor (instead of
            the original request.process_id).
        """
        sys.modules['__main__'].AdvancedEnsemble = computation_delay_regressor.AdvancedEnsemble

        # propagation
        propagation_delay_for_node = self._get_propogation_delay()

        # transmission: same as before
        try:
            tramission_delay_for_node = self._get_tramission_delay(
                request.message_size,
                request.bandwidth / request.load[self.server_index-1],
                request.bandwidth / request.load[self.server_index-1]
            )
        except IndexError as e:
            print(f"[SERVER]    [ERROR] IndexError while accessing load[{self.server_index}]: {e}")
            print(f"[SERVER]     Load array size: {len(request.load)}, server_index: {self.server_index}")
            tramission_delay_for_node = float('inf')  # or some default fallback value
        except Exception as e:
            print(f"[SERVER]    [ERROR] Unexpected error in transmission delay calculation: {e}")
            tramission_delay_for_node = float('inf')  # fallback value


        # computation: choose the best combined row if possible
        selected_row = self._choose_combination_row_for_request(request)
        
        print(f"Row selected for request with process_id: {request.process_id + 2} is {selected_row + 2}", end='\n')

        if selected_row is None:
            # fallback to the request's own process_id
            proc_row_num = int(request.process_id)
        else:
            proc_row_num = int(selected_row)

        computation_delay_for_node = self._get_computation_delay(proc_row_num)
        

        return propagation_delay_for_node + tramission_delay_for_node + computation_delay_for_node

    def _scale_proc_time(self, proc_time: float) -> float:
        return float(proc_time) / TIME_SCALE

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

    def schedule_request(self, request: user.Request, current_time: float = None, do_sleep: bool = False):
        if current_time is None:
            current_time = time.time()

        if not self.check_server_availability(current_time=current_time):
            # compute proc_time even if rejected so controller can log estimate
            # orig_proc = self.get_delays(request)
            # real_proc = self._scale_proc_time(orig_proc)
            return False, "server full", 1e9

        proc_time = self.get_delays(request)
        start_time = current_time
        finish_time = start_time + proc_time

        self.requests.append(request)
        self.active_requests.append({
            'request': request,
            'start_time': start_time,
            'proc_time': proc_time,
            'finish_time': finish_time
        })
        self.num_requests += 1

        if do_sleep and proc_time > 0:
            time.sleep(proc_time)
            self.update_active_requests(current_time=time.time())

        return True, finish_time, proc_time

    def check_server_availability(self, current_time: float = None):
        if current_time is None:
            current_time = time.time()
        self.update_active_requests(current_time=current_time)
        return self.num_requests < 4

    def time_until_next_free(self, current_time: float = None):
        if current_time is None:
            current_time = time.time()
        self.update_active_requests(current_time=current_time)
        if self.num_requests < 4:
            return 0.0
        visible_active = len(self.active_requests)
        hidden_count = max(0, self.num_requests - visible_active)
        if hidden_count > 0:
            return None
        earliest = min(ar['finish_time'] for ar in self.active_requests)
        return max(0.0, earliest - current_time)
