
"""
user.py

Provides Request and Users classes and a small __main__ that prints first requests for verification.

"""
from typing import List, Dict, Any, Optional
import sys
import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import constants

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Request:
    def __init__(
        self,
        request_id: int,
        process_id: int,
        message_size: float = 0.0,
        bandwidth: float = 0.0,
        load: Optional[List[int]] = None,
        ram_usage: float = 0.0,
        cpu_usage: Any = None,
        arrival_time: float = 0.0,
        duration: float = 0.0,
        extras: Optional[Dict[str, Any]] = None,
        gpu_usage: Optional[int] = None,
        cpu_model: Optional[str] = None,
        gpu_model: Optional[str] = None,
        cpu_clock: Optional[float] = None,
        gpu_clock: Optional[float] = None,
        time_util: Optional[List[float]] = None,
    ):
        # actual important members
        self.request_id = int(request_id)
        self.process_id = int(process_id)
        self.message_size = float(message_size)
        self.bandwidth = float(bandwidth)
        self.load = list(load) if load is not None else []
        self.ram_usage = float(ram_usage)
        self.cpu_usage = cpu_usage
        self.arrival_time = float(arrival_time)
        self.duration = float(duration)
        self.extras = dict(extras) if extras is not None else {}

        # additional members
        self.gpu_usage = int(gpu_usage) if gpu_usage is not None and str(gpu_usage) != "" else None
        self.cpu_model = cpu_model
        self.gpu_model = gpu_model
        self.cpu_clock = float(cpu_clock) if cpu_clock is not None and str(cpu_clock) != "" else None
        self.gpu_clock = float(gpu_clock) if gpu_clock is not None and str(gpu_clock) != "" else None
        self.time_util = list(time_util) if time_util is not None else []

    #this is to create old state list keeping it now will change
    def to_state(self) -> Dict[str, Any]:
        load_arr = np.array(self.load if self.load else [], dtype=int)
        return {
            "LOAD": load_arr,
            "MESSAGE_SIZE": float(self.message_size),
            "BANDWIDTH": float(self.bandwidth),
        }

    #added for easy print of request object
    def as_dict(self) -> Dict[str, Any]:
        d = {
            "request_id": self.request_id,
            "process_id": self.process_id,
            "message_size": self.message_size,
            "bandwidth": self.bandwidth,
            "load": self.load,
            "ram_usage": self.ram_usage,
            "cpu_usage": self.cpu_usage,
            "arrival_time": self.arrival_time,
            "duration": self.duration,
            "extras": self.extras,
            "gpu_usage": self.gpu_usage,
            "cpu_model": self.cpu_model,
            "gpu_model": self.gpu_model,
            "cpu_clock": self.cpu_clock,
            "gpu_clock": self.gpu_clock,
            "time_util": self.time_util,
        }
        return d

    def __repr__(self) -> str:
        return (
            f"<Request id={self.request_id} pid={self.process_id} msg={self.message_size}B "
            f"load_len={len(self.load)} ram={self.ram_usage} cpu={self.cpu_usage}>"
        )


class Users:
    
    
    DEFAULT_DETECT_CSV =  './data/updated_Detect.csv'
    DEFAULT_SERVER_STATE_CSV =  './data/server_state.csv'

    def __init__(
        self,
        number_of_requests: int,
        detect_csv: Optional[str] = None,
        server_state_csv: Optional[str] = None,
    ):
        self.number_of_requests = int(number_of_requests)
        self.detect_csv = Path(detect_csv) if detect_csv else self.DEFAULT_DETECT_CSV
        self.server_state_csv = Path(server_state_csv) if server_state_csv else self.DEFAULT_SERVER_STATE_CSV

        if not self.detect_csv.exists():
            raise FileNotFoundError(f"Detect CSV not found: {self.detect_csv}")
        if not self.server_state_csv.exists():
            raise FileNotFoundError(f"Server state CSV not found: {self.server_state_csv}")

        self.detect_df = pd.read_csv(str(self.detect_csv))
        self.server_state_df = pd.read_csv(str(self.server_state_csv))

        available = min(len(self.detect_df), len(self.server_state_df))
        if self.number_of_requests > available:
            logger.warning(
                "Requested %d requests but only %d rows available; capping to %d.",
                self.number_of_requests, available, available,
            )
            self.number_of_requests = available

        self.requests: List[Request] = []
        self._cursor = 0
        self._build_requests()

    #added two functions purely for cleaning and parsing the data 
    # -------------------------
    @staticmethod
    def _parse_message_size(raw: Any) -> float:
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return 0.0
        try:
            return float(raw)
        except Exception:
            s = str(raw).strip()
            
            m = re.search(r"(\d+)\s*[xX]\s*(\d+)", s) #this checks for format {number:width} x or X {number:height} and stores them separately
            if m:
                w = int(m.group(1)) 
                h = int(m.group(2))
                return float(w * h * 3)
            
            
            m2 = re.search(r"(\d+)", s) #this for case when we have straight width*height in csv
            if m2:
                return float(int(m2.group(1)))
        return 0.0
    
    #added two functions purely for cleaning and parsing the data 
    @staticmethod
    def _server_row_to_list(server_row: Dict[str, Any]) -> List[int]:
        kv = []
        for k, v in server_row.items():
            if str(k).strip().lower() == "timestamp":
                continue
            m = re.search(r"(\d+)\s*$", str(k))
            idx = int(m.group(1)) if m else None
            kv.append((idx if idx is not None else float("inf"), k, v))
        kv.sort(key=lambda x: (x[0], x[1]))
        res = []
        for _, _, v in kv:
            try:
                res.append(int(v))
            except Exception:
                try:
                    res.append(int(float(v)))
                except Exception:
                    res.append(0)
        return res


    #actual request builder
    def _build_requests(self) -> None:
        for i in range(self.number_of_requests):
            det_row = self.detect_df.iloc[i]
            ss_row = self.server_state_df.iloc[i]

            request_id = int(i)
            process_id = int(i)

            # message size
            message_size = 0.0
            if "Image Pixel" in det_row.index:
                message_size = self._parse_message_size(det_row["Image Pixel"])
            elif "Image Pix" in det_row.index:
                message_size = self._parse_message_size(det_row["Image Pix"])

            # bandwidth from constants
            bandwidth = float(getattr(constants, "max_bandwidth", 200))

            # load list
            server_state_list = self._server_row_to_list(ss_row.to_dict())
            load_list = list(server_state_list)

            # ram usage
            ram_usage = 0.0
            if "RAM Memory Usage (MB)" in det_row.index:
                try:
                    ram_usage = float(det_row["RAM Memory Usage (MB)"])
                except Exception:
                    ram_usage = 0.0

            # cpu usage
            cpu_usage_val = None
            if "CPU Usage Per Core" in det_row.index:
                cpu_usage_val = det_row["CPU Usage Per Core"]

            # duration
            duration = 0.0
            if "Execution Time (seconds)" in det_row.index:
                try:
                    duration = float(det_row["Execution Time (seconds)"])
                except Exception:
                    duration = 0.0
            elif "Duration" in det_row.index:
                try:
                    duration = float(det_row["Duration"])
                except Exception:
                    duration = 0.0

            # arrival_time
            arrival_time = 0.0
            for c in ("Iteration", "Image No", "Timestamp"):
                if c in det_row.index and not pd.isna(det_row[c]):
                    try:
                        arrival_time = float(det_row[c])
                        break
                    except Exception:
                        pass

            # gpu usage
            gpu_usage_val = None
            if "GPU Usage (%)" in det_row.index:
                try:
                    gpu_usage_val = int(det_row["GPU Usage (%)"])
                except Exception:
                    try:
                        gpu_usage_val = int(float(det_row["GPU Usage (%)"]))
                    except Exception:
                        gpu_usage_val = None

            # cpu/gpu models & clocks
            cpu_model = det_row["CPU Model"] if "CPU Model" in det_row.index else None
            gpu_model = det_row["GPU Model"] if "GPU Model" in det_row.index else None

            cpu_clock_val = None
            if "CPU Clock Speed (MHz)" in det_row.index:
                try:
                    cpu_clock_val = float(det_row["CPU Clock Speed (MHz)"])
                except Exception:
                    cpu_clock_val = None

            gpu_clock_val = None
            if "GPU Clock Speed (MHz)" in det_row.index:
                try:
                    gpu_clock_val = float(det_row["GPU Clock Speed (MHz)"])
                except Exception:
                    gpu_clock_val = None

            # time_util
            proc_t = 0.0
            if "Processing Time" in det_row.index:
                try:
                    proc_t = float(det_row["Processing Time"])
                except Exception:
                    proc_t = 0.0
            exec_t = duration
            time_util = [proc_t, exec_t]

            # extras
            extras = {}
            skip_keys = {
                "Image Pixel", "Image Pix", "RAM Memory Usage (MB)", "CPU Usage Per Core",
                "GPU Usage (%)", "CPU Model", "GPU Model", "CPU Clock Speed (MHz)",
                "GPU Clock Speed (MHz)", "Processing Time", "Execution Time (seconds)",
                "Duration", "Iteration", "Image No", "Timestamp"
            }
            for k, v in det_row.to_dict().items():
                if k not in skip_keys:
                    extras[k] = v

            req = Request(
                request_id=request_id,
                process_id=process_id,
                message_size=message_size,
                bandwidth=bandwidth,
                load=load_list,
                ram_usage=ram_usage,
                cpu_usage=cpu_usage_val,
                arrival_time=arrival_time,
                duration=duration,
                extras=extras,
                gpu_usage=gpu_usage_val,
                cpu_model=cpu_model,
                gpu_model=gpu_model,
                cpu_clock=cpu_clock_val,
                gpu_clock=gpu_clock_val,
                time_util=time_util,
            )

            self.requests.append(req)

        logger.info("Built %d Request objects (rows 0..%d).", len(self.requests), max(0, self.number_of_requests - 1))

    # -------------------------
    def next_request(self) -> Optional[Request]: #this to enumerate through request list
        if self._cursor >= len(self.requests):
            return None
        r = self.requests[self._cursor]
        self._cursor += 1
        return r

    def reset(self) -> None:
        self._cursor = 0

    def get_state_for_request(self, request: Request) -> Dict[str, Any]: #this to get state from user object
        load_arr = np.array(request.load if request.load else [], dtype=int)
        return {
            "LOAD": load_arr,
            "MESSAGE_SIZE": float(request.message_size),
            "BANDWIDTH": float(request.bandwidth),
        }

    def __len__(self) -> int: 
        return len(self.requests)

# main is kept for debugging 


# # -------------------------
# # __main__ for quick check
# # -------------------------
# if __name__ == "__main__":
#     # allow optional args: <num_requests> <detect_csv> <server_state_csv>
#     argv = sys.argv[1:]
#     num = int(argv[0]) if len(argv) >= 1 else 5
#     detect_path = argv[1] if len(argv) >= 2 else None
#     server_state_path = argv[2] if len(argv) >= 3 else None

#     users = Users(num, detect_csv=detect_path, server_state_csv=server_state_path)
#     n = len(users)
#     print(f"Built {n} requests (requested {num})\n")

#     # print details for each built request
#     for i, req in enumerate(users.requests[:min(10, n)]):
#         print(f"--- Request #{i} ---")
#         d = req.as_dict()
#         for k, v in d.items():
#             print(f"{k}: {v}")
#         state = users.get_state_for_request(req)
#         print("state.LOAD (numpy):", state["LOAD"], "dtype:", state["LOAD"].dtype)
#         print("state.MESSAGE_SIZE:", state["MESSAGE_SIZE"])
#         print("state.BANDWIDTH:", state["BANDWIDTH"])
#         print("time_util:", req.time_util)
#         print()
