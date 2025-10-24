from typing import Dict, Any, Optional
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
        load: Optional[np.ndarray] = None,
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
        time_util: Optional[np.ndarray] = None,
    ):
        self.request_id = int(request_id)
        self.process_id = int(process_id)
        self.message_size = float(message_size)
        self.bandwidth = float(bandwidth)

        self.load = np.asarray(load if load is not None else [], dtype=int)
        self.ram_usage = float(ram_usage)
        self.cpu_usage = cpu_usage
        self.arrival_time = float(arrival_time)
        self.duration = float(duration)

        self.extras = {}
        if extras:
            for k, v in extras.items():
                if isinstance(v, (list, tuple)):
                    self.extras[k] = np.asarray(v)
                else:
                    self.extras[k] = v

        self.gpu_usage = int(gpu_usage) if gpu_usage not in (None, "") else None
        self.cpu_model = cpu_model
        self.gpu_model = gpu_model
        self.cpu_clock = float(cpu_clock) if cpu_clock not in (None, "") else None
        self.gpu_clock = float(gpu_clock) if gpu_clock not in (None, "") else None
        self.time_util = np.asarray(time_util if time_util is not None else [], dtype=float)

    def to_state(self) -> Dict[str, Any]:
        return {
            "LOAD": self.load,
            "MESSAGE_SIZE": self.message_size,
            "BANDWIDTH": self.bandwidth,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
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

    def __repr__(self) -> str:
        return (
            f"<Request id={self.request_id} pid={self.process_id} msg={self.message_size}B "
            f"load_len={self.load.size} ram={self.ram_usage} cpu={self.cpu_usage}>"
        )


class Users:
    DEFAULT_DETECT_CSV = '../data/updated_Detect.csv'
    DEFAULT_SERVER_STATE_CSV = '../data/server_state.csv'

    def __init__(self, number_of_requests: int, detect_csv: Optional[str] = None, server_state_csv: Optional[str] = None):
        self.number_of_requests = int(number_of_requests)
        self.detect_csv = Path(detect_csv or self.DEFAULT_DETECT_CSV)
        self.server_state_csv = Path(server_state_csv or self.DEFAULT_SERVER_STATE_CSV)

        if not self.detect_csv.exists():
            raise FileNotFoundError(f"Detect CSV not found: {self.detect_csv}")
        if not self.server_state_csv.exists():
            raise FileNotFoundError(f"Server state CSV not found: {self.server_state_csv}")

        self.detect_df = pd.read_csv(self.detect_csv)
        self.server_state_df = pd.read_csv(self.server_state_csv)

        available = min(len(self.detect_df), len(self.server_state_df))
        if self.number_of_requests > available:
            logger.warning(
                "Requested %d requests but only %d rows available; capping to %d.",
                self.number_of_requests, available, available,
            )
            self.number_of_requests = available

        self.requests = np.empty(self.number_of_requests, dtype=object)
        self._cursor = 0
        self._build_requests()

    @staticmethod
    def _parse_message_size(raw: Any) -> float:
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return 0.0
        try:
            return float(raw)
        except Exception:
            s = str(raw).strip()
            m = re.search(r"(\d+)\s*[xX]\s*(\d+)", s)
            if m:
                return float(int(m.group(1)) * int(m.group(2)) * 3)
            m2 = re.search(r"(\d+)", s)
            return float(int(m2.group(1))) if m2 else 0.0

    @staticmethod
    def _server_row_to_array(server_row: Dict[str, Any]) -> np.ndarray:
        kv = []
        for k, v in server_row.items():
            if str(k).strip().lower() == "timestamp":
                continue
            m = re.search(r"(\d+)\s*$", str(k))
            idx = int(m.group(1)) if m else float("inf")
            kv.append((idx, k, v))
        kv.sort(key=lambda x: (x[0], x[1]))
        values = []
        for _, _, v in kv:
            try:
                values.append(int(float(v)))
            except Exception:
                values.append(0)
        return np.asarray(values, dtype=int)

    def _build_requests(self) -> None:
        for i in range(self.number_of_requests):
            det_row = self.detect_df.iloc[i]
            ss_row = self.server_state_df.iloc[i]

            message_size = 0.0
            for key in ("Image Pixel", "Image Pix"):
                if key in det_row.index:
                    message_size = self._parse_message_size(det_row[key])
                    break

            load_array = self._server_row_to_array(ss_row.to_dict())
            ram_usage = float(det_row.get("RAM Memory Usage (MB)", 0.0))
            cpu_usage = det_row.get("CPU Usage Per Core", None)
            duration = float(det_row.get("Execution Time (seconds)", det_row.get("Duration", 0.0)))
            arrival_time = float(det_row.get("Iteration", det_row.get("Image No", det_row.get("Timestamp", 0.0))))

            gpu_usage = det_row.get("GPU Usage (%)", None)
            gpu_usage = int(gpu_usage) if gpu_usage not in (None, "") else None
            cpu_model = det_row.get("CPU Model", None)
            gpu_model = det_row.get("GPU Model", None)
            cpu_clock = det_row.get("CPU Clock Speed (MHz)", None)
            gpu_clock = det_row.get("GPU Clock Speed (MHz)", None)

            proc_t = float(det_row.get("Processing Time", 0.0))
            time_util = np.asarray([proc_t, duration], dtype=float)

            extras = {
                k: (np.asarray(v) if isinstance(v, (list, tuple)) else v)
                for k, v in det_row.to_dict().items()
                if k not in {
                    "Image Pixel", "Image Pix", "RAM Memory Usage (MB)", "CPU Usage Per Core",
                    "GPU Usage (%)", "CPU Model", "GPU Model", "CPU Clock Speed (MHz)",
                    "GPU Clock Speed (MHz)", "Processing Time", "Execution Time (seconds)",
                    "Duration", "Iteration", "Image No", "Timestamp"
                }
            }

            self.requests[i] = Request(
                request_id=i,
                process_id=i,
                message_size=message_size,
                bandwidth=float(getattr(constants, "max_bandwidth", 200)),
                load=load_array,
                ram_usage=ram_usage,
                cpu_usage=cpu_usage,
                arrival_time=arrival_time,
                duration=duration,
                extras=extras,
                gpu_usage=gpu_usage,
                cpu_model=cpu_model,
                gpu_model=gpu_model,
                cpu_clock=cpu_clock,
                gpu_clock=gpu_clock,
                time_util=time_util,
            )

        logger.info("Built %d Request objects (rows 0..%d).", len(self.requests), self.number_of_requests - 1)

    def next_request(self) -> Optional[Request]:
        if self._cursor >= len(self.requests):
            return None
        req = self.requests[self._cursor]
        self._cursor += 1
        return req

    def reset(self) -> None:
        self._cursor = 0

    def get_state_for_request(self, request: Request) -> Dict[str, Any]:
        return request.to_state()

    def __len__(self) -> int:
        return len(self.requests)
