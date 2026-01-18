from typing import Dict, Any, Optional

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
        combination: str,

        message_size: np.ndarray,
        bandwidth: np.ndarray,
        load: np.ndarray,
        ram_usage: np.ndarray,
        cpu_usage: np.ndarray,

        arrival_time: float,
        duration: np.ndarray,

        time_util: np.ndarray,

        gpu_usage: Optional[np.ndarray] = None,
        cpu_model: Optional[list] = None,
        gpu_model: Optional[list] = None,
        cpu_clock: Optional[np.ndarray] = None,
        gpu_clock: Optional[np.ndarray] = None,

        extras: Optional[Dict[str, Any]] = None,
    ):
        self.request_id = int(request_id)
        self.process_id = int(process_id)
        self.combination = combination

        # -------- server-wise numeric arrays --------
        self.message_size = np.asarray(message_size, dtype=float)
        self.bandwidth = np.asarray(bandwidth, dtype=float)
        self.load = np.asarray(load, dtype=int)
        self.ram_usage = np.asarray(ram_usage, dtype=float)
        self.cpu_usage = np.asarray(cpu_usage, dtype=float)
        self.duration = np.asarray(duration, dtype=float)
        self.time_util = np.asarray(time_util, dtype=float)

        self.gpu_usage = (
            np.asarray(gpu_usage, dtype=int)
            if gpu_usage is not None else None
        )

        self.cpu_clock = (
            np.asarray(cpu_clock, dtype=float)
            if cpu_clock is not None else None
        )

        self.gpu_clock = (
            np.asarray(gpu_clock, dtype=float)
            if gpu_clock is not None else None
        )

        # -------- metadata --------
        self.cpu_model = cpu_model
        self.gpu_model = gpu_model
        self.arrival_time = float(arrival_time)

        self.extras = extras or {}

    def to_state(self) -> Dict[str, Any]:
        return {
            "MESSAGE_SIZE": self.message_size,
            "BANDWIDTH": self.bandwidth,
            "LOAD": self.load,
            "RAM": self.ram_usage,
            "CPU": self.cpu_usage,
        }

    def __repr__(self):
        return (
            f"<Request id={self.request_id} "
            f"combo={self.combination} "
            f"servers={len(self.duration)}>"
        )
