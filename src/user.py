from typing import Dict, Any, Optional
import numpy as np


class Request:
    """
    Pickle-safe Request object.

    Dicts = structure / meaning
    NP arrays = computation
    """

    def __init__(
        self,
        request_id: int,
        process_id: int,
        combination: str,

        message_size: np.ndarray,
        bandwidth: np.ndarray,
        load: np.ndarray,
    ):
        # -------- identifiers --------
        self.request_id: int = int(request_id)
        self.process_id: int = int(process_id)
        self.combination: str = combination

        # -------- base arrays --------
        self.message_size = np.asarray(message_size, dtype=float)
        self.bandwidth = np.asarray(bandwidth, dtype=float)
        self.load = np.asarray(load, dtype=int)

        # -------- server dicts (structure) --------
        self.server_dicts: list[Dict[str, Any]] = [
            {}, {}, {}, {}, {}
        ]

        # -------- server NP arrays (computation) --------
        # Each entry will be ONE np.ndarray or None
        self.server_np: list[Optional[np.ndarray]] = [
            None, None, None, None, None
        ]

    # ==================================================
    # STRUCTURE FILL
    # ==================================================
    def fill_server_dict(
        self,
        server_idx: int,
        *,
        ram_usage: np.ndarray,
        cpu_usage: np.ndarray,
        duration: np.ndarray,
        time_util: np.ndarray,
        gpu_usage: Optional[np.ndarray] = None,
        cpu_model: Optional[list] = None,
        gpu_model: Optional[list] = None,
        cpu_clock: Optional[np.ndarray] = None,
        gpu_clock: Optional[np.ndarray] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        d = self.server_dicts[server_idx - 1]

        d["ram_usage"] = ram_usage
        d["cpu_usage"] = cpu_usage
        d["duration"] = duration
        d["time_util"] = time_util
        d["gpu_usage"] = gpu_usage
        d["cpu_model"] = cpu_model
        d["gpu_model"] = gpu_model
        d["cpu_clock"] = cpu_clock
        d["gpu_clock"] = gpu_clock
        d["extras"] = extras

    # ==================================================
    # COMPUTATION FILL (NUMBERS ONLY)
    # ==================================================
    def fill_server_np(
        self,
        server_idx: int,
        *,
        ram_usage: np.ndarray,
        cpu_usage: np.ndarray,
        duration: np.ndarray,
        time_util: np.ndarray,
        gpu_usage: Optional[np.ndarray] = None,
        cpu_clock: Optional[np.ndarray] = None,
        gpu_clock: Optional[np.ndarray] = None,
    ) -> None:
        arrays = [
            ram_usage,
            cpu_usage,
            duration,
            time_util,
        ]

        if gpu_usage is not None:
            arrays.append(gpu_usage)
        if cpu_clock is not None:
            arrays.append(cpu_clock)
        if gpu_clock is not None:
            arrays.append(gpu_clock)

        dumped = np.concatenate([arr.ravel() for arr in arrays])
        self.server_np[server_idx - 1] = dumped

    def __repr__(self) -> str:
        return f"<Request id={self.request_id} combo={self.combination}>"
