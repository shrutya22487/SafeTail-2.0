from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
import time as time
import pandas as pd

# ==================================================
# CSV CACHE (module-level, pickle-safe)
# ==================================================
# server_idx (1–5) -> DataFrame indexed by Combination
_CSV_CACHE: Dict[int, pd.DataFrame] = {}


def _get_server_df(server_idx: int) -> pd.DataFrame:
    """
    Load server{idx}.csv ONCE, cache it, and return it.

    - Full CSV is loaded (all columns)
    - Indexed by 'Combination' for O(1) lookup
    - Treated as read-only
    """
    server_idx = server_idx+1
    if server_idx in _CSV_CACHE:
        return _CSV_CACHE[server_idx]

    base_dir = Path(__file__).resolve().parent.parent  # Safetail/
    data_dir = base_dir / "data"
    csv_path = data_dir / f"server{server_idx}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    if "Combination" not in df.columns:
        raise ValueError(f"'Combination' column missing in {csv_path.name}")

    df = df.set_index("Combination")
    _CSV_CACHE[server_idx] = df
    return df


def _to_float_array(val) -> np.ndarray:
    """
    Robustly convert CSV cell values into a float numpy array.
    Accepts lists, numpy arrays, or stringified lists.
    """
    if isinstance(val, (list, tuple, np.ndarray)):
        return np.asarray(val, dtype=float)

    if isinstance(val, str):
        try:
            cleaned = val.strip("[]").split(",")
            return np.asarray([float(x) for x in cleaned if x.strip()], dtype=float)
        except ValueError:
            return np.asarray([], dtype=float)

    return np.asarray([], dtype=float)


class Request:
    """
    Pickle-safe Request object.

    server_dicts:
        Rich, semantic, structured data for inspection & debugging

    server_np:
        Flat numeric projection for computation / scheduling / ML
    """

    def __init__(
        self,
        request_id: int,
        process_id: int,
        combination: str,
        message_size: int,
        bandwidth:int,
        load: np.ndarray,
        deadline: np.array
    ):
        self.request_id = int(request_id)
        self.process_id = int(process_id)
        self.combination = combination
        self.deadline = np.asarray([],dtype = float)
        self.arrival_time = time.time()
        self.message_size = int(message_size)
        self.bandwidth = int(bandwidth)
        self.load = np.asarray(load, dtype=int)
        self.step_reward_list = np.asarray([], dtype=float)

        self.server_dicts: List[Dict[str, Any]] = [{} for _ in range(6)]
        self.server_np: List[Optional[np.ndarray]] = [None] * 5

    # ==================================================
    # STRUCTURE FILL
    # ==================================================
    def fill_server_dict(
        self,
        server_idx: int,
        *,
        ram_usage: float,
        cpu_usage: np.ndarray,
        duration: np.ndarray,
        time_exec: float,
        time_process: float,
        time_indi_script_process: np.ndarray,
        time_indi_script_exec: np.ndarray,
        files_per_script: np.ndarray,
        gpu_usage: int,
        gpu_memory: float,
        cpu_core_usage: np.ndarray,
        cpu_core_used: int,
        total_ram: float,
        total_cpu_cores: int,
        total_gpu_memory: float,
        cpu_model: Optional[str],
        gpu_model: Optional[str],
        cpu_clock: np.ndarray,
        gpu_clock: np.ndarray,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:

        if not 0 <= server_idx <= len(self.server_dicts):
            raise ValueError(f"Invalid server_idx: {server_idx}")

        d = self.server_dicts[server_idx]
        d.update(
            {
                "ram_usage": ram_usage,
                "cpu_usage": cpu_usage,
                "duration": duration,
                "time_exec": time_exec,
                "time_process": time_process,
                "time_indi_script_process": time_indi_script_process,
                "time_indi_script_exec": time_indi_script_exec,
                "files_per_script": files_per_script,
                "gpu_usage": gpu_usage,
                "gpu_memory": gpu_memory,
                "cpu_core_usage": cpu_core_usage,
                "cpu_core_used": cpu_core_used,
                "total_ram": total_ram,
                "total_cpu_cores": total_cpu_cores,
                "total_gpu_memory": total_gpu_memory,
                "cpu_model": cpu_model,
                "gpu_model": gpu_model,
                "cpu_clock": cpu_clock,
                "gpu_clock": gpu_clock,
                "extras": extras,
            }
        )

    # ==================================================
    # COMPUTATION FILL (NUMBERS ONLY)
    # ==================================================
    def fill_server_np(
        self,
        server_idx: int,
        *,
        ram_usage: float,
        gpu_usage: int,
        gpu_memory: float,
        time_exec: float,
        time_process: float,
        cpu_core_used: int,
        total_ram: float,
        total_cpu_cores: int,
        total_gpu_memory: float,
        cpu_usage: np.ndarray,
        time_indi_script_process: np.ndarray,
        time_indi_script_exec: np.ndarray,
        files_per_script: np.ndarray,
        cpu_clock: np.ndarray,
        gpu_clock: np.ndarray,
    ) -> None:
        """
        Numeric projection layout (ORDER IS CONTRACT):

        [ ram_usage,
          gpu_usage,
          gpu_memory,
          time_exec,
          time_process,
          cpu_core_used,
          total_ram,
          total_cpu_cores,
          total_gpu_memory,
          cpu_usage...,
          time_indi_script_process...,
          time_indi_script_exec...,
          files_per_script...,
          cpu_clock...,
          gpu_clock... ]
        """

        if not 0 <= server_idx < len(self.server_np):
            raise ValueError(f"Invalid server_idx: {server_idx}")

        scalar_block = np.array(
            [
                ram_usage,
                gpu_usage,
                gpu_memory,
                time_exec,
                time_process,
                cpu_core_used,
                total_ram,
                total_cpu_cores,
                total_gpu_memory,
            ],
            dtype=float,
        )

        arrays = [
            scalar_block,
            cpu_usage.ravel(),
            time_indi_script_process.ravel(),
            time_indi_script_exec.ravel(),
            files_per_script.ravel(),
            cpu_clock.ravel(),
            gpu_clock.ravel(),
        ]

        self.server_np[server_idx] = np.concatenate(arrays)

    # ==================================================
    # POPULATE FROM CSV
    # ==================================================
    def populate_request_from_csv(self, server_idx: int, combined_str: str) -> None:
        df = _get_server_df(server_idx)
        # print('a')
        if combined_str not in df.index:
            raise ValueError(
                f"No row for combination '{combined_str}' "
                f"in server{server_idx}.csv"
            )

        row = df.loc[combined_str]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        # print('a')
        ram_usage = float(row.get("Peak RAM Usage (MB)", 0.0))
        gpu_usage = int(row.get("Peak GPU Usage (%)", 0))
        gpu_memory = float(row.get("Peak GPU Memory (MB)", 0.0))

        time_exec = float(row.get("Total Execution Time (sec)", 0.0))
        time_process = float(row.get("Total Processing Time (sec)", 0.0))

        cpu_core_used = int(row.get("Number of Cores Used", 0))
        total_ram = float(row.get("Total RAM (GB)", 0.0))
        total_cpu_cores = int(row.get("Total CPU Cores", 0))
        total_gpu_memory = float(row.get("Total GPU Memory (GB)", 0.0))

        cpu_model = row.get("CPU Model")
        gpu_model = row.get("GPU Model")

        cpu_usage = _to_float_array(row.get("Average CPU Usage Per Core"))
        cpu_clock = _to_float_array(row.get("Average CPU Clock (MHz)"))
        gpu_clock = _to_float_array(row.get("Average GPU Clock (MHz)"))

        time_indi_script_process = _to_float_array(
            row.get("Individual Processing Times")
        )
        time_indi_script_exec = _to_float_array(
            row.get("Individual Execution Times")
        )

        files_per_script = _to_float_array(
            row.get("Files Processed Per Script")
        )

        duration = np.asarray([time_exec], dtype=float)
        

        self.fill_server_dict(
            server_idx,
            ram_usage=ram_usage,
            cpu_usage=cpu_usage,
            duration=duration,
            time_exec=time_exec,
            time_process=time_process,
            time_indi_script_process=time_indi_script_process,
            time_indi_script_exec=time_indi_script_exec,
            files_per_script=files_per_script,
            gpu_usage=gpu_usage,
            gpu_memory=gpu_memory,
            cpu_core_usage=cpu_usage,
            cpu_core_used=cpu_core_used,
            total_ram=total_ram,
            total_cpu_cores=total_cpu_cores,
            total_gpu_memory=total_gpu_memory,
            cpu_model=cpu_model,
            gpu_model=gpu_model,
            cpu_clock=cpu_clock,
            gpu_clock=gpu_clock,
            
        )

        self.fill_server_np(
            server_idx,
            ram_usage=ram_usage,
            gpu_usage=gpu_usage,
            gpu_memory=gpu_memory,
            time_exec=time_exec,
            time_process=time_process,
            cpu_core_used=cpu_core_used,
            total_ram=total_ram,
            total_cpu_cores=total_cpu_cores,
            total_gpu_memory=total_gpu_memory,
            cpu_usage=cpu_usage,
            time_indi_script_process=time_indi_script_process,
            time_indi_script_exec=time_indi_script_exec,
            files_per_script=files_per_script,
            cpu_clock=cpu_clock,
            gpu_clock=gpu_clock,
        )

    def __repr__(self) -> str:
        return (f"<Request id={self.request_id} combo={self.combination} request_id:{self.request_id}, "
                f"process_id: {self.process_id}, message_size: {self.message_size},bandwidth: {self.bandwidth},load:{self.load}>")
