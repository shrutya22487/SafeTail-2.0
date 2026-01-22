from typing import Dict, Any, Optional
import numpy as np


class Request:
    """
    Request represents ONE logical request evaluated across MULTIPLE servers.

    DESIGN PHILOSOPHY
    -----------------
    This object intentionally separates DATA INTO TWO FORMS:

    1. server*_dict  → STRUCTURE / MEANING
       - Named fields (ram_usage, cpu_usage, etc.)
       - Easy to debug, log, reason about
       - Flexible: fields can change without breaking code

    2. server*_NP    → COMPUTATION
       - One flat NumPy array of numbers only
       - No headers, no structure, no metadata
       - Optimized for schedulers, math, ML, cost functions

    This separation avoids a common mistake:
    mixing "meaning" and "math" into the same container.

    
    """

    # ==================================================
    # CONSTRUCTOR
    # ==================================================
    def __init__(
        self,
        request_id: int,
        process_id: int,
        combination: str,

        # Base per-request arrays (already fixed by you)
        message_size: np.ndarray,
        bandwidth: np.ndarray,
        load: np.ndarray,
    ):
        # -------- identifiers --------
        self.request_id: int = int(request_id)
        self.process_id: int = int(process_id)
        self.combination: str = combination

        # -------- base arrays (server-aligned) --------
        self.message_size: np.ndarray = np.asarray(message_size, dtype=float)
        self.bandwidth: np.ndarray = np.asarray(bandwidth, dtype=float)
        self.load: np.ndarray = np.asarray(load, dtype=int)

        # ==================================================
        # SERVER STRUCTURE (DICT FORM)
        # ==================================================
        # These store NAMED fields.
        # Used for debugging, logging, inspection, validation.
        self.server1_dict: Dict[str, Any] = {}
        self.server2_dict: Dict[str, Any] = {}
        self.server3_dict: Dict[str, Any] = {}
        self.server4_dict: Dict[str, Any] = {}
        self.server5_dict: Dict[str, Any] = {}

        # Internal indexed access (avoids if/else chains)
        self._server_dicts = [
            self.server1_dict,
            self.server2_dict,
            self.server3_dict,
            self.server4_dict,
            self.server5_dict,
        ]

        # ==================================================
        # SERVER COMPUTATION (NUMPY FORM)
        # ==================================================
        # These store ONLY numbers.
        # No headers, no structure.
        # Used ONLY for math / scheduling / ML.
        self.server1_NP: Optional[np.ndarray] = None
        self.server2_NP: Optional[np.ndarray] = None
        self.server3_NP: Optional[np.ndarray] = None
        self.server4_NP: Optional[np.ndarray] = None
        self.server5_NP: Optional[np.ndarray] = None

        # Setter indirection to keep code simple and explicit
        self._server_np_setters = [
            lambda v: setattr(self, "server1_NP", v),
            lambda v: setattr(self, "server2_NP", v),
            lambda v: setattr(self, "server3_NP", v),
            lambda v: setattr(self, "server4_NP", v),
            lambda v: setattr(self, "server5_NP", v),
        ]

    # ==================================================
    # FILL SERVER DICT (STRUCTURE)
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
        """
        Fill STRUCTURED server data.

        This function:
        - Preserves semantic meaning
        - Does NOT flatten
        - Does NOT do computation
        """
        d = self._server_dicts[server_idx - 1]

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
    # FILL SERVER NP (COMPUTATION)
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
        """
        Fill COMPUTATION array for a server.

        This function:
        - Flattens all arrays
        - Concatenates ALL numbers
        - Produces ONE 1D np.ndarray
        - Loses semantic boundaries by design
        """
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

        flat_numeric_array = np.concatenate(
            [arr.ravel() for arr in arrays]
        )

        self._server_np_setters[server_idx - 1](flat_numeric_array)

    # ==================================================
    # DEBUG / LOGGING
    # ==================================================
    def __repr__(self) -> str:
        return (
            f"<Request id={self.request_id} "
            f"combo={self.combination}>"
        )
