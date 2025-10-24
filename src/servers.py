import random
import numpy as np
import pickle
import pandas as pd
import time
import user
import computation_delay_regressor

TIME_SCALE = 10.0  # factor to speed up time in simulation
class Server:
    def __init__(self, server_index):
        self.server_index = server_index
        self.propagation_delays = pickle.load(open('../data/propagation_delays.pkl', 'rb')) 
        self.server_data = pd.read_csv('../data/server_data.csv')
        self.num_requests = 0
        self.requests = []
        # internal tracking of visible active requests with times:
        # list of dicts: {'request': req, 'start_time': t, 'proc_time': p, 'finish_time': t+p}
        self.active_requests = []

    def check_server_availability(self, current_time: float = None):
        if current_time is None:
            current_time = time.time()

        self.update_active_requests(current_time=current_time)
        return self.num_requests < 4

    def _get_propogation_delay(self):
        return random.choice(self.propagation_delays[self.server_index - 1])
    
    def _get_tramission_delay(self, message_size, upload_bandwidth, download_bandwidth):
        message_size = 8 * message_size
        uplink = upload_bandwidth * 1000
        downlink = download_bandwidth * 1000
        return (message_size / uplink) + (message_size / downlink)
    
    def _get_computation_delay(self, process_id):
        return computation_delay_regressor.predict_rows(row_num=process_id)

    def get_delays(self, request: user.Request):
        propagation_delay_for_node = self._get_propogation_delay()
        tramission_delay_for_node = self._get_tramission_delay(
            request.message_size,
            request.bandwidth / request.load[self.server_index],
            request.bandwidth / request.load[self.server_index]
        )
        computation_delay_for_node = self._get_computation_delay(request.process_id)
        return propagation_delay_for_node + tramission_delay_for_node + computation_delay_for_node

    def _scale_proc_time(self, proc_time: float) -> float:
        # divide original seconds by time_scale to get real-time seconds
        return float(proc_time) / TIME_SCALE
    
    def update_active_requests(self, current_time: float = None):
        """
        Remove finished visible requests whose finish_time <= current_time.
        Decrements self.num_requests for each finished visible request.
        Return number of freed slots.
        """
        if current_time is None:
            current_time = time.time()
        freed = 0
        remaining = []
        for ar in self.active_requests:
            if ar['finish_time'] <= current_time:
                # this visible request finished
                freed += 1
                # try to remove from self.requests if present
                try:
                    self.requests.remove(ar['request'])
                except ValueError:
                    pass
                # decrement authoritative count, but never below 0
                self.num_requests = max(0, self.num_requests - 1)
            else:
                remaining.append(ar)
        self.active_requests = remaining
        return freed    

    def schedule_request(self, request: user.Request, current_time: float = None, do_sleep: bool = False):
        """
        Schedule a visible request in *real time*, using scaled delays.
        If do_sleep==True this call will actually sleep for the scaled proc_time
        (so it's suitable for real-time integration tests/demos).
        Returns:
          (True, finish_time, real_proc_time) if scheduled
          (False, reason, real_proc_time) if cannot schedule
        """
        if current_time is None:
            current_time = time.time()

        if not self.check_server_availability(current_time=current_time):
            # compute proc_time even if rejected so controller can log estimate
            orig_proc = self.get_delays(request)
            real_proc = self._scale_proc_time(orig_proc)
            return False, "server full", real_proc

        orig_proc = self.get_delays(request)
        real_proc = self._scale_proc_time(orig_proc)

        start_time = current_time
        finish_time = start_time + real_proc

        # record visible request
        self.requests.append(request)
        self.active_requests.append({
            'request': request,
            'start_time': start_time,
            'proc_time': real_proc,
            'orig_proc_time': orig_proc,
            'finish_time': finish_time
        })
        self.num_requests += 1

        # Optionally block the calling thread for the duration of the (scaled) processing.
        if do_sleep and real_proc > 0:
            time.sleep(real_proc)
            # after sleep, mark finished immediately (equivalent to update_active_requests)
            # but we still call update_active_requests to keep logic consistent
            self.update_active_requests(current_time=time.time())

        return True, finish_time, real_proc

    def time_until_next_free(self, current_time: float = None):
        """
        Return:
          - 0 if currently there is at least one free slot (num_requests < 4)
          - float > 0: seconds until the earliest visible active request completes
          - None: if occupancy is due to hidden requests (can't predict)
        """
        if current_time is None:
            current_time = time.time()

        self.update_active_requests(current_time=current_time)

        if self.num_requests < 4:
            return 0.0

        visible_active = len(self.active_requests)
        hidden_count = max(0, self.num_requests - visible_active)
        if hidden_count > 0:
            # we can't predict when hidden ones finish
            return None

        # earliest finish among visible actives
        earliest = min(ar['finish_time'] for ar in self.active_requests)
        return max(0.0, earliest - current_time)
