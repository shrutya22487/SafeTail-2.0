from collections import deque
import numpy as np
import random
import time
import math
import servers

class Controller:
    def __init__(self) -> None:
        pass
    def _get_min_delay(self, state, servers_to_be_queried):
        """
        Query the servers and get the minimum delay among them.
        """
        min_delay = float('inf')
        
        servers_obj = servers.Servers()

        for server in servers_to_be_queried:
            min_delay = min(min_delay, servers_obj.get_delays(state, server))
        return min_delay