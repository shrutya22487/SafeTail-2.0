import random
import numpy as np
import pickle
import pandas as pd
import user
import computation_delay_regressor

class Servers:
    """
    servers.py
    -----------
    Provides the Servers class for simulating delays in a networked environment.

    Sample Usage:
    -------------
    from servers import Servers

    # Example state dictionary
    state = {
        'LOAD': [1, 2, 3],
        'MESSAGE_SIZE': 1024,
        'BANDWIDTH': 10000
    }
    servers = Servers(user_index=1)
    total_delay = servers.get_delays(state, server_index=1)
    print(f"Total delay for node 1: {total_delay}")
    """
    def __init__(self):
        """
        Initialize the Server by loading required models and data.
        Servers class for simulating network/server-side delays.
        Loads propagation delays and server data for computation, transmission, and propagation delay estimation.
        Initialize the Servers by loading required models and data.

        """
        self.propagation_delays = pickle.load(open('./data/propagation_delays.pkl', 'rb')) 
        self.server_data = pd.read_csv('./data/server_data.csv')
    
    def _get_propogation_delay(self, server_index):
        """
        Get a random propagation delay for a given node.
        Returns:
            float: Randomly selected propagation delay.
        Get a random propagation delay for a given node.
        Args:
            message_size (float): Size of the message in bytes.
            upload_bandwidth (float): Uplink bandwidth in Mbps.
            download_bandwidth (float): Downlink bandwidth in Mbps.
        Returns:
            float: Randomly selected propagation delay.
        """
        return random.choice(self.propagation_delays[server_index-1])
    
    def _get_tramission_delay(self, message_size, upload_bandwidth, download_bandwidth):
        message_size = 8 * message_size
        uplink = upload_bandwidth * 1000
        downlink = download_bandwidth * 1000
        return (message_size / uplink) + (message_size / downlink)
    
    def _get_computation_delay(self, process_id):
        return computation_delay_regressor.predict_rows(row_num=process_id)
    
    def get_delays(self, server_index, request: user.Request):
        """
        Get the total delay (propagation, transmission, computation) for a node.
        Args:
            server_index (int): Node index (1-based).
        Returns:
            float: Total delay for the node.
        """
        propagation_delay_for_node = self._get_propogation_delay(request.load[server_index])
        tramission_delay_for_node = self._get_tramission_delay(request.message_size , request.bandwidth / request.load[server_index] , request.bandwidth / request.load[server_index])
        computation_delay_for_node = self._get_computation_delay(request.process_id)
        return propagation_delay_for_node + tramission_delay_for_node + computation_delay_for_node


    
