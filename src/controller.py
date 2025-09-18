from collections import deque
import numpy as np
import random
import time
import math
import servers
import user
from typing import List
import agent
import pandas as pd

# TODO (@BingoBoy479)(@theshamiksinha) Resolve comments here
class Controller:
    def __init__(self, requests : List[user.Request]):
        self.requests = requests
        # add encoder here
        # add any other initialization you need
        self.agent = agent.DQNAgent()
    
    # TODO (@BingoBoy479): Add first 5 columns of the server_state.csv to the state as well
    def _merge_server_state_with_request(self):
        """
        Reads server_data.csv and assigns a random process_id to each request.
        The process_id is a random row index where Current Script is 'detect.py'.
        """
        df = pd.read_csv('data/server_data.csv')
        detect_indices = df.index[df['Current Script'] == 'detect.py'].tolist()

        if detect_indices:
            for req in self.requests:
                req.process_id = random.choice(detect_indices)
                req.cpu_usage = df[req.process_id]['CPU Usage']
                req.ram_usage = df[req.process_id]['RAM Usage']
        else:
            raise ValueError("No entries found with 'Current Script' as 'detect.py'")
    
    def run(self):
        self._merge_server_state_with_request()
        # add encoder here to and connect with agent here and pass the requests to it
        
        
    