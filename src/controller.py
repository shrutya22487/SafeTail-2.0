from collections import deque
import numpy as np
import pandas as pd
import agent
import random
import time
import math
import servers
import user
from typing import List

class Controller:
    def __init__(self, requests : List[user.Request], input_dim=10, output_dim=5):
        self.requests = requests
        # add encoder here
        # add any other initialization you need
        self.agent = agent.DQNAgent()
        self.encoder = agent.Encoder(input_dim, output_dim)
        self.server_state = pd.read_csv('data/server_state.csv').iloc[:, 1:6]
    
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
                req.server_state = self.server_state.sample(n=1).iloc[0].tolist()
        else:
            raise ValueError("No entries found with 'Current Script' as 'detect.py'")
    
    def run(self):
        self._merge_server_state_with_request()
        #connect with contoller here and pass the requests to it
        
        
    