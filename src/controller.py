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
        import constants
        self.requests = requests
        self.agent = agent.DQNAgent(
            nS=constants.nS,
            nA=constants.nA,
            alpha=constants.alpha,
            gamma=constants.discount_rate,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=constants.gamma_decay,
            batch_size=constants.batch_size,
            beta=constants.beta,
            median_computation_delay=constants.median_computation_delay,
            learning_rate=constants.learning_rate,
            episodes=constants.no_of_episodes
        )
    
    def run(self):
        """
        Pass each request object directly to the agent for processing.
        """
        for req in self.requests:
            self.agent.process_request(req)
    

