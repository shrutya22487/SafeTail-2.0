from collections import deque
import numpy as np
import pandas as pd
import agent
import random
import time
import math
import servers
import user
import constants
from typing import List

class Controller:
    def __init__(self, requests : List[user.Request], input_dim=10, output_dim=5):

        self.requests = requests
        self.agent = agent.DQNAgent(
            states=constants.nS,
            actions=constants.nA,
            alpha=constants.alpha,
            reward_gamma=constants.discount_rate,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=constants.gamma_decay,
            batch_size=constants.batch_size,
            beta=constants.beta,
            median_computation_delay=constants.median_computation_delay,
            learning_rate=constants.learning_rate,
            task=None,
            epochs=constants.no_of_episodes,
            request=None  # will be set per request
        )    
        
    def run(self):
        """
        Pass each request object directly to the agent for processing.
        """
        for req in self.requests:
            self.agent.process_request(req)
    