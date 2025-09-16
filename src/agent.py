import random
import numpy as np
import time
from collections import deque
import tensorflow.keras as keras

from keras.layers import Dense
from keras.optimizers import Adam
from encoder import Encoder
from controller import Controller


def get_subsets(fullset):
    """Helper: return all non-empty subsets of a set"""
    listrep = list(fullset)
    subsets = []
    for i in range(2**len(listrep)):
        subset = []
        for k in range(len(listrep)):
            if i & 1 << k:
                subset.append(listrep[k])
        subsets.append(subset)
    return subsets[1:]


def get_state_input(state):
    """
    Flatten the state dict into a list of values.
    Used before encoding.
    """
    output = []
    for i in state['LOAD']:
        output.append(i)
    if "MESSAGE_SIZE" in state.keys():
        output.append(state['MESSAGE_SIZE'])
    if "RESOLUTION" in state.keys():
        output.append(state['RESOLUTION'])
    if "BANDWIDTH" in state.keys():
        output.append(state['BANDWIDTH'])
    if "PROPOGATION" in state.keys():
        for i in state['PROPOGATION']:
            output.append(i)
    return output


class DQNAgent:
    def __init__(self, states, actions, alpha, reward_gamma, epsilon,
                 epsilon_min, epsilon_decay, batch_size, beta,
                 median_computation_delay, learning_rate, task, epochs,
                 encoder_output_dim=32):
        self.nS = states
        self.nA = actions
        self.memory = deque([], maxlen=2500)
        self.alpha = alpha
        self.reward_gamma = reward_gamma
        self.beta = beta
        self.median_computation_delay = median_computation_delay
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate 
        self.task = task
        self.epochs = epochs

        # Encoder: projects variable-length state vectors to fixed size
        self.encoder = Encoder(input_dim=self.nS, output_dim=encoder_output_dim)

        # RL model: same as before
        self.model = self.build_model(encoder_output_dim)

        # Stats
        self.loss = []
        self.val_loss = []
        self.exploit_or_explore = []
        self.epsilon_curve = []
        self.episode_access_rate = []
        self.latencies = []
        self.deviations = []
        self.rewards = []
        self.action = []
        self.load_arr = []

        # Controller for reward calculation
        self.controller = Controller()

    def build_model(self, encoded_dim):
        model = keras.Sequential() 
        model.add(keras.layers.Dense(encoded_dim*2, input_dim=encoded_dim, activation='sigmoid')) 
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(encoded_dim*4, activation='sigmoid')) 
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(self.nA, activation='softmax')) 

        model.compile(loss='categorical_crossentropy', 
                      optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate)) 
        return model
    
    def store(self, state, action, reward , next_state):
        # Encode before storing
        encoded_state = self.encoder(np.array(get_state_input(state)).reshape(1, -1))
        encoded_next_state = self.encoder(np.array(get_state_input(next_state)).reshape(1, -1))
        self.memory.append((encoded_state, action, reward, encoded_next_state))
    
    def get_action(self,state):
        no_of_edges = self.beta
        subsets = get_subsets(set([x for x in range(self.beta)]))
        state_flattened = np.array(get_state_input(state)).reshape(1, self.nS)
        encoded_state = self.encoder(state_flattened).numpy()

        if np.random.rand() <= self.epsilon:
            self.exploit_or_explore.append("explore")
            action = np.random.randint(0, self.nA)
        else:
            self.exploit_or_explore.append("exploit")
            start_time = time.time()
            action_vals = self.model.predict(encoded_state , verbose=0)
            end_time = time.time()
            print("Time taken to Predict: {:.4f} seconds".format(end_time - start_time))  
            action = np.argmax(action_vals[0])

        return_arr = subsets[action]
   
        self.episode_access_rate.append(float(len(return_arr))/no_of_edges)
        self.action.append(subsets[action])
        return return_arr , action 

    def experience_replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = zip(*minibatch)
        states = np.vstack(states)
        next_states = np.vstack(next_states)

        current_q = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)

        targets = current_q.copy()
        targets[np.arange(batch_size), actions] = (rewards) + self.reward_gamma * np.amax(next_q_values, axis=1) 

        hist = self.model.fit(states, targets, epochs=self.epochs, verbose=0, validation_split=0.2)

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        self.loss.append(hist.history['loss'][0])
        self.val_loss.append(hist.history['val_loss'][0])

    def reward(self, action, state):
        """
        Uses Controller to compute min latency among chosen servers.
        """
        MEDIAN_LATENCY = self.median_computation_delay

        servers_to_be_queried = action
        obs_latency = self.controller._get_min_delay(state, servers_to_be_queried)

        # same reward structure as before
        if abs(obs_latency - MEDIAN_LATENCY) < 1000:
            lamda = (obs_latency - MEDIAN_LATENCY)
            reward = 0
            delta = 0
            gamma = len(action)-1 

            if lamda < 0:
                delta = (self.alpha * np.exp(-1 * lamda))
            else:
                delta = (self.alpha * np.exp(lamda))
            
            if lamda == 0:
                reward = 0
            elif lamda > 0 and self.beta - gamma == 1:
                reward = 0
            elif lamda > 0 and self.beta - gamma > 1 :
                reward = (-1 * np.exp(self.beta - gamma - 1) * delta)
            elif lamda < 0:
                reward = (-1 * np.exp(gamma) * delta)

            self.latencies.append(obs_latency)
            self.deviations.append(abs(obs_latency - MEDIAN_LATENCY))
            self.rewards.append(reward)    

            if len(self.memory) > self.batch_size:
                self.experience_replay(self.batch_size)

            return reward
