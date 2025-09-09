from collections import deque
import tensorflow.keras as keras
import numpy as np
import random
import time
import math


def get_subsets(fullset):
    """
    Generate all non-empty subsets of a given set.
    Args:
        fullset (set): The set to generate subsets from.
    Returns:
        list: A list of all non-empty subsets.
    """
    listrep = list(fullset)
    subsets = []
    for i in range(2**len(listrep)):
        subset = []
        for k in range(len(listrep)):
            if i & 1<<k:
                subset.append(listrep[k])
        subsets.append(subset)
    return subsets[1:]

def get_state_input(state):
    """
    Convert a state dictionary into a flat list suitable for model input.
    Args:
        state (dict): The state containing various keys.
    Returns:
        list: Flattened state values.
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

class Controller:
    """
    Controller class for managing RL agent's actions, memory, and model.
    Handles experience replay, action selection, and reward calculation.
    """
    def __init__(self, states, actions, alpha, reward_gamma, epsilon,epsilon_min, epsilon_decay , batch_size , beta , median_computation_delay , learning_rate, task,epochs):
        """
        Initialize the Controller with hyperparameters and build the model.
        Args:
            states (int): Number of state features.
            actions (int): Number of possible actions.
            alpha (float): Alpha parameter for reward calculation.
            reward_gamma (float): Discount factor for rewards.
            epsilon (float): Initial epsilon for exploration.
            epsilon_min (float): Minimum epsilon value.
            epsilon_decay (float): Epsilon decay rate.
            batch_size (int): Batch size for experience replay.
            beta (int): Number of edges/nodes.
            median_computation_delay (float): Median computation delay for reward.
            learning_rate (float): Learning rate for optimizer.
            task (str): Task type (e.g., 'instance', 'noise').
            epochs (int): Number of epochs for model training.
        """
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
        self.model = self.build_model()
        self.epochs = epochs
        self.loss = []
        self.val_loss = []
        self.exploit_or_explore = []
        self.epsilon_curve=[]
        self.episode_access_rate = []
        self.latencies = []
        self.deviations = []
        self.rewards = []
        self.action = []
        self.load_arr = []

    def build_model(self) :
        """
        Build and compile the neural network model for Q-learning.
        Returns:
            keras.Model: Compiled Keras model.
        """
        model = keras.Sequential() 
        model.add(keras.layers.Dense(self.nS*2, input_dim=self.nS, activation='sigmoid')) 
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(self.nS*4, activation='sigmoid')) 
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(self.nA, activation='softmax')) 

        model.compile(loss='categorical_crossentropy', 
                      optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate)) 
        return model
    
    def store(self, state, action, reward , next_state):
        """
        Store a transition in the replay memory.
        Args:
            state (dict): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (dict): Next state after action.
        """
        self.memory.append( ( get_state_input(state), action, reward , get_state_input(next_state)) )
    
    def get_action(self,state):
        """
        Select an action using epsilon-greedy policy.
        Args:
            state (dict): Current state.
        Returns:
            tuple: (Selected subset of actions, action index)
        """
        no_of_edges = self.beta
        subsets = get_subsets(set([x for x in range(self.beta)]))
        valid_subsets = get_subsets(set([x for x in range(no_of_edges)]))
        state_flattened = get_state_input(state)
        state_flattened = np.array(state_flattened).reshape(1,self.nS)
        if np.random.rand() <= self.epsilon:
            self.exploit_or_explore.append("explore")
            action = np.random.randint(0,self.nA)
        else:
            self.exploit_or_explore.append("exploit")
            start_time = time.time()
            action_vals = self.model.predict(state_flattened , verbose =0) #Exploit: Use the NN to predict the correct action from this state
            end_time = time.time()
            print("Time taken to Predict: {:.4f} seconds".format(end_time - start_time))  
            # keras.backend.clear_session() # memory leak fix
            action = np.argmax(action_vals[0])

        return_arr = subsets[action]
   
        self.episode_access_rate.append(float(len(return_arr))/no_of_edges)
        self.action.append(subsets[action])
        return return_arr , action 

    def experience_replay(self, batch_size):
        """
        Perform experience replay to train the model on a random batch from memory.
        Args:
            batch_size (int): Number of samples to train on.
        """
        minibatch = random.sample( self.memory, batch_size ) #Randomly sample from memory
        states, actions, rewards, next_states = map(np.array, zip(*minibatch))
        states = np.array(states)
        current_q = self.model.predict(states, verbose =0)
        # keras.backend.clear_session() # memory leak fix
        start_time = time.time()

        next_q_values = self.model.predict(next_states,verbose =0)
        # keras.backend.clear_session() # memory leak fix
        end_time = time.time()
        print("Time taken to Train: {:.4f} seconds".format(end_time - start_time))


        targets = current_q.copy()
        # print("rewards" , rewards)
        targets[np.arange(batch_size), actions] = (rewards) + self.reward_gamma * np.amax(next_q_values, axis=1) 
        # print("targets" , targets)

        hist = self.model.fit(states, targets, epochs=self.epochs, verbose=0 , validation_split=0.2)

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        loss_sum = hist.history['loss'][0]
        val_loss_sum = hist.history['val_loss'][0]
        

        self.loss.append(loss_sum)
        self.val_loss.append(val_loss_sum)
        end_time = time.time()

        #Reshape for Keras Fit
        #Decay Epsilon
       
    ####################################### Integrate with server.py #########################################
    def reward(self, action , state):
        """
        Calculate the reward for a given action and state.
        Args:
            action (list): List of selected nodes/actions.
            state (dict): Current state.
        Returns:
            float: Calculated reward value.
        """
        # print("State" , state)
        # print("Action" , action)
        MEDIAN_LATENCY = self.median_computation_delay

        obs_latency = 100000000
        # print("Action" , action)
        # print("State" , get_state_input(state))
        for node in action :
            computaion_delay_node = get_computation_delay(state,node,self.task)
            tramission_delay_node = get_tramission_delay(state['MESSAGE_SIZE'] , state['BANDWIDTH'] / state['LOAD'][node] ,state['BANDWIDTH']/ state['LOAD'][node])
            propogation_delay_node = get_propogation_delay(state['LOAD'][node])
            # print("Node" , node , "Computation Delay" , computaion_delay_node , "Tramission Delay" , tramission_delay_node, "Propogation Delay" , propogation_delay_node)
            # propogation_delay_node =  get_propogation_delay()
            node_latency =  computaion_delay_node + tramission_delay_node + propogation_delay_node

            if self.task == "instance":
                node_latency = math.log(1+node_latency)
                node_latency = np.array([node_latency])
            
            if self.task == "noise":
                node_latency = math.log(1+node_latency)
                node_latency = np.array([node_latency])

            
            # node_latency =  computaion_delay_node
            obs_latency = min(obs_latency , node_latency)


        if abs(obs_latency - MEDIAN_LATENCY) < 1000:
            # reward1 = self.alpha*(len(action))
            # reward2 = math.log(1+abs(obs_latency - MEDIAN_LATENCY))
            # reward = -1* (self.alpha*(len(action)) + math.log(1+abs(obs_latency - MEDIAN_LATENCY)) ) 
            lamda = (obs_latency - MEDIAN_LATENCY)
            reward = 0
            delta = 0
            gamma = len(action)-1 
            # print("gamma" , gamma)

            if lamda < 0:
                delta = (self.alpha * np.exp(-1 * lamda))

            else:
                delta = (self.alpha * np.exp(lamda))
            
            if lamda == 0:
                reward = 0
                
            elif lamda > 0 and self.beta - gamma == 1:

                reward = 0

            elif lamda > 0 and self.beta - gamma > 1 :
            
                reward = (-1 * np.exp(self.beta - gamma - 1) * delta)[0]
                

            elif lamda < 0:
                
                reward = (-1 * np.exp(gamma) * delta)[0]

            self.latencies.append(obs_latency)
            self.deviations.append(abs(obs_latency - MEDIAN_LATENCY))
            self.rewards.append(reward)    
            # print("Reward" , reward)
            # print("Latency" , obs_latency)
            # print("Deviation" , abs(obs_latency - MEDIAN_LATENCY))
            # print("---------------------------")
            if len(self.memory) > self.batch_size:
                self.experience_replay(self.batch_size)

            return reward