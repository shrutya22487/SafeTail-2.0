import random
import numpy as np
import time
from collections import deque
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.layers import Dense
from keras.optimizers import Adam
from encoder import Encoder
import servers
import user
import traceback
from tensorflow.keras import layers

def get_subsets(fullset):
    """Helper: return all non-empty subsets of a set"""
    
    # Converting set to list for indexing
    listrep = list(fullset)
    subsets = []
    
    # There are 2^n subsets for a set of size n
    # Looping over all the subsets
    for i in range(2**len(listrep)):
        # Building each subset using bitmasking
        subset = []
        for k in range(len(listrep)):
            if i & 1 << k:
                subset.append(listrep[k])
        subsets.append(np.array(subset))  # convert each subset to np.array
    return np.array(subsets[1:], dtype=object)  # return numpy array of subsets


def request_to_state_array(request_obj):
    """
    Converts a Request object into a flattened numeric NumPy array.
    Handles nested arrays, lists, dicts, and None values safely.
    Replaces None or invalid numeric values with -1.0.
    """
    import numpy as np
    import traceback

    flat_values = []

    for attr, value in vars(request_obj).items():
        try:
            # None → -1.0
            if value is None:
                flat_values.append(-1.0)

            # Scalars (int, float, numpy scalar)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                flat_values.append(float(value))

            # Lists, tuples, or numpy arrays
            elif isinstance(value, (list, tuple, np.ndarray)):
                try:
                    arr = np.asarray(value, dtype=float).flatten()
                    flat_values.extend(arr.tolist())
                except Exception as e:
                    print(f"[AGENT]    [WARN] Could not convert array-like attribute '{attr}' → {type(value)}: {e}")
                    flat_values.append(-1.0)
                    continue

            # Dictionaries (flatten recursively)
            elif isinstance(value, dict):
                for k, v in value.items():
                    try:
                        if v is None:
                            flat_values.append(-1.0)
                        elif isinstance(v, (int, float, np.integer, np.floating)):
                            flat_values.append(float(v))
                        elif isinstance(v, (list, tuple, np.ndarray)):
                            arr = np.asarray(v, dtype=float).flatten()
                            flat_values.extend(arr.tolist())
                        else:
                            flat_values.append(-1.0)
                    except Exception as e:
                        print(f"[AGENT]    [WARN] Could not process dict key '{k}' in '{attr}': {e}")
                        flat_values.append(-1.0)
                        continue

            # Unsupported types (like strings or objects) → -1.0
            else:
                flat_values.append(-1.0)

        except Exception as e:
            print(f"[AGENT]    [ERROR] Failed to process attribute '{attr}' ({type(value)}): {e}")
            traceback.print_exc()
            flat_values.append(-1.0)
            continue

    # Final safety conversion
    try:
        return np.array(flat_values, dtype=float)
    except Exception as e:
        print(f"[AGENT]    [FATAL] Could not create NumPy array from collected values: {e}")
        traceback.print_exc()
        return np.full(1, -1.0, dtype=float)  # fallback to safe dummy vector



class DQNAgent:
    def __init__(self, states, actions, alpha, reward_gamma, epsilon,
                 epsilon_min, epsilon_decay, batch_size, beta,
                 median_computation_delay, learning_rate, task, epochs, request : user.Request, server_list: list[servers.Server]
                ,encoder_output_dim=32):
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
        self.encoder = Encoder(hidden_dim=128, output_dim=encoder_output_dim)

        # RL model: same as before
        self.model = self.build_model()

        # Stats (initialized as empty numpy arrays)
        self.loss = np.array([])
        self.val_loss = np.array([])
        self.exploit_or_explore = np.array([], dtype='<U8')  # strings like "explore"/"exploit"
        self.epsilon_curve = np.array([])
        self.episode_access_rate = np.array([])
        self.latencies = np.array([])
        self.deviations = np.array([])
        self.rewards = np.array([])
        self.action = np.array([], dtype=object)
        self.load_arr = np.array([])

        self.request = request
        self.server_list = server_list
        
        self.prediction_times = []
        
        
    def timed_predict(self, state_tensor):
        """Runs model inference and measures time."""
        start_time = time.time()
        q_values = self.model(state_tensor, training=False)
        end_time = time.time()

        elapsed = end_time - start_time
        self.prediction_times.append(elapsed)

        print(f"[AGENT] ⏱ Prediction time: {elapsed:.6f}s")
        return q_values


    def build_model(self):
        model = keras.Sequential() 
        
        # TODO:
        # first of all should the encoder be a part of the model?
        # Our encoder is currently not trained end-to-end with the DQN model.
        
        # Input size = nS, Hidden layer size = 2×nS, should this be changed to the dimensions of the encoder output?
        model.add(keras.layers.Dense(self.nS*2, input_dim=self.nS, activation='sigmoid')) 
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(self.nS*4, activation='sigmoid')) 
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(self.nA, activation='softmax')) 

        model.compile(loss='categorical_crossentropy', 
                      optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate)) 
        return model


    # TODO: 
    # Instead of state, i am passing the request object itself.
    # Earlier the next state was being passed, how do we handle that here?
    # We dont have a next state request object in the current code.
    def store(self, state_request, action, reward, next_state_request):
        """
        Store experience (state, action, reward, next_state) into replay memory.
        Automatically converts Request objects into encoded vectors.
        """
        
        # Convert request objects → flattened numpy arrays
        state_vec = request_to_state_array(state_request).reshape(1, -1, 1)
        next_state_vec = request_to_state_array(next_state_request).reshape(1, -1, 1)

        # Encode using the same encoder as before
        encoded_state = self.encoder(state_vec, training=False).numpy()
        encoded_next_state = self.encoder(next_state_vec, training=False).numpy()

        # Append to replay memory
        self.memory.append((encoded_state, action, reward, encoded_next_state))
    
    
    def get_action(self, request):
        """
        Selects an action (subset of servers) given the current Request.
        Uses epsilon-greedy strategy.
        """

        # 1️⃣ Convert Request object → numeric state vector
        state_flattened = request_to_state_array(request).reshape(1, -1)
        state_flattened_tf = tf.convert_to_tensor(state_flattened.reshape(1, -1, 1), dtype=tf.float32)

        # 2️⃣ Encode using your encoder (no .numpy())
        encoded_state = self.encoder(state_flattened_tf, training=False)
        
        # print(state_flattened)
        print(f"[AGENT] Encoded state shape: {encoded_state.shape}")
        # print(encoded_state)

        # 3️⃣ Build all possible non-empty subsets of servers (0-based indexing)
        subsets = get_subsets(set(range(self.beta)))

        # 4️⃣ Epsilon-greedy exploration
        if np.random.rand() <= self.epsilon:
            self.exploit_or_explore = np.append(self.exploit_or_explore, "explore")
            action = np.random.randint(0, self.nA)

        else:
            self.exploit_or_explore = np.append(self.exploit_or_explore, "exploit")
            
            action_vals = self.timed_predict(encoded_state)

            # Convert to NumPy for argmax
            action = np.argmax(action_vals.numpy()[0])

        # 5️⃣ Get subset of servers for this action
        return_arr = subsets[action]

        # 6️⃣ Log and return
        self.episode_access_rate = np.append(self.episode_access_rate, len(return_arr) / self.beta)
        self.action = np.append(self.action, [return_arr])

        return return_arr, action


    #TODO:
    # Need to adapt this method to work with the encoded states.
    # Since we are using request objects, 
    # How to modify the state and next_state parameters here ?
    def experience_replay(self, batch_size):
        minibatch = random.sample( self.memory, batch_size ) #Randomly sample from memory
        states, actions, rewards, next_states = map(np.array, zip(*minibatch))
        states = np.array(states)
        current_q = self.model.predict(states, verbose =0)

        next_q_values = self.model.predict(next_states,verbose =0)

        targets = current_q.copy()
        targets[np.arange(batch_size), actions] = (rewards) + self.reward_gamma * np.amax(next_q_values, axis=1) 


        hist = self.model.fit(states, targets, epochs=self.epochs, verbose=0 , validation_split=0.2)

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        loss_sum = hist.history['loss'][0]
        val_loss_sum = hist.history['val_loss'][0]
        

        self.loss.append(loss_sum)
        self.val_loss.append(val_loss_sum)
    
    
    
    #TODO: 
    # @Shrutya :  verify usage of .compute_request_time() instead of the .get_delay() method.
    def get_min_delay(self, request, servers_to_be_queried):
        """
        Query the servers and get the minimum delay among them.
        """
        min_delay = float('inf')
        
        for server in servers_to_be_queried:
            min_delay = min(min_delay, self.server_list[server].compute_request_time(request))
        return min_delay


    #TODO: 
    # @Shrutya :  verify usage of .compute_request_time() instead of the .get_delay() method.
    # 
    # 
    # @Medha :  Review this reward function again. 
    # Need to replace with new reward function.
    def reward(self, action, request):
        """
        Used to compute min latency among chosen servers.
        """
        MEDIAN_LATENCY = self.median_computation_delay

        servers_to_be_queried = action
        obs_latency = self.get_min_delay(request, servers_to_be_queried)

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

            self.latencies = np.append(self.latencies, obs_latency)
            self.deviations = np.append(self.deviations, abs(obs_latency - MEDIAN_LATENCY))
            self.rewards = np.append(self.rewards, reward)    

            if len(self.memory) > self.batch_size:
                print("[AGENT] Performing experience replay...")
                self.experience_replay(self.batch_size)

            return reward