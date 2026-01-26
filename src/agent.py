import random
import numpy as np
import time
from collections import deque
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorflow as tf

import servers
import user
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
            
            elif isinstance(value, dict):
                continue  # skip dictionaries entirely
            
            # Scalars (int, float, numpy scalar)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                flat_values.append(float(value))

            # Lists, tuples, or numpy arrays
            elif isinstance(value, (list, tuple, np.ndarray)):

                # Case 1: list/tuple of arrays or lists (ragged)
                if isinstance(value, (list, tuple)) and any(
                    isinstance(v, (list, tuple, np.ndarray)) for v in value
                ):
                    for v in value:
                        try:
                            flat_values.extend(np.asarray(v, dtype=float).flatten().tolist())
                        except Exception:
                            flat_values.append(-1.0)

                # Case 2: normal numeric array / flat list
                else:
                    try:
                        flat_values.extend(np.asarray(value, dtype=float).flatten().tolist())
                    except Exception:
                        flat_values.append(-1.0)

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
        self.subsets = get_subsets(set(range(self.beta)))
        self.median_computation_delay = median_computation_delay
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.task = task
        self.epochs = epochs

        # Encoder is now integrated into the model (no separate encoder instance)
        # Build integrated Encoder + DQN model
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
        """
        Build integrated model: Encoder + DQN layers (trained end-to-end).
        Uses Functional API to handle variable-length inputs.

        Architecture:
        - Input: (batch, variable_length, 1) - raw flattened request state
        - Encoder: Dense → Dense → GlobalAvgPool → Dense (output: 32-dim)
        - DQN: Dense → BN → Dense → BN → Dense (output: nA actions)
        """
        # Input: variable-length flattened state (batch, variable_length, 1)
        encoder_input = keras.Input(shape=(None, 1), name='raw_state_input')

        # ===== ENCODER LAYERS =====
        # These will be trained along with the DQN!
        x = layers.Dense(128, activation='relu', name='encoder_expand')(encoder_input)
        x = layers.Dense(64, activation='relu', name='encoder_project')(x)
        x = layers.GlobalAveragePooling1D(name='encoder_pool')(x)  # (batch, 64)
        encoded = layers.Dense(32, activation='relu', name='encoder_output')(x)  # (batch, 32)

        # ===== DQN LAYERS (operating on fixed 32-dim encoded state) =====
        x = layers.Dense(64, activation='sigmoid', name='dqn_hidden1')(encoded)
        x = layers.BatchNormalization(name='dqn_bn1')(x)
        x = layers.Dense(128, activation='sigmoid', name='dqn_hidden2')(x)
        x = layers.BatchNormalization(name='dqn_bn2')(x)
        q_values = layers.Dense(self.nA, activation='softmax', name='dqn_output')(x)

        # Create the combined model
        model = keras.Model(inputs=encoder_input, outputs=q_values, name='integrated_encoder_dqn')

        model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate)
        )

        print("[AGENT] ✅ Built integrated Encoder+DQN model (end-to-end trainable)")
        print(model.summary())
        return model


    def store(self, state_request, action, reward, next_state_request):
        """
        Store RAW (unencoded) states in replay memory.
        The model will encode them during training, allowing encoder to learn.
        """
        # Convert to flattened arrays and reshape to (variable_length, 1)
        state_vec = request_to_state_array(state_request).reshape(-1, 1)
        next_state_vec = request_to_state_array(next_state_request).reshape(-1, 1)

        # Store RAW states (NOT encoded!)
        self.memory.append((state_vec, action, reward, next_state_vec))


    def get_action(self, request):
        """
        Selects an action (subset of servers) given the current Request.
        Uses epsilon-greedy strategy.
        Model automatically encodes the raw state internally.
        """

        # 1️⃣ Convert Request object → numeric state vector (variable length)
        state_flattened = request_to_state_array(request)

        # 2️⃣ Reshape for model input: (1, variable_length, 1)
        state_tensor = state_flattened.reshape(1, -1, 1).astype(np.float32)

        # 4️⃣ Epsilon-greedy exploration
        if np.random.rand() <= self.epsilon:
            self.exploit_or_explore = np.append(self.exploit_or_explore, "explore")
            action = np.random.randint(0, self.nA)

        else:
            self.exploit_or_explore = np.append(self.exploit_or_explore, "exploit")

            # Model will automatically encode the raw state internally
            action_vals = self.timed_predict(state_tensor)

            # Convert to NumPy for argmax
            action = np.argmax(action_vals.numpy()[0])

        # 5️⃣ Get subset of servers for this action, Access all possible non-empty subsets of servers (0-based indexing)
        return_arr = self.subsets[action]

        # 6️⃣ Log and return
        self.episode_access_rate = np.append(self.episode_access_rate, len(return_arr) / self.beta)
        self.action = np.append(self.action, [return_arr])

        return return_arr, action


    def experience_replay(self, batch_size):
        """
        Sample from memory and train the integrated model.
        States are stored RAW, so model encodes them during forward pass.
        Gradients flow through encoder layers during backpropagation.
        """
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = map(list, zip(*minibatch))

        # Pad states to same length for batching
        # Each state is (variable_length, 1), we need to make them uniform length
        states_padded = tf.keras.preprocessing.sequence.pad_sequences(
            [s.flatten() for s in states],
            padding='post',
            dtype='float32',
            value=-1.0
        )
        next_states_padded = tf.keras.preprocessing.sequence.pad_sequences(
            [s.flatten() for s in next_states],
            padding='post',
            dtype='float32',
            value=-1.0
        )

        # Reshape to (batch, max_length, 1) for model input
        states_padded = states_padded.reshape(batch_size, -1, 1)
        next_states_padded = next_states_padded.reshape(batch_size, -1, 1)

        # Forward pass (model encodes internally)
        current_q = self.model.predict(states_padded, verbose=0)
        next_q_values = self.model.predict(next_states_padded, verbose=0)

        # Compute targets using Bellman equation
        targets = current_q.copy()
        targets[np.arange(batch_size), actions] = rewards + self.reward_gamma * np.amax(next_q_values, axis=1)

        # Train (gradients flow through encoder!)
        hist = self.model.fit(states_padded, targets, epochs=self.epochs, verbose=1, validation_split=0.2)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        self.loss = np.append(self.loss, hist.history['loss'][0])
        self.val_loss = np.append(self.val_loss, hist.history['val_loss'][0])



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
    # NOTE: This method is NO LONGER USED with step/episodic reward architecture.
    # Kept for backward compatibility. Rewards are now computed in controller.py
    def reward(self, action, request):
        """
        DEPRECATED: Used to compute min latency among chosen servers.

        With the new step/episodic reward architecture:
        - Step rewards are computed in controller.compute_step_reward()
        - Episodic rewards are computed in controller.compute_episodic_reward()
        - Training happens once per episode in controller.finalize_episode()
        """
        MEDIAN_LATENCY = self.median_computation_delay

        servers_to_be_queried = action
        obs_latency = self.get_min_delay(request, servers_to_be_queried)

        # Log latency metrics (still useful for analysis)
        self.latencies = np.append(self.latencies, obs_latency)
        self.deviations = np.append(self.deviations, abs(obs_latency - MEDIAN_LATENCY))

        # Return observed latency (not used for training anymore)
        return obs_latency