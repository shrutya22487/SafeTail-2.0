import random
import time
import traceback
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

import constants
import servers
import user

BASE_DIR = Path(__file__).resolve().parent


def get_subsets(fullset):
    """Helper: return all non-empty subsets of a set"""
    # Converting set to list for indexing
    listrep = list(fullset)
    subsets = []
    # There are 2^n subsets for a set of size n
    # Looping over all the subsets
    for i in range(2 ** len(listrep)):
        # Building each subset using bitmasking
        subset = []
        for k in range(len(listrep)):
            if i & 1 << k:
                subset.append(listrep[k])
        subsets.append(np.array(subset))  # convert each subset to np.array
    return np.array(subsets[1:], dtype=object)  # return numpy array of subsets


def request_to_state_array(request_obj, remove_nan=True):
    """
    Converts a Request object into a flattened numeric NumPy array.
    Handles nested arrays, lists, dicts, and None values safely.
    Replaces None, NaN, and ±inf with -1.0.
    """
    flat_values = []
    for attr, value in vars(request_obj).items():
        try:
            # None → -1.0
            if value is None:
                flat_values.append(-1.0)
            elif isinstance(value, dict):
                continue  # skip dictionaries entirely
            # Scalars
            elif isinstance(value, (int, float, np.integer, np.floating)):
                val = float(value)
                if remove_nan and not np.isfinite(val):
                    print(f"[SANITIZE] Attribute '{attr}' scalar was {val}, replaced with -1.0")
                    val = -1.0
                flat_values.append(val)
            # Lists / tuples / numpy arrays
            elif isinstance(value, (list, tuple, np.ndarray)):
                # Ragged structures
                if isinstance(value, (list, tuple)) and any(
                        isinstance(v, (list, tuple, np.ndarray)) for v in value
                ):
                    for v in value:
                        try:
                            flat = np.asarray(v, dtype=float).flatten()
                            if remove_nan and not np.isfinite(flat).all():
                                print(f"[SANITIZE] Attribute '{attr}' contained NaN/inf")
                                flat = np.nan_to_num(
                                    flat,
                                    nan=-1.0,
                                    posinf=-1.0,
                                    neginf=-1.0
                                )
                            flat_values.extend(flat.tolist())
                        except Exception:
                            flat_values.append(-1.0)
                # Normal numeric array / flat list
                else:
                    try:
                        flat = np.asarray(value, dtype=float).flatten()
                        if remove_nan and not np.isfinite(flat).all():
                            print(f"[SANITIZE] Attribute '{attr}' contained NaN/inf")
                            flat = np.nan_to_num(
                                flat,
                                nan=-1.0,
                                posinf=-1.0,
                                neginf=-1.0
                            )
                        flat_values.extend(flat.tolist())
                    except Exception:
                        flat_values.append(-1.0)
            # Unsupported types
            else:
                flat_values.append(-1.0)
        except Exception as e:
            print(f"[AGENT][ERROR] Failed to process attribute '{attr}' ({type(value)}): {e}")
            traceback.print_exc()
            flat_values.append(-1.0)
    # Final safety conversion (last line of defense)
    try:
        arr = np.asarray(flat_values, dtype=float)
        if remove_nan:
            arr = np.nan_to_num(
                arr,
                nan=-1.0,
                posinf=-1.0,
                neginf=-1.0
            )
        return arr
    except Exception as e:
        print(f"[AGENT][FATAL] Could not create NumPy array: {e}")
        traceback.print_exc()
        return np.full(1, -1.0, dtype=float)


class DQNAgent:
    def __init__(self, states, actions, alpha, reward_gamma, epsilon,
                 epsilon_min, epsilon_decay, batch_size, beta,
                 median_computation_delay, learning_rate, task, epochs, request: user.Request,
                 server_list: list[servers.Server], lr_decay_rate=0.995, lr_min=1e-5,
                 ):
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
        self.lr_decay_rate = constants.lr_decay_rate
        self.lr_min = constants.lr_min
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
        self.remove_nan_in_state = False  # whether to sanitize NaN/inf in state representation

        # DEBUGGING: dump replay batches to file
        self.debug_replay_dump = False
        self.replay_dump_file = BASE_DIR / "replay_dump" / "replay_debug_dump.txt"
        # Ensure directory exists
        self.replay_dump_file.parent.mkdir(parents=True, exist_ok=True)
        # Clear file once at start
        with open(self.replay_dump_file, "w") as f:
            f.write("REPLAY DEBUG LOG\n")

    def decay_learning_rate(self):
        """Exponentially decay learning rate after each replay."""
        current_lr = float(self.model.optimizer.learning_rate)
        new_lr = max(current_lr * self.lr_decay_rate, self.lr_min)
        self.model.optimizer.learning_rate.assign(new_lr)
        return new_lr

    # Helper function to dump replay batch for debugging
    def dump_replay_batch(
            self,
            states,
            next_states,
            states_padded,
            next_states_padded,
            actions,
            rewards
    ):
        with open(self.replay_dump_file, "a") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("NEW EXPERIENCE REPLAY MINIBATCH\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Batch size: {len(states)}\n\n")
            for i in range(len(states)):
                f.write(f"--- SAMPLE {i} ---\n")
                f.write("RAW STATE (flattened):\n")
                f.write(f"{states[i].flatten().tolist()}\n\n")
                f.write("RAW NEXT STATE (flattened):\n")
                f.write(f"{next_states[i].flatten().tolist()}\n\n")
                f.write(f"ACTION: {actions[i]}\n")
                f.write(f"REWARD: {rewards[i]}\n\n")
                f.write("PADDED STATE:\n")
                f.write(f"{states_padded[i].flatten().tolist()}\n\n")
                f.write("PADDED NEXT STATE:\n")
                f.write(f"{next_states_padded[i].flatten().tolist()}\n\n")
            f.write("\n")

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
        - Encoder: Dense → Dropout → Dense → Dropout → GlobalAvgPool → Dense (output: 24-dim)
        - DQN: Dense → BN → Dropout → Dense → BN → Dropout → Dense (output: nA actions)

        Changes to reduce overfitting:
        - Reduced layer sizes (128→64, 64→32, 64→48, 128→64)
        - Replaced sigmoid with relu in DQN (sigmoid saturates, slows learning, encourages memorization)
        - Added Dropout(0.3) after encoder dense layers
        - Added Dropout(0.4) after DQN hidden layers
        - Removed softmax from q_values output (incorrect for DQN - use linear)
        """
        # Input: variable-length flattened state (batch, variable_length, 1)
        encoder_input = keras.Input(shape=(None, 1), name='raw_state_input')

        # ===== ENCODER LAYERS =====
        x = layers.Dense(64, activation='relu', name='encoder_expand')(encoder_input)  # 128 → 64
        x = layers.Dropout(0.3, name='encoder_drop1')(x)
        x = layers.Dense(32, activation='relu', name='encoder_project')(x)  # 64 → 32
        x = layers.Dropout(0.3, name='encoder_drop2')(x)
        x = layers.GlobalAveragePooling1D(name='encoder_pool')(x)  # (batch, 32)
        encoded = layers.Dense(24, activation='relu', name='encoder_output')(x)  # 32 → 24

        # ===== DQN LAYERS (operating on fixed 24-dim encoded state) =====
        x = layers.Dense(48, activation='relu', name='dqn_hidden1')(encoded)  # 64 → 48
        x = layers.BatchNormalization(name='dqn_bn1')(x)
        x = layers.Dropout(0.4, name='dqn_drop1')(x)
        x = layers.Dense(64, activation='relu', name='dqn_hidden2')(x)  # 128 → 64
        x = layers.BatchNormalization(name='dqn_bn2')(x)
        x = layers.Dropout(0.4, name='dqn_drop2')(x)
        q_values = layers.Dense(self.nA, activation='linear', name='dqn_output')(x)  # linear, not softmax

        # Create the combined model
        model = keras.Model(inputs=encoder_input, outputs=q_values, name='integrated_encoder_dqn')
        model.compile(
            loss='mse',  # mse for Q-value regression
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
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
        state_vec = request_to_state_array(state_request, self.remove_nan_in_state).reshape(-1, 1)
        next_state_vec = request_to_state_array(next_state_request, self.remove_nan_in_state).reshape(-1, 1)
        # Store RAW states (NOT encoded!)
        self.memory.append((state_vec, action, reward, next_state_vec))

    def get_action(self, request):
        """
        Selects an action (subset of servers) given the current Request.
        Uses epsilon-greedy strategy.
        Model automatically encodes the raw state internally.
        """
        # 1️⃣ Convert Request object → numeric state vector (variable length)
        state_flattened = request_to_state_array(request, self.remove_nan_in_state)
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

        if self.debug_replay_dump:
            self.dump_replay_batch(
                states=states,
                next_states=next_states,
                states_padded=states_padded,
                next_states_padded=next_states_padded,
                actions=actions,
                rewards=rewards
            )

        # Forward pass (model encodes internally)
        current_q = self.model.predict(states_padded, verbose=0)
        next_q_values = self.model.predict(next_states_padded, verbose=0)
        # Compute targets using Bellman equation
        targets = current_q.copy()
        targets[np.arange(batch_size), actions] = (
                rewards + self.reward_gamma * np.amax(next_q_values, axis=1)
        )
        # Clip targets to a sane range to prevent MSE loss explosion
        targets = np.clip(targets, -10.0, 10.0)
        # Train (gradients flow through encoder!)
        hist = self.model.fit(
            states_padded,
            targets,
            epochs=constants.epochs,
            verbose=1,
            validation_split=0.2
        )
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        self.decay_learning_rate()

        self.loss = np.append(self.loss, hist.history['loss'][0])
        self.val_loss = np.append(self.val_loss, hist.history['val_loss'][0])
        self.rewards = np.append(self.rewards, np.mean(rewards))

    def get_min_delay(self, request, servers_to_be_queried):
        """
        Query the servers and get the minimum delay among them.
        """
        min_delay = float('inf')
        for server in servers_to_be_queried:
            min_delay = min(min_delay, self.server_list[server].compute_request_time(request))
        return min_delay

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

    def plot_training_loss(self, save_path=None, show_plot=True):
        """
        Plot training and validation loss curves.

        Args:
            save_path (str, optional): Path to save the plot. If None, plot is not saved.
            show_plot (bool): Whether to display the plot interactively.
        """
        if len(self.loss) == 0:
            print("[AGENT] No training loss dataset to plot.")
            return

        plt.figure(figsize=(12, 6))

        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(self.loss, label='Training Loss', color='blue', linewidth=2)
        plt.xlabel('Training Iteration', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot validation loss
        plt.subplot(1, 2, 2)
        if len(self.val_loss) > 0:
            plt.plot(self.val_loss, label='Validation Loss', color='orange', linewidth=2)
            plt.xlabel('Training Iteration', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('Validation Loss over Time', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No validation loss dataset',
                     ha='center', va='center', fontsize=14)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[AGENT] Loss plot saved to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_all_metrics(self, save_dir=None, show_plot=True):
        """
        Create comprehensive visualization of all training metrics.
        Only shows plots with available dataset, displays 'No dataset yet' for empty arrays.

        Args:
            save_dir (str, optional): Directory to save plots. If None, plots are not saved.
            show_plot (bool): Whether to display plots interactively.
        """
        fig = plt.figure(figsize=(16, 12))

        # 1. Training and Validation Loss
        ax1 = plt.subplot(3, 3, 1)
        has_data = False
        if len(self.loss) > 0:
            ax1.plot(self.loss, label='Training Loss', color='blue', alpha=0.7, linewidth=1.5)
            has_data = True
        if len(self.val_loss) > 0:
            ax1.plot(self.val_loss, label='Validation Loss', color='orange', alpha=0.7, linewidth=1.5)
            has_data = True
        if has_data:
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Loss')
            ax1.set_title('Loss Curves', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No loss dataset yet', ha='center', va='center',
                     fontsize=11, transform=ax1.transAxes, color='gray')
            ax1.set_title('Loss Curves', fontweight='bold')

        # 2. Epsilon Decay
        ax2 = plt.subplot(3, 3, 2)
        if len(self.epsilon_curve) > 0:
            ax2.plot(self.epsilon_curve, color='green', linewidth=2)
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Epsilon')
            ax2.set_title('Exploration Rate (Epsilon) Decay', fontweight='bold')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No epsilon dataset yet', ha='center', va='center',
                     fontsize=11, transform=ax2.transAxes, color='gray')
            ax2.set_title('Exploration Rate (Epsilon) Decay', fontweight='bold')

        # 3. Explore vs Exploit Distribution
        ax3 = plt.subplot(3, 3, 3)
        if len(self.exploit_or_explore) > 0:
            unique, counts = np.unique(self.exploit_or_explore, return_counts=True)
            colors = ['#ff6b6b' if u == 'explore' else '#4ecdc4' for u in unique]
            ax3.bar(unique, counts, color=colors)
            ax3.set_ylabel('Count')
            ax3.set_title('Exploration vs Exploitation', fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
        else:
            ax3.text(0.5, 0.5, 'No strategy dataset yet', ha='center', va='center',
                     fontsize=11, transform=ax3.transAxes, color='gray')
            ax3.set_title('Exploration vs Exploitation', fontweight='bold')

        # 4. Rewards over Time
        ax4 = plt.subplot(3, 3, 4)

        if len(self.rewards) > 0:
            ax4.plot(self.rewards, color='purple', alpha=0.7, linewidth=1.5)

            if hasattr(self, "testing_start_reward_index"):
                ax4.axvline(
                    x=self.testing_start_reward_index,
                    color='black',
                    linestyle='--',
                    linewidth=1,
                    label='Testing Starts'
                )

            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Reward')
            ax4.set_title('Reward History', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        else:
            ax4.text(
                0.5, 0.5, 'No reward dataset yet',
                ha='center', va='center',
                transform=ax4.transAxes
            )
            ax4.set_title('Reward History', fontweight='bold')

        ax5 = plt.subplot(3, 3, 5)

        if len(self.latencies) > 0:
            ax5.plot(self.latencies, color='brown', alpha=0.7, linewidth=1.4)

            if hasattr(self, "testing_start_latency_index"):
                ax5.axvline(
                    x=self.testing_start_latency_index,
                    color='black',
                    linestyle='--',
                    linewidth=1,
                    label='Testing Starts'
                )

            ax5.set_xlabel('Request')
            ax5.set_ylabel('Latency (ms)')
            ax5.set_title('Observed Latencies', fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        else:
            ax5.text(0.5, 0.5, 'No latency dataset yet',
                     ha='center', va='center',
                     transform=ax5.transAxes)
            ax5.set_title('Observed Latencies', fontweight='bold')

        # 6. Server Access Rate
        ax6 = plt.subplot(3, 3, 6)

        if len(self.episode_access_rate) > 0:
            ax6.plot(self.episode_access_rate, color='teal', alpha=0.7, linewidth=1.4)

            self._draw_testing_boundary(ax6)

            ax6.axhline(
                y=np.mean(self.episode_access_rate),
                color='orange',
                linestyle='--',
                linewidth=1.5,
                label=f'Mean: {np.mean(self.episode_access_rate):.2f}'
            )

            ax6.set_xlabel('Episode')
            ax6.set_ylabel('Access Rate')
            ax6.set_title('Server Access Rate', fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        else:
            ax6.text(0.5, 0.5, 'No access-rate dataset yet',
                     ha='center', va='center',
                     transform=ax6.transAxes)
            ax6.set_title('Server Access Rate', fontweight='bold')

        # 7. Latency Deviations
        ax7 = plt.subplot(3, 3, 7)

        if len(self.deviations) > 0:
            ax7.plot(self.deviations, color='darkred', alpha=0.7, linewidth=1.4)

            self._draw_testing_boundary(ax7)

            if hasattr(self, "testing_start_deviation_index"):
                ax7.axvline(
                    x=self.testing_start_deviation_index,
                    color='black',
                    linestyle='--',
                    linewidth=1,
                    label='Testing Starts'
                )

            ax7.set_xlabel('Request')
            ax7.set_ylabel('Deviation')
            ax7.set_title('Latency Deviations', fontweight='bold')
            ax7.legend()
            ax7.grid(True, alpha=0.3)

        else:
            ax7.text(0.5, 0.5, 'No deviation dataset yet',
                     ha='center', va='center',
                     transform=ax7.transAxes)
            ax7.set_title('Latency Deviations', fontweight='bold')

        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir) / 'all_metrics.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[AGENT] Comprehensive metrics plot saved to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def _draw_testing_boundary(self, ax):
        """
        Draw vertical marker where testing phase starts.
        """
        if hasattr(self, "testing_start_index"):
            try:
                idx = int(self.testing_start_index)
                ax.axvline(
                    x=idx,
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    label="Testing Starts"
                )
            except Exception:
                pass

    def save_metrics_summary(self, filepath):
        """
        Save a text summary of all training metrics.

        Args:
            filepath (str): Path to save the summary file.
        """
        # filepath = Path(filepath)
        # filepath.parent.mkdir(parents=True, exist_ok=True)
        #
        # with open(filepath, 'w') as f:
        #     f.write("=" * 80 + "\n")
        #     f.write("DQN AGENT TRAINING METRICS SUMMARY\n")
        #     f.write("=" * 80 + "\n\n")
        #
        #     # Loss statistics
        #     if len(self.loss) > 0:
        #         f.write("TRAINING LOSS:\n")
        #         f.write(f"  Mean: {np.mean(self.loss):.6f}\n")
        #         f.write(f"  Std:  {np.std(self.loss):.6f}\n")
        #         f.write(f"  Min:  {np.min(self.loss):.6f}\n")
        #         f.write(f"  Max:  {np.max(self.loss):.6f}\n")
        #         f.write(f"  Final: {self.loss[-1]:.6f}\n\n")
        #
        #     if len(self.val_loss) > 0:
        #         f.write("VALIDATION LOSS:\n")
        #         f.write(f"  Mean: {np.mean(self.val_loss):.6f}\n")
        #         f.write(f"  Std:  {np.std(self.val_loss):.6f}\n")
        #         f.write(f"  Min:  {np.min(self.val_loss):.6f}\n")
        #         f.write(f"  Max:  {np.max(self.val_loss):.6f}\n")
        #         f.write(f"  Final: {self.val_loss[-1]:.6f}\n\n")
        #
        #     # Rewards
        #     if len(self.rewards) > 0:
        #         f.write("REWARDS:\n")
        #         f.write(f"  Mean: {np.mean(self.rewards):.6f}\n")
        #         f.write(f"  Std:  {np.std(self.rewards):.6f}\n")
        #         f.write(f"  Min:  {np.min(self.rewards):.6f}\n")
        #         f.write(f"  Max:  {np.max(self.rewards):.6f}\n\n")
        #
        #     # Latencies
        #     if len(self.latencies) > 0:
        #         f.write("LATENCIES:\n")
        #         f.write(f"  Mean: {np.mean(self.latencies):.6f}\n")
        #         f.write(f"  Std:  {np.std(self.latencies):.6f}\n")
        #         f.write(f"  Min:  {np.min(self.latencies):.6f}\n")
        #         f.write(f"  Max:  {np.max(self.latencies):.6f}\n")
        #         f.write(f"  Median Baseline: {self.median_computation_delay:.6f}\n\n")
        #
        #     # Exploration stats
        #     if len(self.exploit_or_explore) > 0:
        #         unique, counts = np.unique(self.exploit_or_explore, return_counts=True)
        #         f.write("EXPLORATION vs EXPLOITATION:\n")
        #         for action, count in zip(unique, counts):
        #             f.write(f"  {action}: {count} ({100 * count / len(self.exploit_or_explore):.2f}%)\n")
        #         f.write("\n")
        #
        #     # Prediction times
        #     if len(self.prediction_times) > 0:
        #         f.write("PREDICTION TIMES:\n")
        #         f.write(f"  Mean: {np.mean(self.prediction_times):.6f}s\n")
        #         f.write(f"  Std:  {np.std(self.prediction_times):.6f}s\n")
        #         f.write(f"  Min:  {np.min(self.prediction_times):.6f}s\n")
        #         f.write(f"  Max:  {np.max(self.prediction_times):.6f}s\n\n")
        #
        #     # Memory usage
        #     f.write("REPLAY MEMORY:\n")
        #     f.write(f"  Current size: {len(self.memory)}\n")
        #     f.write(f"  Max capacity: {self.memory.maxlen}\n\n")
        #
        #     # Final epsilon
        #     f.write("EPSILON:\n")
        #     f.write(f"  Final value: {self.epsilon:.6f}\n")
        #     f.write(f"  Min threshold: {self.epsilon_min:.6f}\n")
        #
        # print(f"[AGENT] Metrics summary saved to {filepath}")
        pass
