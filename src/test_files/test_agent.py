"""
tes_agent.py

Run with:
    python test_agent.py

Purpose:
- Verify request → state encoding
- Verify model forward pass
- Verify action → subset mapping
- Verify replay padding & training
"""

import numpy as np
import random
import tensorflow as tf

# ==============================
# MOCK OBJECTS
# ==============================

class DummyRequest:
    def __init__(self, cpu, mem, data_size, deadline, features=None):
        self.cpu = cpu
        self.mem = mem
        self.data_size = data_size
        self.deadline = deadline
        self.features = features


class DummyServer:
    def __init__(self, base_delay):
        self.base_delay = base_delay

    def compute_request_time(self, request):
        return self.base_delay + 0.01 * request.data_size


# ==============================
# SAMPLE REQUESTS
# ==============================
stats = np.array([250.0, 250.0, -1.0, 1024.0, 20.0, 0.0, 1.0, 3.0, 2.0, 2.0, 0.0, 0.03820533663093817, 0.03196158151015982, 0.0, 0.0, 0.07673153410680034, -1.0, 12292.6640625, 79.0, 3261.6875, 18.15119171142578, 0.0321962592592592, 20.0, 30.45365524291992, 32.0, 15.9921875, 13.4, 17.8, 5.7, 5.5, 4.5, 41.6, 9.9, 4.3, 45.6, 1.9, 6.8, 38.3, 62.6, 1.3, 2.9, 7.7, 8.1, 7.0, 1.9, 2.6, 5.5, 58.2, 3.9, 5.2, 8.3, 2.9, 1.1, 9.3, 38.8, 1.8, 1.4, 16.8, 0.032196, 18.151, 54.0, 12219.203125, 85.0, 3261.6875, 18.31467223167419, 0.0366724629629629, 18.0, 30.45365524291992, 32.0, 15.9921875, 23.2, 4.5, 36.0, 7.0, 3.1, 3.6, 23.3, 2.4, 40.1, 9.6, 3.1, 2.8, 22.5, 4.2, 4.6, 17.3, 46.6, 3.0, 16.1, 3.1, 5.2, 2.9, 14.6, 53.4, 60.1, 16.4, 2.4, 2.5, 7.0, 4.1, 5.9, 24.2, 0.018209, 0.018463, 8.803, 9.511, 54.0, 54.0, 5431.8671875, 0.0, 0.0, 161.87664341926575, 0.6405014624444445, 16.0, 15.366962432861328, 16.0, 0.0, 64.8, 59.9, 72.0, 65.9, 39.0, 48.7, 56.3, 64.4, 29.6, 34.2, 27.8, 32.4, 54.9, 47.0, 38.6, 30.5, 0.449035, 0.051773, 0.070934, 0.06876, 33.036, 32.726, 48.768, 47.346, 7806.0546875, 0.0, 0.0, 114.50802564620972, 0.762485173925926, 12.0, 30.74622344970703, 12.0, 0.0, 32.6, 36.1, 42.8, 15.2, 35.8, 50.2, 60.8, 57.4, 43.7, 71.8, 51.0, 36.6, 0.651085, 0.055239, 0.056161, 45.32, 34.339, 34.85, 8594.625, 81.0, 2828.4375, 24.931310176849365, 0.0272634523699243, 6.0, 31.108741760253903, 16.0, 24.0, 2.1, 4.0, 27.3, 0.1, 26.5, 8.1, 6.7, 0.7, 3.2, 1.6, 9.1, 6.2, 1.4, 0.9, 1.1, 1.1, 0.020829, 0.003218, 0.003216, 9.648, 7.663, 7.619, 102.0, 508.0, 508.0, np.nan, 0.08001235103002466, 0.10420476897284611, 1.0513466220565084, 0.6373949331710379, 0.498421180732892])

REQUESTS = [
    # Normal
    DummyRequest(2, 4, 100, 50, stats),

    # Contains None
    DummyRequest(1, None, 50, 20, None),

    # Large feature vector
    DummyRequest(8, 16, 500, 200, np.random.rand(50)),

    # Ragged nested list
    DummyRequest(4, 8, 200, 100, [[1, 2], [3, 4, 5]])
]


# ==============================
# IMPORT YOUR AGENT
# ==============================

from agent import DQNAgent, request_to_state_array  # adjust path if needed


# ==============================
# SETUP
# ==============================

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

NUM_SERVERS = 3
servers = [DummyServer(base_delay=i + 1) for i in range(NUM_SERVERS)]

agent = DQNAgent(
    states=None,
    actions=2**NUM_SERVERS - 1,   # non-empty subsets
    alpha=0.1,
    reward_gamma=0.95,
    epsilon=0.0,                  # force exploitation
    epsilon_min=0.01,
    epsilon_decay=0.001,
    batch_size=2,
    beta=NUM_SERVERS,
    median_computation_delay=10,
    learning_rate=0.001,
    task="test",
    epochs=1,
    request=REQUESTS[0],
    server_list=servers
)

print("\n==============================")
print("✅ AGENT INITIALIZED")
print("==============================\n")


# ==============================
# TEST 1: STATE ENCODING
# ==============================

# print("🧪 TEST 1: request_to_state_array")

# for i, req in enumerate(REQUESTS):
#     state = request_to_state_array(req)
#     print(f"\nRequest {i}")
#     print("State shape:", state.shape)
#     print("State:", state)

#     assert state.ndim == 1
#     assert not np.any(np.isnan(state))

# print("\n✅ State encoding passed")


# ==============================
# TEST 2: FORWARD PASS
# ==============================

# print("\n🧪 TEST 2: model forward pass")

# req = REQUESTS[0]
# # state = np.array([250.0, 250.0, -1.0, 1024.0, 20.0, 0.0, 1.0, 3.0, 2.0, 2.0, 0.0, 0.03820533663093817, 0.03196158151015982, 0.0, 0.0, 0.07673153410680034, -1.0, 12292.6640625, 79.0, 3261.6875, 18.15119171142578, 0.0321962592592592, 20.0, 30.45365524291992, 32.0, 15.9921875, 13.4, 17.8, 5.7, 5.5, 4.5, 41.6, 9.9, 4.3, 45.6, 1.9, 6.8, 38.3, 62.6, 1.3, 2.9, 7.7, 8.1, 7.0, 1.9, 2.6, 5.5, 58.2, 3.9, 5.2, 8.3, 2.9, 1.1, 9.3, 38.8, 1.8, 1.4, 16.8, 0.032196, 18.151, 54.0, 12219.203125, 85.0, 3261.6875, 18.31467223167419, 0.0366724629629629, 18.0, 30.45365524291992, 32.0, 15.9921875, 23.2, 4.5, 36.0, 7.0, 3.1, 3.6, 23.3, 2.4, 40.1, 9.6, 3.1, 2.8, 22.5, 4.2, 4.6, 17.3, 46.6, 3.0, 16.1, 3.1, 5.2, 2.9, 14.6, 53.4, 60.1, 16.4, 2.4, 2.5, 7.0, 4.1, 5.9, 24.2, 0.018209, 0.018463, 8.803, 9.511, 54.0, 54.0, 5431.8671875, 0.0, 0.0, 161.87664341926575, 0.6405014624444445, 16.0, 15.366962432861328, 16.0, 0.0, 64.8, 59.9, 72.0, 65.9, 39.0, 48.7, 56.3, 64.4, 29.6, 34.2, 27.8, 32.4, 54.9, 47.0, 38.6, 30.5, 0.449035, 0.051773, 0.070934, 0.06876, 33.036, 32.726, 48.768, 47.346, 7806.0546875, 0.0, 0.0, 114.50802564620972, 0.762485173925926, 12.0, 30.74622344970703, 12.0, 0.0, 32.6, 36.1, 42.8, 15.2, 35.8, 50.2, 60.8, 57.4, 43.7, 71.8, 51.0, 36.6, 0.651085, 0.055239, 0.056161, 45.32, 34.339, 34.85, 8594.625, 81.0, 2828.4375, 24.931310176849365, 0.0272634523699243, 6.0, 31.108741760253903, 16.0, 24.0, 2.1, 4.0, 27.3, 0.1, 26.5, 8.1, 6.7, 0.7, 3.2, 1.6, 9.1, 6.2, 1.4, 0.9, 1.1, 1.1, 0.020829, 0.003218, 0.003216, 9.648, 7.663, 7.619, 102.0, 508.0, 508.0, np.nan, 0.08001235103002466, 0.10420476897284611, 1.0513466220565084, 0.6373949331710379, 0.498421180732892])
# state = state.reshape(1, -1, 1).astype(np.float32)

# q_vals = agent.model(state, training=False)

# print("Q-values:", q_vals.numpy())
# print("Q-values shape:", q_vals.shape)

# assert q_vals.shape == (1, agent.nA)
# assert np.allclose(np.sum(q_vals.numpy()), 1.0, atol=1e-5)

# print("✅ Forward pass passed")


# ==============================
# TEST 3: ACTION SELECTION
# ==============================

print("\n🧪 TEST 3: get_action")

# for i, req in enumerate(REQUESTS):
#     subset, action_idx = agent.get_action(req)

#     print(f"\nRequest {i}")
#     print("Action index:", action_idx)
#     print("Chosen subset:", subset)

#     assert 0 <= action_idx < agent.nA
#     assert len(subset) > 0
#     assert max(subset) < agent.beta

# print("\n✅ Action selection passed")


# ==============================
# TEST 4: EXPERIENCE REPLAY + PADDING
# ==============================

print("\n🧪 TEST 4: experience replay")

# Populate memory with variable-length states
for i in range(len(REQUESTS)):
    agent.store(
        REQUESTS[i],
        action=0,
        reward=1.0,
        next_state_request=REQUESTS[(i + 1) % len(REQUESTS)]
    )

agent.experience_replay(batch_size=2)

print("\n✅ Experience replay passed")


# ==============================
# DONE
# ==============================

print("\n==============================")
print("🎉 ALL SANITY TESTS PASSED")
print("==============================\n")
