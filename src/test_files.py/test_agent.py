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

REQUESTS = [
    # Normal
    DummyRequest(2, 4, 100, 50, [1.0, 0.5, 0.2]),

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

print("🧪 TEST 1: request_to_state_array")

for i, req in enumerate(REQUESTS):
    state = request_to_state_array(req)
    print(f"\nRequest {i}")
    print("State shape:", state.shape)
    print("State:", state)

    assert state.ndim == 1
    assert not np.any(np.isnan(state))

print("\n✅ State encoding passed")


# ==============================
# TEST 2: FORWARD PASS
# ==============================

print("\n🧪 TEST 2: model forward pass")

req = REQUESTS[0]
state = request_to_state_array(req).reshape(1, -1, 1).astype(np.float32)

q_vals = agent.model(state, training=False)

print("Q-values:", q_vals.numpy())
print("Q-values shape:", q_vals.shape)

assert q_vals.shape == (1, agent.nA)
assert np.allclose(np.sum(q_vals.numpy()), 1.0, atol=1e-5)

print("✅ Forward pass passed")


# ==============================
# TEST 3: ACTION SELECTION
# ==============================

print("\n🧪 TEST 3: get_action")

for i, req in enumerate(REQUESTS):
    subset, action_idx = agent.get_action(req)

    print(f"\nRequest {i}")
    print("Action index:", action_idx)
    print("Chosen subset:", subset)

    assert 0 <= action_idx < agent.nA
    assert len(subset) > 0
    assert max(subset) < agent.beta

print("\n✅ Action selection passed")


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
