import numpy as np


def normalize_state(state):
    state = np.array(state, dtype=np.float32)
    state = np.clip(state, -10, 10)
    state = state / 10.0
    return state
