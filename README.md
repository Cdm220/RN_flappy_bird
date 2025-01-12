# RN_flappy_bird

# Arhitecture and implementation details

## Neural Network Arhitecture
- Input Layer: 10 neurons (preprocessed state features)
- Hidden Layers:
  1. Layer 1: 256 neurons with ReLU activation + Dropout (0.2)
  2. Layer 2: 128 neurons with ReLU activation + Dropout (0.2)
  3. Layer 3: 64 neurons with ReLU activation
- Output Layer: 2 neurons (one for each action)
- Weight Initialization: Xavier Uniform
- Optimizer: Adam
- Loss Function: Smooth L1 Loss (Huber Loss)

## State normalization
```
def normalize_state(state):
    state = np.array(state, dtype=np.float32)
    state = np.clip(state, -10, 10)
    state = state / 10.0
    return state
```

# Hyperparameters
```
GAMMA = 0.95
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
MEMORY_SIZE = 100000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995
TARGET_UPDATE_FREQ = 100
MIN_MEMORY_SIZE = 1000
EPISODES = 5000
```

# Experimental results
1. 1000 episodes
```
Best Score: 20
Average Score: 6.3
```
2. 3000 episodes
```
Best Score: 10
Avg Score: 3.7
```  
3. 5000 episodes
```
Best Score: 13
Average Score: 4.7
```
# Highest score
![image](https://github.com/user-attachments/assets/251306b8-4649-42d9-8934-f261759362a8)
