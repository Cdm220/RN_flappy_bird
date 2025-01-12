import torch


class Config:
    GAMMA = 0.95
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 64
    MEMORY_SIZE = 100000
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.9995
    TARGET_UPDATE_FREQ = 100
    MIN_MEMORY_SIZE = 1000
    EPISODES = 1000

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    HIDDEN_SIZES = [256, 128, 64]
    DROPOUT_RATE = 0.2
