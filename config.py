# config.py
import torch

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    
    # Training parameters
    EPISODES = 3
    WINDOW_SIZE = 10
    BATCH_SIZE = 1024
    GAMMA = 0.99
    N_STEPS = 3
    MEMORY_CAPACITY = 100000
    
    # Network parameters
    FEATURE_DIM = 9
    ACTION_SIZE = 3
    LSTM_HIDDEN = 64
    
    # Optimizer parameters
    LEARNING_RATE = 0.003
    WEIGHT_DECAY = 0.01
    
    # Exploration parameters
    EPSILON_START = 1.0
    EPSILON_MIN = 0.3
    EPSILON_DECAY = 0.998
    
    # PER parameters
    BETA_START = 0.4
    BETA_FRAMES = 50000
    
    # Data parameters
    TRAIN_START_DATE = '2010-01-01'
    TRAIN_END_DATE = '2020-12-31'
    TEST_END_DATE = '2023-12-31'

    ALPHA_VANTAGE_API_KEY = 'lol'