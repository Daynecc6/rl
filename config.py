# config.py
import torch

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    
    # Training parameters
    EPISODES = 50
    WINDOW_SIZE = 10
    BATCH_SIZE = 1024
    GAMMA = 0.99
    N_STEPS = 3
    MEMORY_CAPACITY = 100000
    
    # Network parameters
    FEATURE_DIM = 14  # Increased for additional features
    ACTION_SIZE = 3
    LSTM_HIDDEN = 128  # Increased from 64
    
    # Optimizer parameters
    LEARNING_RATE = 0.002
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
    
    # New parameters for enhanced trading
    TRANSACTION_COST = 0.001  # 0.1% transaction cost
    MAX_POSITION_SIZE = 1.0   # Maximum position size (1.0 = 100% of capital)
    
    # Confidence-based position sizing
    CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for full position size
    
    # Risk management parameters
    STOP_LOSS_PCT = 0.05      # 5% stop loss (dynamic)
    TAKE_PROFIT_PCT = 0.10    # 10% take profit (dynamic)
    
    # Volatility regime parameters
    LOW_VOL_THRESHOLD = 0.01  # 1% daily standard deviation
    HIGH_VOL_THRESHOLD = 0.03 # 3% daily standard deviation
    
    # API keys
    ALPHA_VANTAGE_API_KEY = 'lol'
    
    # Sentiment analysis parameters
    SENTIMENT_WINDOW = 5      # Number of days for sentiment smoothing