import pandas as pd
import numpy as np

def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_state_seq(df, t, window_size):
    features = []
    start = t - window_size + 1 if t - window_size + 1 >= 0 else 0
    
    for i in range(start, t + 1):
        row = df.iloc[i]
        state = [
            row['Returns'],
            (row['Adj Close'] - row['SMA_50']) / (row['SMA_50'] + 1e-8),
            (row['Adj Close'] - row['SMA_200']) / (row['SMA_200'] + 1e-8),
            row['RSI'],
            row['Volatility']
        ]
        features.append(state)
    
    if len(features) < window_size:
        pad = [[0, 0, 0, 50, 0]] * (window_size - len(features))
        features = pad + features
        
    return np.array(features)