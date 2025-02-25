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

    # Python
def compute_MACD(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_hist

def compute_bollinger_bands(series, window=20, num_std=2):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std
    # Compute the %B which indicates where the price is in-between the bands
    percent_b = (series - lower_band) / (upper_band - lower_band + 1e-8)
    return percent_b

# Python
def compute_ATR(df, window=14):
    # Use High, Low, and Adj Close as the price data
    high = df['High']
    low = df['Low']
    close = df['Adj Close']
    
    # Compute previous close shifted by one
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

# Python
# Python
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
            row['Volatility'],
            row['MACD'],
            row['BBP'],
            row['ATR'],  # ATR feature
            row['VIX']   # VIX feature
        ]
        features.append(state)
    
    if len(features) < window_size:
        pad = [[0, 0, 0, 50, 0, 0, 0, 0, 0]] * (window_size - len(features))
        features = pad + features
        
    return np.array(features)


