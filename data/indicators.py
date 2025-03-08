import pandas as pd
import numpy as np
from scipy import stats
from config import Config

def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi

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

def compute_volatility_regime(df, low_threshold=Config.LOW_VOL_THRESHOLD, 
                             high_threshold=Config.HIGH_VOL_THRESHOLD, 
                             window=20):
    """
    Compute volatility regime: 
    0 = low volatility, 1 = normal, 2 = high volatility
    """
    # Use daily returns volatility
    returns = df['Returns']
    volatility = returns.rolling(window=window).std()
    
    # Categorize the volatility
    regime = pd.Series(1, index=volatility.index)  # Default is normal
    regime[volatility <= low_threshold] = 0  # Low volatility
    regime[volatility >= high_threshold] = 2  # High volatility
    
    return regime

def compute_trend_strength(df, window=50):
    """
    Compute the strength of the current trend using linear regression.
    Returns the R-squared value and slope.
    """
    price = df['Adj Close']
    trend_strength = pd.Series(index=price.index, dtype=float)
    slope = pd.Series(index=price.index, dtype=float)
    
    for i in range(window, len(price)):
        x = np.arange(window)
        y = price.iloc[i-window:i].values
        slope_val, intercept, r_value, _, _ = stats.linregress(x, y)
        
        trend_strength.iloc[i] = r_value ** 2  # R-squared
        slope.iloc[i] = slope_val
    
    # Fill missing values
    trend_strength.fillna(0, inplace=True)
    slope.fillna(0, inplace=True)
    
    # Normalize slope
    price_mean = price.mean()
    slope = slope / price_mean * 100  # Percent change per day
    
    return trend_strength, slope

def compute_market_regime(df, window=20):
    """
    Determine market regime based on price action:
    -1 = downtrend, 0 = sideways, 1 = uptrend
    """
    price = df['Adj Close']
    # Fast and slow moving averages
    sma_fast = price.rolling(window=window//2).mean()
    sma_slow = price.rolling(window=window).mean()
    
    # Compute the trend based on SMA relationship
    regime = pd.Series(0, index=price.index)  # Default is sideways
    
    # Uptrend: Fast SMA > Slow SMA and both sloped up
    regime[(sma_fast > sma_slow) & (sma_slow > sma_slow.shift(5))] = 1
    
    # Downtrend: Fast SMA < Slow SMA and both sloped down
    regime[(sma_fast < sma_slow) & (sma_slow < sma_slow.shift(5))] = -1
    
    return regime

def compute_mean_reversion(df, window=20):
    """
    Compute mean reversion indicator:
    Percentage deviation from moving average
    """
    price = df['Adj Close']
    sma = price.rolling(window=window).mean()
    
    # Calculate percentage deviation
    deviation = (price - sma) / sma * 100
    
    # Z-score of deviation (standardize)
    z_deviation = (deviation - deviation.rolling(window=100).mean()) / deviation.rolling(window=100).std()
    z_deviation.fillna(0, inplace=True)
    
    return z_deviation

def compute_rsi_divergence(df, window=14):
    """
    Compute RSI divergence signal:
    1 = bullish divergence, -1 = bearish divergence, 0 = no divergence
    """
    price = df['Adj Close']
    rsi = compute_RSI(price, period=window)
    
    divergence = pd.Series(0, index=price.index)
    
    # Look back period for local extrema
    lookback = 10
    
    for i in range(lookback + window, len(price)):
        # Find local price minimum in lookback period
        price_window = price.iloc[i-lookback:i]
        rsi_window = rsi.iloc[i-lookback:i]
        
        price_min_idx = price_window.idxmin()
        price_max_idx = price_window.idxmax()
        
        rsi_min_idx = rsi_window.idxmin()
        rsi_max_idx = rsi_window.idxmax()
        
        # Bullish divergence: price makes lower low but RSI makes higher low
        if price_min_idx > rsi_min_idx and price.loc[price_min_idx] < price.loc[rsi_min_idx] and rsi.loc[price_min_idx] > rsi.loc[rsi_min_idx]:
            divergence.iloc[i] = 1
            
        # Bearish divergence: price makes higher high but RSI makes lower high
        elif price_max_idx > rsi_max_idx and price.loc[price_max_idx] > price.loc[rsi_max_idx] and rsi.loc[price_max_idx] < rsi.loc[rsi_max_idx]:
            divergence.iloc[i] = -1
    
    return divergence

def add_enhanced_features(df):
    """
    Add all enhanced features to dataframe
    """
    # Original technical indicators
    df["Returns"] = df["Adj Close"].pct_change()
    df["SMA_50"] = df["Adj Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Adj Close"].rolling(window=200).mean()
    df["Volatility"] = df["Returns"].rolling(window=20).std()
    df["RSI"] = compute_RSI(df["Adj Close"], period=14)
    df["MACD"] = compute_MACD(df["Adj Close"])
    df["BBP"] = compute_bollinger_bands(df["Adj Close"])
    df["ATR"] = compute_ATR(df)
    
    # New enhanced features
    df["Vol_Regime"] = compute_volatility_regime(df)
    df["Trend_Strength"], df["Slope"] = compute_trend_strength(df)
    df["Market_Regime"] = compute_market_regime(df)
    df["Mean_Reversion"] = compute_mean_reversion(df)
    df["RSI_Divergence"] = compute_rsi_divergence(df)
    
    # Drop NaN values
    df.dropna(inplace=True)
    return df

def get_state_seq(df, t, window_size):
    """
    Create state sequence with enhanced features
    """
    features = []
    start = t - window_size + 1 if t - window_size + 1 >= 0 else 0
    
    for i in range(start, t + 1):
        row = df.iloc[i]
        state = [
            # Original features
            row['Returns'],
            (row['Adj Close'] - row['SMA_50']) / (row['SMA_50'] + 1e-8),
            (row['Adj Close'] - row['SMA_200']) / (row['SMA_200'] + 1e-8),
            row['RSI'] /.100,  # Normalize RSI to 0-1 range
            row['Volatility'],
            row['MACD'],
            row['BBP'],
            row['ATR'],
            row['VIX'] / 100,  # Normalize VIX
            
            # New features
            row['Vol_Regime'] / 2.0,  # Normalize to 0-1
            row['Trend_Strength'],    # Already 0-1
            row['Slope'] / 10.0,      # Normalize slope
            (row['Market_Regime'] + 1) / 2.0, # Convert -1,0,1 to 0,0.5,1
            row['Mean_Reversion'] / 4.0,  # Normalize z-score
        ]
        features.append(state)
    
    if len(features) < window_size:
        # Pad with zeros if we don't have enough history
        pad = [[0] * len(features[0])] * (window_size - len(features))
        features = pad + features
        
    return np.array(features)