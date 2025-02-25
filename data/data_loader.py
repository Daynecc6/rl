import os
import time
import pandas as pd
import pandas_datareader.data as web
from alpha_vantage.timeseries import TimeSeries
from data.indicators import compute_RSI, compute_MACD, compute_bollinger_bands, compute_ATR
from config import Config

CACHE_FILE = os.path.join(os.getcwd(), "data_cache.csv")

def fetch_fred_data(symbol, start_date, end_date):
    """Fetches data from FRED (No API key required)."""
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    data = web.DataReader(symbol, 'fred', start_dt, end_dt)
    return data.rename(columns={symbol: "VIX"})

def fetch_alpha_vantage_data(symbol, start_date, end_date):
    """Fetches stock data from Alpha Vantage (Requires API key)."""
    ts = TimeSeries(key=Config.ALPHA_VANTAGE_API_KEY, output_format="pandas")
    data, _ = ts.get_daily(symbol=symbol, outputsize="full")
    time.sleep(12)  # Avoid rate limiting

    # Rename columns to standard names
    data.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. volume": "Volume"
    }, inplace=True)
    
    # Also add Adj Close (same as Close for this data source)
    data["Adj Close"] = data["Close"]
    
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)

    # Clip data with boolean indexing to avoid partial-slice errors
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    data = data[(data.index >= start_dt) & (data.index <= end_dt)]
    return data

def download_sp500_data(start_date=Config.TRAIN_START_DATE, end_date=Config.TEST_END_DATE):
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached data from {CACHE_FILE}...")
        df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
    else:
        print("Downloading SPY (S&P 500) and VIX data...")
        try:
            spy_data = fetch_alpha_vantage_data("SPY", start_date, end_date)
            vix_data = fetch_fred_data("VIXCLS", start_date, end_date)
        except Exception as e:
            print(f"Error downloading data: {e}")
            raise

        # Merge and fill - fixed deprecated method
        df = spy_data.join(vix_data, how="left")
        df = df.ffill()  # Use ffill() instead of fillna(method="ffill")

        # Calculate indicators
        df["Returns"] = df["Adj Close"].pct_change()
        df["SMA_50"] = df["Adj Close"].rolling(window=50).mean()
        df["SMA_200"] = df["Adj Close"].rolling(window=200).mean()
        df["Volatility"] = df["Returns"].rolling(window=20).std()
        df["RSI"] = compute_RSI(df["Adj Close"], period=14)
        df["MACD"] = compute_MACD(df["Adj Close"])
        df["BBP"] = compute_bollinger_bands(df["Adj Close"])
        df["ATR"] = compute_ATR(df)

        df.dropna(inplace=True)
        print(f"Downloaded {len(df)} rows of data.")

        # Cache to CSV
        df.to_csv(CACHE_FILE)
        print(f"Data cached to {CACHE_FILE}")

    return df

def split_data(df, train_end_date=Config.TRAIN_END_DATE):
    # Convert the train_end_date to a Timestamp so slicing won't fail
    train_dt = pd.to_datetime(train_end_date)
    train_df = df.loc[df.index <= train_dt]
    test_df = df.loc[df.index > train_dt]
    print(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows.")
    return train_df, test_df