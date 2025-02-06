import yfinance as yf
from data.indicators import compute_RSI
from config import Config

def download_sp500_data(start_date=Config.TRAIN_START_DATE, 
                       end_date=Config.TEST_END_DATE):
    print("Downloading S&P 500 data...")
    df = yf.download('^GSPC', start=start_date, end=end_date)
    
    # Calculate indicators
    df['Returns'] = df['Adj Close'].pct_change()
    df['SMA_50'] = df['Adj Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Adj Close'].rolling(window=200).mean()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    df['RSI'] = compute_RSI(df['Adj Close'], period=14)
    
    df = df.dropna()
    print(f"Downloaded {len(df)} days of data")
    return df

def split_data(df, train_end_date=Config.TRAIN_END_DATE):
    train_df = df.loc[df.index <= train_end_date]
    test_df = df.loc[df.index > train_end_date]
    print(f"Train set: {len(train_df)} days, Test set: {len(test_df)} days")
    return train_df, test_df