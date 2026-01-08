import pandas as pd
import numpy as np

def compute_features(df):
    """
    Compute features for ML model: returns, volatility, trend slope.
    """
    df = df.copy()
    
    # 1. Returns (percent change)
    df['return'] = df['Close'].pct_change()
    
    # 2. Volatility (rolling standard deviation of returns)
    df['volatility'] = df['return'].rolling(window=24).std()
    
    # 3. Trend slope (difference between short and long moving averages)
    df['ma_12'] = df['Close'].rolling(window=12).mean()
    df['ma_24'] = df['Close'].rolling(window=24).mean()
    df['trend'] = df['ma_12'] - df['ma_24']
    
    df.dropna(inplace=True)  # remove rows with missing values
    return df

# Test code
if __name__ == "__main__":
    import data
    df = data.get_btc_data()
    df = compute_features(df)
    print(df.head())
