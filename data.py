import yfinance as yf
import pandas as pd

def get_price_data(ticker="BTC-USD", interval="1h", period="730d"):
    """
    Download OHLCV price data for any Yahoo Finance ticker.
    Examples:
      BTC-USD, ETH-USD, SOL-USD
    """
    df = yf.download(ticker, interval=interval, period=period, progress=False)
    df = df.dropna().copy()
    return df


