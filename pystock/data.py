import yfinance as yf
import numpy as np
import pandas as pd


def fetch_data(symbols_filepath: str, start_date:str = "2023-01-01", end_date:str = "2024-01-01"):
    with open(symbols_filepath, "r") as file:
        symbols = file.read().splitlines()
    returns = {}
    returns_data = pd.DataFrame()
    for symbol in symbols:
        try:
            data = yf.Ticker(symbol).history(start=start_date, end=end_date)["Close"]
        except Exception as e:
            raise(f"Error fetching data for {symbol}: {e}")
        daily_returns = np.log(data / data.shift(1)).dropna()  # type: ignore
        expected_return = daily_returns.mean() * 252
        returns[symbol] = {
            "daily_returns": daily_returns,
            "expected_return": expected_return,
        }
        returns_data[symbol] = daily_returns
    cov_matrix = returns_data.cov() * 252
    return returns, cov_matrix.to_numpy()
