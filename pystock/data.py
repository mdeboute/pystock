import yfinance as yf
import numpy as np
import pandas as pd


def fetch_data(ISIN_symbols_filepath: str):
    with open(ISIN_symbols_filepath, "r") as file:
        ISIN_symbols = file.read().splitlines()
    returns = {}
    returns_data = pd.DataFrame()
    for isin in ISIN_symbols:
        data = yf.Ticker(isin).history(period="max")["Close"]
        daily_returns = np.log(data / data.shift(1)).dropna()  # type: ignore
        expected_return = daily_returns.mean() * 252
        returns[isin] = {
            "daily_returns": daily_returns,
            "expected_return": expected_return,
        }
        returns_data[isin] = daily_returns
    cov_matrix = returns_data.cov() * 252
    return returns, cov_matrix.to_numpy()
