import data
import portfolio
import numpy as np

data, cov_matrix = data.fetch_data(r"D:\perso\pystock\data\stocks.txt")

names = list(data.keys())
expected_returns = np.array([data[symbol]["expected_return"] for symbol in names])

portfolio = portfolio.Portfolio(names=data.keys(), expected_returns=expected_returns, cov_matrix=cov_matrix, weights=np.array([1/len(names)]*len(names)), capital=1000)

