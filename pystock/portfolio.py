from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pystock.constants as cst
from pystock.asset import Asset


class Portfolio:
    def __init__(self, assets: List[Asset], weights: List[float] | np.ndarray):
        """Portfolio object.

        Args:
            assets (List[Asset]): The list of assets composing the portfolio.
            weights (List[float] | np.ndarray): The proportion of each asset in the portfolio.
        """
        self.assets = assets
        self.weights = weights
        self._cov_matrix = None
        self._expected_returns = None
        self.period_in_days = min([len(asset.daily_returns) for asset in self.assets])
        self._check_consistency()

    @classmethod
    def from_xlsx_file(cls, file_path: str) -> "Portfolio":
        """Load a portfolio from an XLSX file.

        Args:
            file_path (str): Path to the XLSX file.

        Raises:
            ValueError: If the columns `SYMBOL_COL` and `WEIGHT_COL` are not present in the file.
        """
        df = pd.read_excel(file_path)
        if cst.SYMBOL_COL not in df.columns or cst.WEIGHT_COL not in df.columns:
            raise ValueError(f"Columns {cst.SYMBOL_COL} and {cst.WEIGHT_COL} must be present in the file")
        assets = []
        weights = []
        for _, row in df.iterrows():
            asset = Asset(row[cst.SYMBOL_COL])
            assets.append(asset)
            weights.append(row[cst.WEIGHT_COL])
        return cls(assets, np.array(weights))

    def to_xlsx_file(self, file_path: str):
        """Save the portfolio to an XLSX file.

        Args:
            file_path (str): Path to the XLSX file.
        """
        data = {
            cst.SYMBOL_COL: [asset.symbol for asset in self.assets],
            cst.WEIGHT_COL: self.weights,
        }
        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)

    def _check_consistency(self):
        if len(self.assets) != len(self.weights):
            raise ValueError("The number of assets and weights must be the same!")
        if not np.isclose(np.sum(self.weights), 1):
            raise ValueError("The sum of the weights must be equal to 1!")

    def _reset_cache(self):
        self._cov_matrix = None
        self._expected_returns = None

    @property
    def cov_matrix(self) -> np.ndarray:
        """Compute the covariance matrix of the portfolio.

        Returns:
            np.ndarray: The covariance matrix.
        """
        if self._cov_matrix is None:
            daily_returns = [
                asset.daily_returns[: self.period_in_days]
                for asset in self.assets  # type: ignore
            ]
            self._cov_matrix = np.cov(np.array(daily_returns)) * self.period_in_days
        return self._cov_matrix

    @property
    def expected_returns(self) -> np.ndarray:
        """Compute the expected returns of the portfolio.

        Returns:
            np.ndarray: The expected returns.
        """
        if self._expected_returns is None:
            self._expected_returns = np.array([asset.expected_return for asset in self.assets])
        return self._expected_returns

    @property
    def variance(self) -> float:
        """Compute the variance of the portfolio.

        Returns:
            float: The variance of the portfolio.
        """
        return np.dot(self.weights, np.dot(self.cov_matrix, self.weights))

    @property
    def risk(self) -> float:
        """Compute the risk of the portfolio.

        Returns:
            float: The risk of the portfolio.
        """
        return np.sqrt(self.variance)

    @property
    def portfolio_return(self) -> float:
        """Compute the return of the portfolio.

        Returns:
            float: The return of the portfolio.
        """
        return np.dot(self.weights, self.expected_returns)

    @property
    def sharpe_ratio(self) -> float:
        """Compute the Sharpe ratio of the portfolio.

        Returns:
            float: The Sharpe ratio of the portfolio.
        """
        return (self.portfolio_return - cst.DEFAULT_RISK_FREE_RATE) / self.risk

    def add_asset(self, asset: Asset, weight: float):
        """Add an asset to the portfolio.

        Args:
            asset (Asset): The asset to add.
            weight (float): The weight of the asset in the portfolio.
        """
        self.assets.append(asset)
        self.weights = np.append(self.weights, weight)
        self._reset_cache()
        if not np.isclose(np.sum(self.weights), 1):
            print("Warning: The sum of the weights is not equal to 1!")

    def remove_asset(self, asset: Asset):
        """Remove an asset from the portfolio.

        Args:
            asset (Asset): The asset to remove.

        Raises:
            ValueError: If the asset is not in the portfolio.
        """
        if asset not in self.assets:
            raise ValueError("The asset is not in the portfolio.")
        idx = self.assets.index(asset)
        self.assets.pop(idx)
        self.weights = np.delete(self.weights, idx)
        self._reset_cache()
        if not np.isclose(np.sum(self.weights), 1):
            print("Warning: The sum of the weights is not equal to 1!")

    def __str__(self) -> str:
        return f"Portfolio({[asset.name for asset in self.assets]})"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def pie_chart(self) -> go.Figure:
        """Create a pie chart of the portfolio.

        Returns:
            go.Figure: The pie chart of the portfolio.
        """
        symbols = [asset.symbol for asset in self.assets]
        selected_assets = [asset for asset, weight in zip(symbols, self.weights) if weight > 0.01]
        portfolio_weights = [weight for weight in self.weights if weight > 0.01]

        fig = make_subplots(rows=1, cols=1, specs=[[{"type": "domain"}]])

        pie_trace = go.Pie(
            labels=selected_assets,
            values=portfolio_weights,
            textinfo="label+percent",
            hoverinfo="label+value+percent",
            opacity=0.8,
        )
        fig.add_trace(pie_trace, row=1, col=1)

        fig.update_layout(
            title=f"Return of {self.portfolio_return:.2f} with a sharpe ratio of {self.sharpe_ratio:.2f} and a risk of {self.risk:.2f} over {self.period_in_days} days",
            height=600,
            width=1000,
        )
        return fig
