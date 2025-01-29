import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.covariance import OAS, LedoitWolf

import pystock.constants as cst
from pystock.asset import Asset


class Portfolio:
    """Portfolio object."""

    def __init__(self, assets: list[Asset], weights: list[float] | np.ndarray) -> None:
        """Portfolio object.

        Args:
            assets (list[Asset]): The list of assets composing the portfolio.
            weights (list[float] | np.ndarray): The proportion of each asset in the portfolio.

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

    def to_xlsx_file(self, file_path: str) -> None:
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

    def _check_consistency(self) -> None:
        if len(self.assets) != len(self.weights):
            raise ValueError("The number of assets and weights must be the same!")
        if not np.isclose(np.sum(self.weights), 1):
            raise ValueError("The sum of the weights must be equal to 1!")

    def _reset_cache(self) -> None:
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
    def historical_expected_returns(self) -> np.ndarray:
        """Compute the expected returns of the portfolio from historical data.

        Returns:
            np.ndarray: The expected returns.

        """
        if self._expected_returns is None:
            self._expected_returns = np.array([asset.expected_return for asset in self.assets])
        return self._expected_returns

    @property
    def historical_returns(self) -> np.ndarray:
        """Compute the returns of the portfolio from historical data.

        Returns:
            np.ndarray: The returns.

        """
        return np.array([asset.daily_returns[: self.period_in_days] for asset in self.assets])

    @property
    def variance(self) -> float:
        """Compute the variance of the portfolio.

        Returns:
            float: The variance of the portfolio.

        """
        return np.dot(self.weights, np.dot(self.cov_matrix, self.weights))

    @property
    def risk(self) -> float:
        """Compute the risk of the portfolio as the standard deviation of the returns.

        Returns:
            float: The risk of the portfolio.

        """
        return np.sqrt(self.variance)

    @property
    def historical_expected_return(self) -> float:
        """Compute the expected return of the portfolio based on the historical expected returns of the assets.

        Returns:
            float: The return of the portfolio.

        """
        return np.dot(self.weights, self.historical_expected_returns)

    @property
    def sharpe_ratio(self) -> float:
        """Compute the Sharpe ratio of the portfolio.

        Returns:
            float: The Sharpe ratio of the portfolio.

        """
        return (self.historical_expected_return - cst.DEFAULT_RISK_FREE_RATE) / self.risk

    def add_asset(self, asset: Asset, weight: float) -> None:
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

    def remove_asset(self, asset: Asset) -> None:
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

    @property
    def pie_chart(self) -> go.Figure:
        """Create a pie chart of the portfolio.

        Returns:
            go.Figure: The pie chart of the portfolio.

        """
        symbols = [asset.symbol for asset in self.assets]
        selected_assets = [asset for asset, weight in zip(symbols, self.weights) if weight > 0.01]  # noqa: PLR2004
        portfolio_weights = [weight for weight in self.weights if weight > 0.01]  # noqa: PLR2004

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
            title=(
                f"Return of {self.historical_expected_return:.2f} with a sharpe ratio of "
                f"{self.sharpe_ratio:.2f} and a risk of {self.risk:.2f} over {self.period_in_days} days"
            ),
            height=600,
            width=1000,
        )
        return fig

    def shrink_covariance(self, method: str = "ledoit-wolf", **kwargs: dict) -> np.ndarray:
        """Shrinks the covariance matrix of the portfolio returns to improve its condition.

        Shrinkage is especially useful when the number of observations (T) is smaller
        or comparable to the number of assets (N), as the sample covariance matrix
        may be poorly conditioned or singular.

        Args:
            method (str, optional): The shrinkage method to use. Options are:
                - "ledoit-wolf": Bayesian shrinkage towards a diagonal matrix.
                  Automatically calculates the shrinkage intensity.
                - "oas": Oracle Approximating Shrinkage, a refined version of Ledoit-Wolf
                  that minimizes the Frobenius norm of the difference between the true
                  covariance matrix and the shrunk matrix.
                - "ridge": Adds a constant value (lambda) to the diagonal of the covariance matrix.
                  Requires `lambda` to be provided in `kwargs`.
            **kwargs: Additional parameters for specific shrinkage methods:
                - For "ridge": `lambda` (float) specifies the shrinkage intensity (default: 0.1).

        Returns:
            numpy.ndarray: The shrunk covariance matrix.

        Raises:
            ValueError: If an unsupported shrinkage method is specified or required parameters are missing.

        Notes:
            Shrinkage improves the stability of the covariance matrix by introducing a bias
            towards a more stable target (e.g., a diagonal matrix). It is particularly beneficial
            when the number of observations is small compared to the number of assets.

            Method details:
            - Ledoit-Wolf: Automatically determines the optimal shrinkage intensity based on the data,
              making it a robust choice for poorly conditioned matrices.
            - OAS: Offers better performance than Ledoit-Wolf when the assumption of multivariate normality
              holds, as it minimizes the Frobenius norm between the estimated and true covariance matrices.
            - Ridge: Provides manual control over the shrinkage intensity. Useful for testing or scenarios
              where specific domain knowledge informs the choice of lambda, though its effectiveness depends
              on careful tuning of the parameter.

        """
        if not hasattr(self, "returns"):
            raise AttributeError("Portfolio must have 'returns' attribute with historical data.")

        returns = self.historical_returns

        if method == "ledoit-wolf":
            lw = LedoitWolf()
            lw.fit(returns)
            self._cov_matrix = lw.covariance_

        elif method == "oas":
            oas = OAS()
            oas.fit(returns)
            self._cov_matrix = oas.covariance_

        elif method == "ridge":
            lambda_ = kwargs.get("lambda", 0.1)
            if not isinstance(lambda_, int | float) or lambda_ < 0:
                raise ValueError("For 'ridge' method, 'lambda' must be a non-negative float.")

            empirical_cov = np.cov(returns, rowvar=False)
            identity = np.eye(empirical_cov.shape[0])
            self._cov_matrix = (1 - lambda_) * empirical_cov + lambda_ * identity

        else:
            raise ValueError(f"Unsupported shrinkage method: {method}. Choose from 'ledoit-wolf', 'oas', or 'ridge'.")

        return self._cov_matrix

    def compute_betas(self, market_returns: np.ndarray) -> np.ndarray:
        """Compute the beta of each asset in the portfolio.

        Beta measures an asset's sensitivity to market returns and is computed as:

            Beta = Cov(R_asset, R_market) / Var(R_market)

        Returns:
            numpy.ndarray: A 1D array of beta values for each asset.

        Raises:
            ValueError: If the length of market returns does not match the length of asset returns.

        Notes:
            - A beta of 1 means the asset moves in line with the market.
            - A beta > 1 means higher volatility than the market.
            - A beta < 1 means lower volatility than the market.
            - Negative beta values indicate an inverse relationship with the market.

        """
        returns = self.historical_returns
        if len(market_returns) != len(returns):
            raise ValueError("Length of market returns must match the length of portfolio returns.")

        excess_market_returns = market_returns - cst.DEFAULT_RISK_FREE_RATE
        excess_assets_returns = returns - cst.DEFAULT_RISK_FREE_RATE

        return np.array(
            [
                np.cov(asset_returns, excess_market_returns, bias=True)[0, 1] / np.var(excess_market_returns, ddof=0)
                for asset_returns in excess_assets_returns.T
            ],
        )

    def compute_capm_expected_returns(self, market_returns: np.ndarray) -> np.ndarray:
        """Calculate the expected returns of assets in the portfolio using the CAPM.

        This method estimates the expected returns based on the Capital Asset Pricing Model (CAPM):

            E(R) = R_f + β * (E(R_m) - R_f)

        where:
        - `E(R)` is the expected return of an asset.
        - `R_f` is the risk-free rate.
        - `β` is the beta of the asset.
        - `E(R_m)` is the expected return of the market.

        Args:
            market_returns (numpy.ndarray): A 1D array of market returns.

        Returns:
            numpy.ndarray: A 1D array of expected returns for each asset in the portfolio.

        Raises:
            ValueError: If the length of `market_returns` does not match the length of portfolio returns.

        Notes:
            - The CAPM assumes a linear relationship between an asset's returns and market returns.
            - It relies on the assumption that beta remains stable over time.
            - This method calculates beta dynamically based on historical data.

        """
        return cst.DEFAULT_RISK_FREE_RATE + self.compute_betas(market_returns) * np.mean(
            market_returns - cst.DEFAULT_RISK_FREE_RATE,
        )

    def __str__(self) -> str:
        return f"Portfolio({[asset.name for asset in self.assets]})"

    def __repr__(self) -> str:
        return self.__str__()
