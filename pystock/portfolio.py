import pyomo.environ as pyo
import pandas as pd
import pystock.config as cfg
import numpy as np

from pystock.asset import Asset
from typing import List


class Portfolio:
    def __init__(
        self,
        assets: List[Asset],
        weights: np.ndarray,
    ):
        self.assets = assets
        self.weights = weights
        self._cov_matrix = None
        self._expected_returns = None
        self._check_consistency()

    @classmethod
    def from_xlsx_file(cls, file_path: str):
        df = pd.read_excel(file_path)
        if cfg.SYMBOL_COL not in df.columns or cfg.WEIGHT_COL not in df.columns:
            raise ValueError(
                f"Columns {cfg.SYMBOL_COL} and {cfg.WEIGHT_COL} must be present in the file"
            )
        assets = []
        weights = []
        for _, row in df.iterrows():
            asset = Asset(row[cfg.SYMBOL_COL])
            assets.append(asset)
            weights.append(row[cfg.WEIGHT_COL])
        return cls(assets, np.array(weights))

    def to_xlsx_file(self, file_path: str):
        data = {
            "Symbol": [asset.symbol for asset in self.assets],
            "Weight": self.weights,
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
    def cov_matrix(self):
        if self._cov_matrix is None:
            period_in_days = len(self.assets[0].daily_returns)  # type: ignore
            self._cov_matrix = (
                np.cov(np.array([asset.daily_returns for asset in self.assets]))
                * period_in_days
            )
        return self._cov_matrix

    @property
    def expected_returns(self):
        if self._expected_returns is None:
            self._expected_returns = np.array(
                [asset.expected_return for asset in self.assets]
            )
        return self._expected_returns

    def _build_model(self, desired_return: float) -> pyo.ConcreteModel:
        model = pyo.ConcreteModel()
        model.N = pyo.RangeSet(0, len(self.weights) - 1)
        model.w = pyo.Var(model.N, within=pyo.NonNegativeReals)
        model.objective = pyo.Objective(
            expr=pyo.quicksum(
                pyo.quicksum(
                    self.cov_matrix[i][j] * model.w[i] * model.w[j]  # type: ignore
                    for j in model.N  # type: ignore
                )
                for i in model.N  # type: ignore
            ),
            sense=pyo.minimize,
        )

        def _weight_constraint_rule(model):
            return sum(model.w[i] for i in model.N) == 1

        model.weight_constraint = pyo.Constraint(rule=_weight_constraint_rule)

        def _link_constraint_rule(model, i):
            return (
                sum(model.w[i] * self.expected_returns[i] for i in model.N)
                >= desired_return
            )

        model.link_constraint = pyo.Constraint(model.N, rule=_link_constraint_rule)
        return model

    def optimize(self, desired_return: float, solver_name: str = "knitroampl"):
        """Minimize the variance of the portfolio with a given desired return"""
        self.desired_return = desired_return
        model = self._build_model(desired_return)
        solver = pyo.SolverFactory(solver_name)
        results = solver.solve(model)
        self.weights = np.array([pyo.value(model.w[j]) for j in model.N])  # type: ignore
        return results

    @property
    def variance(self):
        return np.dot(self.weights, np.dot(self.cov_matrix, self.weights))

    @property
    def risk(self):
        return np.sqrt(self.variance)

    @property
    def portfolio_return(self):
        return np.dot(self.weights, self.expected_returns)

    @property
    def sharpe_ratio(self):
        return (self.portfolio_return - cfg.RISK_FREE_RATE) / self.risk

    def add_asset(self, asset: Asset, weight: float):
        self.assets.append(asset)
        self.weights = np.append(self.weights, weight)
        self._reset_cache()

    def remove_asset(self, asset: Asset):
        idx = self.assets.index(asset)
        self.assets.pop(idx)
        self.weights = np.delete(self.weights, idx)
        self._reset_cache()

    def __str__(self) -> str:
        return f"Portfolio({[asset.name for asset in self.assets]})"

    def __repr__(self) -> str:
        return self.__str__()
