import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyomo.environ as pyo
from joblib import Parallel, delayed
from scipy.linalg import inv

import pystock.constants as cst
from pystock.portfolio import Portfolio


class PortfolioOptimizer:
    """An optimizer for your Portfolio."""

    def __init__(self, portfolio: Portfolio, solver: str = cst.DEFAULT_SOLVER) -> None:
        """Initialize the PortfolioOptimizer.

        Args:
            portfolio (Portfolio): The portfolio that you want to optimize.
            solver (str): The nonlinear solver to use for optimization.

        """
        self.portfolio = portfolio
        self.model = self._build_core_model()
        self.solver = pyo.SolverFactory(solver)

    def _build_core_model(self) -> pyo.ConcreteModel:
        model = pyo.ConcreteModel()
        model.N = pyo.RangeSet(0, len(self.portfolio.weights) - 1)
        model.w = pyo.Var(model.N, within=pyo.NonNegativeReals)

        def _weight_constraint_rule(model: pyo.ConcreteModel) -> pyo.Constraint:
            return sum(model.w[i] for i in model.N) == 1

        model.weight_constraint = pyo.Constraint(rule=_weight_constraint_rule)

        return model

    def minimize_variance_with_return(self, desired_return: float) -> np.ndarray:
        """Minimize portfolio variance with a given return.

        Args:
            desired_return (float): Desired return.

        Returns:
            np.ndarray: Optimal portfolio weights.

        """
        self.model.objective = pyo.Objective(
            expr=pyo.quicksum(
                pyo.quicksum(
                    self.portfolio.cov_matrix[i][j] * self.model.w[i] * self.model.w[j]  # type: ignore
                    for j in self.model.N  # type: ignore
                )
                for i in self.model.N  # type: ignore
            ),
            sense=pyo.minimize,
        )

        def _link_constraint_rule(model: pyo.ConcreteModel) -> pyo.Constraint:
            return sum(model.w[i] * self.portfolio.historical_expected_returns[i] for i in model.N) >= desired_return

        self.model.link_constraint = pyo.Constraint(self.model.N, rule=_link_constraint_rule)
        self.solver.solve(self.model)
        return np.array([pyo.value(self.model.w[j]) for j in self.model.N])  # type: ignore


class MonteCarloSimulator:
    """A Monte Carlo Simulator for your portfolio."""

    def __init__(self, portfolio: Portfolio) -> None:
        """Initialize the MonteCarloSimulator.

        Args:
            portfolio (Portfolio): The portfolio you want to work on.

        """
        self.portfolio = portfolio

    def _calculate_metrics(
        self,
        weights: np.ndarray,
    ) -> tuple[float, float, float, list[float]]:
        self.portfolio.weights = weights
        return (
            self.portfolio.historical_expected_return,
            self.portfolio.risk,
            self.portfolio.sharpe_ratio,
            weights.tolist(),
        )

    def simulation(
        self,
        num_simulations: int = 5000,
        num_jobs: int = -1,
    ) -> pd.DataFrame:
        """Run a Monte Carlo simulation for portfolio optimization.

        Args:
            num_simulations (int): Number of simulations to run.
            num_jobs (int): Number of parallel jobs to use (-1 for all available CPUs).

        Returns:
            pd.DataFrame: Results of the simulation (returns, risk, Sharpe ratio, weights).

        """
        if num_simulations <= 0:
            raise ValueError("Number of simulations must be a positive integer.")

        n_assets = len(self.portfolio.historical_expected_returns)
        simulations = np.random.dirichlet(np.ones(n_assets), size=num_simulations)  # noqa: NPY002

        results = Parallel(n_jobs=num_jobs, backend="threading")(
            delayed(self._calculate_metrics)(weights) for weights in simulations
        )

        return pd.DataFrame(
            results,
            columns=["Returns", "Risk", "Sharpe Ratio", "Weights"],  # type: ignore
        )

    @staticmethod
    def create_efficient_frontier(df: pd.DataFrame) -> go.Figure:
        """Plot the Pareto front using Plotly with an optimal portfolio marker.

        Args:
            df (pd.DataFrame): DataFrame of simulation results.

        Returns:
            px.scatter: Plotly scatter plot.

        """
        df["Weights"] = df["Weights"].apply(lambda x: str(x))

        fig = px.scatter(
            df,
            x="Risk",
            y="Returns",
            color="Sharpe Ratio",
            color_continuous_scale="Viridis",
            title="Monte Carlo Simulation: Portfolio Risk vs. Return",
            labels={"Risk": "Risk (Standard Deviation)", "Returns": "Return"},
        )

        fig.update_layout(
            xaxis={"title": "Risk (Standard Deviation)"},
            yaxis={"title": "Return"},
            coloraxis_colorbar={"title": "Sharpe Ratio"},
        )
        return fig


class BlackLitterman:
    """Black-Litterman model for portfolio optimization.

    This model combines market equilibrium returns with investor views to generate
    a set of adjusted expected returns for portfolio optimization.
    """

    def __init__(self, portfolio: Portfolio, tau: float = 0.05) -> None:
        """Initialize the Black-Litterman model.

        Args:
            portfolio (Portfolio): The portfolio object containing asset returns and covariance matrix.
            tau (float, optional): Scaling factor for the market equilibrium returns. Defaults to 0.05.

        """
        self.portfolio = portfolio
        self.tau = tau
        self.market_implied_returns = self.compute_market_implied_returns()

    def compute_market_implied_returns(self) -> np.ndarray:
        """Compute the market-implied returns using the reverse optimization approach.

        Returns:
            numpy.ndarray: A 1D array of market-implied returns for each asset in the portfolio.

        """
        weights = self.portfolio.weights  # market capitalization weights
        cov_matrix = self.portfolio.cov_matrix

        return self.tau * cov_matrix @ weights

    def adjust_returns_with_views(self, P: np.ndarray, Q: np.ndarray, omega: np.ndarray | None = None) -> np.ndarray:
        """Adjust market-implied returns using investor views.

        Args:
            P (numpy.ndarray): A matrix linking views to assets (k x n).
            Q (numpy.ndarray): A vector of expected returns based on views (k x 1).
            omega (numpy.ndarray, optional): A diagonal covariance matrix for the views. If None,
                                            it is set to tau * P @ cov_matrix @ P.T.

        Returns:
            numpy.ndarray: The adjusted expected returns incorporating investor views.

        """
        cov_matrix = self.portfolio.cov_matrix
        pi = self.market_implied_returns

        if omega is None:
            omega = self.tau * P @ cov_matrix @ P.T

        # Black-Litterman formula
        M1 = inv(inv(self.tau * cov_matrix) + P.T @ inv(omega) @ P)
        M2 = inv(self.tau * cov_matrix) @ pi + P.T @ inv(omega) @ Q

        return M1 @ M2
