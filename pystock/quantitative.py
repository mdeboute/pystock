import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyomo.environ as pyo
from joblib import Parallel, delayed

import pystock.constants as cst
from pystock.portfolio import Portfolio


class PortfolioOptimizer:
    """An optimizer for your Portfolio."""

    def __init__(self, portfolio: Portfolio) -> None:
        """Initialize the PortfolioOptimizer.

        Args:
            portfolio (Portfolio): The portfolio that you want to optimize.

        """
        self.portfolio = portfolio
        self.model = self._build_core_model()
        self.solver = pyo.SolverFactory(cst.DEFAULT_SOLVER)

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
        risk_free_rate: float = 0.01,
    ) -> tuple[float, float, float, list[float]]:
        portfolio_return = np.dot(weights, self.portfolio.historical_expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(self.portfolio.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
        return portfolio_return, portfolio_risk, sharpe_ratio, weights.tolist()

    def simulation(
        self,
        num_simulations: int = 5000,
        risk_free_rate: float = cst.DEFAULT_RISK_FREE_RATE,
        num_jobs: int = -1,
    ) -> pd.DataFrame:
        """Run a Monte Carlo simulation for portfolio optimization.

        Args:
            num_simulations (int): Number of simulations to run.
            risk_free_rate (float): Risk-free rate.
            num_jobs (int): Number of parallel jobs to use (-1 for all available CPUs).

        Returns:
            pd.DataFrame: Results of the simulation (returns, risk, Sharpe ratio, weights).
        """
        if num_simulations <= 0:
            raise ValueError("Number of simulations must be a positive integer.")

        n_assets = len(self.portfolio.historical_expected_returns)
        simulations = np.random.dirichlet(np.ones(n_assets), size=num_simulations)  # noqa: NPY002

        results = Parallel(n_jobs=num_jobs, backend="threading")(
            delayed(self._calculate_metrics)(weights, risk_free_rate) for weights in simulations
        )

        return pd.DataFrame(
            results,
            columns=["Returns", "Risk", "Sharpe Ratio", "Weights"],  # type: ignore
        )

    @staticmethod
    def get_gareto_front(df: pd.DataFrame) -> pd.DataFrame:
        """Identify the Pareto front from the simulation results.

        Args:
            df (pd.DataFrame): Simulation results.

        Returns:
            pd.DataFrame: DataFrame containing Pareto-optimal portfolios.
        """
        return df.sort_values("Risk").drop_duplicates("Returns", keep="first")

    @staticmethod
    def create_efficient_frontier(df: pd.DataFrame) -> go.Figure:
        """Plot the Pareto front using Plotly.

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
