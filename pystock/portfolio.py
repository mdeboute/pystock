import pyomo.environ as pyo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


class Portfolio:
    def __init__(
        self,
        names: list[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        weights: np.ndarray,
        capital: float,
    ):
        self.names = names
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.weights = weights
        self.N = len(weights)
        self.capital = capital

    def _build_model(self, desired_return: float) -> pyo.ConcreteModel:
        model = pyo.ConcreteModel()
        model.N = pyo.RangeSet(0, self.N - 1)
        model.w = pyo.Var(model.N, within=pyo.NonNegativeReals)
        model.objective = pyo.Objective(
            expr=pyo.quicksum(
                pyo.quicksum(
                    self.cov_matrix[i][j] * model.w[i] * model.w[j] for j in model.N
                )
                for i in model.N
            ),
            sense=pyo.minimize,
        )

        def weight_constraint_rule(model):
            return sum(model.w[i] for i in model.N) == 1

        model.weight_constraint = pyo.Constraint(rule=weight_constraint_rule)

        def link_constraint_rule(model, i):
            return (
                sum(model.w[i] * self.expected_returns[i] for i in model.N)
                >= desired_return
            )

        model.link_constraint = pyo.Constraint(model.N, rule=link_constraint_rule)
        return model

    def optimize(self, desired_return: float, solver_name: str = "knitroampl"):
        self.desired_return = desired_return
        model = self._build_model(desired_return)
        solver = pyo.SolverFactory(solver_name)
        results = solver.solve(model)
        self.weights = np.array([pyo.value(model.w[j]) for j in model.N])
        return results

    def compute_variance(self):
        return sum(
            sum(
                self.cov_matrix[i][j] * self.weights[i] * self.weights[j]
                for j in range(self.N)
            )
            for i in range(self.N)
        )

    def compute_risk(self):
        return np.sqrt(np.dot(self.weights, np.dot(self.cov_matrix, self.weights)))

    def compute_return(self):
        return np.dot(self.weights, self.expected_returns)

    def display_portfolio(self):
        selected_assets = [
            asset for asset, weight in zip(self.names, self.weights) if weight > 0
        ]
        portfolio_weights = [weight for weight in self.weights if weight > 0]

        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.5, 0.5],
            subplot_titles=("Bar chart", "Pie chart"),
            specs=[[{"type": "xy"}, {"type": "domain"}]],
        )

        # Bar chart
        fig.add_trace(
            go.Bar(
                x=selected_assets,
                y=portfolio_weights,
                text=[f"{weight:.2f}" for weight in portfolio_weights],
                textposition="auto",
                opacity=0.8,
                name="Bar Chart",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=selected_assets,
                values=portfolio_weights,
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                opacity=0.8,
                name="Pie Chart",
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            title=f"Return of {self.compute_return():.2f} for a variance of {self.compute_variance():.2f} with an associated risk of {self.compute_risk():.2f}",
            xaxis_title="Assets",
            yaxis_title="Weight",
            height=600,
            width=1000,
        )

        fig.show()

    def plot_capital_evo(self):
        # Compute the evolution of the capital over time with this portfolio
        capital_evo = [self.capital]
        for _ in range(252):
            daily_return = np.dot(self.weights, np.random.multivariate_normal(self.expected_returns, self.cov_matrix))
            self.capital *= (1 + daily_return)
            capital_evo.append(self.capital)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(253)), y=capital_evo, mode="lines", name="Capital evolution"))
        fig.update_layout(title="Capital evolution over time", xaxis_title="Days", yaxis_title="Capital")
        fig.show()
