import streamlit as st
from pystock.portfolio import Portfolio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
import pystock.config as cfg


def create_pie_chart(portfolio: Portfolio) -> go.Figure:
    symbols = [asset.symbol for asset in portfolio.assets]
    selected_assets = [
        asset for asset, weight in zip(symbols, portfolio.weights) if weight > 0.01
    ]
    portfolio_weights = [weight for weight in portfolio.weights if weight > 0.01]

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
        title=f"Return of {portfolio.portfolio_return:.2f} with a sharpe ratio of {portfolio.sharpe_ratio:.2f} and a risk of {portfolio.risk:.2f}",
        height=600,
        width=1000,
    )
    return fig


def fetch_sp500_closes() -> np.ndarray:
    return yf.download(cfg.SP500_SYMBOL, period=cfg.PERIOD, interval=cfg.INTERVAL)[
        "Close"
    ].values


def calculate_portfolio_value_over_time(
    portfolio: Portfolio, capital: float, cashflow: float
) -> pd.DataFrame:
    closes = np.array([asset.closes for asset in portfolio.assets])
    weights = portfolio.weights
    n_assets, n_days = closes.shape

    # Fetch S&P 500 closing prices
    sp500_closes = fetch_sp500_closes()

    # Ensure we don't exceed the available S&P 500 data length
    n_sp500_days = len(sp500_closes)
    n_days = min(n_days, n_sp500_days)  # Limit to the available data

    portfolio_value = np.zeros(n_days)
    sp500_value = np.zeros(n_days)

    # Calculate initial investment per asset
    initial_investment = np.array([capital * weight for weight in weights])
    initial_prices = closes[:, 0]
    quantities = initial_investment / initial_prices  # Initial quantities of each asset

    # Initial S&P 500 investment
    sp500_initial_price = sp500_closes[0]
    sp500_quantity = capital / sp500_initial_price  # Initial quantity of S&P 500 shares

    for day in range(n_days):
        if day % cfg.MONTHLY_CONTRIBUTION_INTERVAL == 0 and day != 0:
            # Add monthly cashflow to each asset according to weights
            monthly_investment = np.array([cashflow * weight for weight in weights])
            new_quantities = monthly_investment / closes[:, day]
            quantities += new_quantities

            # Update S&P 500 quantity
            sp500_quantity += cashflow / sp500_closes[day]

        # Calculate portfolio and S&P 500 values for the day
        portfolio_value[day] = np.sum(
            closes[:, day] * quantities
        ).item()  # Ensure scalar
        sp500_value[day] = (sp500_quantity * sp500_closes[day]).item()  # Ensure scalar

    return pd.DataFrame({"Portfolio": portfolio_value, "S&P 500": sp500_value})


def plot_portfolio_vs_sp500(portfolio_values: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=portfolio_values.index,
            y=portfolio_values["Portfolio"],
            mode="lines",
            name="Total Portfolio Value",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=portfolio_values.index,
            y=portfolio_values["S&P 500"],
            mode="lines",
            name="S&P 500",
        )
    )

    fig.update_layout(
        title="Portfolio Value vs. S&P 500 Over Time with Monthly Contributions",
        xaxis_title="Days",
        yaxis_title="Value (€)",
        template="plotly_white",
    )
    return fig


def dashboard():
    st.title("My Portfolio Dashboard")

    st.sidebar.title("Settings")
    capital = st.sidebar.number_input("Capital", value=2200)
    cashflow = st.sidebar.number_input("Cashflow", value=200)

    st.sidebar.write("Upload your portfolio file")
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        portfolio = Portfolio.from_xlsx_file(uploaded_file)  # type: ignore
        fig_1 = create_pie_chart(portfolio)
        st.plotly_chart(fig_1)

        desired_return = st.sidebar.number_input(
            "Desired Return", min_value=0.0, value=0.1, step=0.01
        )
        if st.sidebar.button("Optimize Portfolio"):
            portfolio.optimize(desired_return)
            st.header("Optimized Portfolio:")
            fig_2 = create_pie_chart(portfolio)
            st.plotly_chart(fig_2)

        portfolio_values = calculate_portfolio_value_over_time(
            portfolio, capital, cashflow
        )
        fig_3 = plot_portfolio_vs_sp500(portfolio_values)
        st.plotly_chart(fig_3)


def main():
    dashboard()


if __name__ == "__main__":
    main()
