import streamlit as st

from app.views.dashboard import show_dashboard
from app.views.optimization import show_portfolio_optimization
from app.views.simulator import show_monte_carlo_simulation


def main():
    st.sidebar.title("Navigation")
    view = st.sidebar.radio("Go to", ["Dashboard", "Monte Carlo Simulation", "Portfolio Optimization"])

    if view == "Dashboard":
        show_dashboard()
    elif view == "Monte Carlo Simulation":
        show_monte_carlo_simulation()
    elif view == "Portfolio Optimization":
        show_portfolio_optimization()


if __name__ == "__main__":
    main()
