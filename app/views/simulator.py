import streamlit as st

from pystock.portfolio import Portfolio
from pystock.quantitative import MonteCarloSimulator


def show_monte_carlo_simulation():
    st.title("🎲 Monte Carlo Simulation")

    uploaded_file = st.file_uploader("Upload your portfolio file (.xlsx)", type=["xlsx"])
    if uploaded_file:
        portfolio = Portfolio.from_xlsx_file(uploaded_file)  # type: ignore
        simulator = MonteCarloSimulator(portfolio)

        num_simulations = st.number_input("Number of Simulations", min_value=1000, max_value=50000, value=5000)

        if st.button("Run Simulation"):
            results = simulator.simulation(num_simulations=num_simulations)
            st.plotly_chart(simulator.create_efficient_frontier(results), use_container_width=True)
    else:
        st.info("Please upload a portfolio file to begin.")
