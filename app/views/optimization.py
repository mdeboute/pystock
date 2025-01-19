import streamlit as st

from pystock.portfolio import Portfolio
from pystock.quantitative import PortfolioOptimizer


def show_portfolio_optimization():
    st.title("🔧 Portfolio Optimization")

    uploaded_file = st.file_uploader(
        "Upload your portfolio file (.xlsx)", type=["xlsx"]
    )
    if uploaded_file:
        portfolio = Portfolio.from_xlsx_file(uploaded_file)  # type: ignore

        desired_return = st.number_input(
            "Desired Return", min_value=0.0, value=0.1, step=0.01
        )
        if st.button("Optimize Portfolio"):
            optimizer = PortfolioOptimizer(portfolio)
            new_weights = optimizer.minimize_variance_with_return(desired_return)
            portfolio.weights = new_weights
            st.plotly_chart(portfolio.pie_chart, use_container_width=True)
    else:
        st.info("Please upload a portfolio file to begin.")
