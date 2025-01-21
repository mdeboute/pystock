import streamlit as st

from pystock.portfolio import Portfolio


def show_dashboard():
    st.title("📊 Portfolio Dashboard")

    uploaded_file = st.file_uploader("Upload your portfolio file (.xlsx)", type=["xlsx"])
    if uploaded_file:
        portfolio = Portfolio.from_xlsx_file(uploaded_file)  # type: ignore
        st.subheader("Portfolio Overview")
        st.plotly_chart(portfolio.pie_chart, use_container_width=True)
    else:
        st.info("Please upload a portfolio file to get started.")
