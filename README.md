# pystock

A tool that helps you better invest in the stock market.

## Features

- **Portfolio Management**: Create and manage your investment portfolio.
- **Portfolio Optimization**: Optimize your portfolio to achieve desired returns with minimal risk.
- **Monte Carlo Simulation**: Run simulations to predict future performance of your portfolio.
- **Dashboard**: Visualize your portfolio's performance with interactive charts.

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/pystock.git
    cd pystock
    ```

2. Install the dependencies:

    ```sh
    uv sync
    ```

## Usage

1. Run the Streamlit app:

    ```sh
    uv run streamlit run app/main.py
    ```

2. Navigate through the sidebar to access different features:
    - **Dashboard**: Upload your portfolio file to view its performance.
    - **Monte Carlo Simulation**: Run simulations to predict future performance.
    - **Portfolio Optimization**: Optimize your portfolio to achieve desired returns.

## Project Structure

- [app](./app/) Contains the Streamlit app and views.
- [pystock](./pystock/): Contains the core logic for portfolio management and optimization.
- [tests](./tests/): Contains unit tests for the project.
- [pyproject.toml](./pyproject.toml): Project configuration and dependencies.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License.
