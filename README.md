# pystock

A tool that helps you better invest in the stock market.

## Features

- **Portfolio Management**: Create and manage your investment portfolio.
- **Portfolio Optimization**: Optimize your portfolio to achieve desired returns with minimal risk.
- **Monte Carlo Simulation**: Run simulations to predict future performance of your portfolio.
- **Dashboard**: Visualize your portfolio's performance with interactive charts.

## Installation

 ```bash
pip install pystock
```

## Usage

1. Run the Streamlit app:

    ```bash
    streamlit run app/app.py
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

Please follow the [CONTRIBUTING.md](./CONTRIBUTING.md) guidelines.

### Packaging and tools

[uv](https://docs.astral.sh/uv/) is used to manage Python version, library dependencies, packaging and publishing.

To install the needed dependencies:

```bash
uv sync
```

If you add or update some dependencies, you must push the `uv.lock`.

### Formatting and linting

[Ruff](https://docs.astral.sh/ruff/) is used to ensure isort and flake8 standards with some linting functionnalities.

You can format using:

```bash
ruff format pystock/
```

For linting, you can check for errors with (adding option ```--fix``` will update your files when ruff is able to do so):

```bash
ruff check pystock/
```

Configuration can be found in `pyproject.toml`.

### Code coverage

Code coverage is done with

```bash
pytest --cov=.
```

### Building documentation

Documentation can be built using sphinx and by installing required dependencies.

```bash
uv sync --extra dev
python -m sphinx docs/source docs/build/
```

## License

This project is licensed under the [MIT License](./LICENSE).
