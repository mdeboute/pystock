import numpy as np
import pandas as pd
import yfinance as yf

import pystock.constants as cst


class Asset:
    def __init__(self, symbol: str, listed: bool = True):
        """Asset object.

        Args:
            symbol (str): Bloomberg ticker of the asset.
            listed (bool, optional): If the asset is listed on Yahoo Finance or not. Default to True.
        """
        self.symbol = symbol
        if listed:
            self.ticker = yf.Ticker(self.symbol)
        else:
            self.ticker = None
        self.listed = listed
        self._historic_data = None
        self._closes = None
        self._daily_returns = None
        self._daily_log_returns = None
        self._expected_log_return = None
        self._expected_return = None
        self._period_in_day = None

    @property
    def name(self) -> str:
        """Return the name of the asset.

        Returns:
            str: The name of the asset.
        """
        return self.ticker.info[cst.SHORT_NAME] if self.listed else self.symbol  # type: ignore

    def fetch_historical_data(
        self, period: str = cst.DEFAULT_PERIOD, interval: str = cst.DEFAULT_INTERVAL
    ) -> pd.DataFrame | None:
        """Fetch historical data for the asset.

        Args:
            period (str, optional): The period of the data. Defaults to cst.DEFAULT_PERIOD.
            interval (str, optional): The interval of the data. Defaults to cst.DEFAULT_INTERVAL.

        Raises:
            Exception: Error while fetching data for the asset.

        Returns:
            pd.DataFrame | None: The historical data of the asset.
        """
        if not self.listed:
            print(f"{self.name} is not listed! You should provide the data.")
            return None
        if self._historic_data is None:
            try:
                self._historic_data = self.ticker.history(  # type: ignore
                    period=period, interval=interval
                )
                self._period_in_day = self._historic_data.shape[0]
            except Exception as e:
                raise Exception(f"Error while fetching data for {self.name}: {e}") from e
        return self._historic_data

    @property
    def historical_data(self) -> pd.DataFrame | None:
        """Return the historical data of the asset.

        Returns:
            pd.DataFrame | None: The historical data of the asset.
        """
        return self.fetch_historical_data()

    @property
    def closes(self) -> pd.Series | None:
        """Return the closing prices of the asset.

        Returns:
            pd.Series | None: The closing prices of the asset.
        """
        if self._closes is None:
            self._closes = self.historical_data[cst.CLOSE]  # type: ignore
        return self._closes

    @property
    def daily_returns(self) -> np.ndarray | None:
        """Return the daily returns of the asset.

        Returns:
            np.ndarray | None: The daily returns of the asset.
        """
        if self._daily_returns is None:
            closes = self.historical_data[cst.CLOSE]  # type: ignore
            daily_returns = closes.pct_change().dropna().to_numpy()
            self._daily_returns = daily_returns
        return self._daily_returns  # type: ignore

    @daily_returns.setter
    def daily_returns(self, data: list[float]):
        """Set the daily returns of the asset.

        Args:
            data (list[float]): The daily returns of the asset.
        """
        self._daily_returns = data

    @property
    def daily_log_returns(self) -> pd.Series | None:
        """Return the daily log returns of the asset.

        Returns:
            pd.Series | None: The daily log returns of the asset.
        """
        if self._daily_log_returns is None:
            closes = self.historical_data[cst.CLOSE]  # type: ignore
            daily_log_returns = np.log(closes / closes.shift(1)).dropna()  # type: ignore
            self._daily_log_returns = daily_log_returns
        return self._daily_log_returns  # type: ignore

    @daily_log_returns.setter
    def daily_log_returns(self, data: list[float]):
        """Set the daily log returns of the asset.

        Args:
            data (list[float]): The daily log returns of the asset.
        """
        self._daily_log_returns = data

    @property
    def expected_log_return(self) -> float | None:
        """Return the expected log return of the asset.

        Returns:
            float | None: The expected log return of the asset.
        """
        if self._expected_log_return is None:
            if self._expected_return is not None:
                self._expected_log_return = np.log(1 + self._expected_return)
            else:
                self._expected_log_return = (
                    self.daily_log_returns.mean() * self._period_in_day  # type: ignore
                )
        return self._expected_log_return

    @expected_log_return.setter
    def expected_log_return(self, value: float):
        """Set the expected log return of the asset.

        Args:
            value (float): The expected log return of the asset.
        """
        self._expected_log_return = value

    @property
    def expected_return(self) -> float | None:
        """Return the expected return of the asset.

        Returns:
            float | None: The expected return of the asset.
        """
        if self._expected_return is None:
            self._expected_return = np.exp(self.expected_log_return) - 1  # type: ignore
        return self._expected_return

    @expected_return.setter
    def expected_return(self, value: float):
        """Set the expected return of the asset.

        Args:
            value (float): The expected return of the asset.
        """
        self._expected_return = value

    @property
    def volatility(self) -> float:
        """Return the volatility of the asset.

        Returns:
            float: The volatility of the asset.
        """
        return self.daily_log_returns.std() * np.sqrt(self._period_in_day)  # type: ignore

    def __str__(self) -> str:
        return f"Asset({self.symbol})"

    def __repr__(self) -> str:
        return self.__str__()
