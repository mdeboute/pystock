import yfinance as yf
import numpy as np
import pandas as pd


class Asset:
    def __init__(self, symbol: str, listed: bool = True):
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
        return self.ticker.info["shortName"] if self.listed else self.symbol  # type: ignore

    def fetch_historical_data(
        self, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame | None:
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
                raise Exception(f"Error while fetching data for {self.name}: {e}")
        return self._historic_data

    @property
    def historical_data(self) -> pd.DataFrame | None:
        return self.fetch_historical_data()

    @property
    def closes(self) -> pd.Series | None:
        if self._closes is None:
            self._closes = self.historical_data["Close"]  # type: ignore
        return self._closes

    @property
    def daily_returns(self) -> np.ndarray | None:
        if self._daily_returns is None:
            closes = self.historical_data["Close"]  # type: ignore
            daily_returns = closes.pct_change().dropna().to_numpy()
            self._daily_returns = daily_returns
        return self._daily_returns  # type: ignore

    @daily_returns.setter
    def daily_returns(self, data: list[float]):
        self._daily_returns = data

    @property
    def daily_log_returns(self) -> pd.Series | None:
        if self._daily_log_returns is None:
            closes = self.historical_data["Close"]  # type: ignore
            daily_log_returns = np.log(closes / closes.shift(1)).dropna()  # type: ignore
            self._daily_log_returns = daily_log_returns
        return self._daily_log_returns  # type: ignore

    @daily_log_returns.setter
    def daily_log_returns(self, data: list[float]):
        self._daily_log_returns = data

    @property
    def expected_log_return(self) -> float | None:
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
        self._expected_log_return = value

    @property
    def expected_return(self) -> float | None:
        if self._expected_return is None:
            self._expected_return = np.exp(self.expected_log_return) - 1  # type: ignore
        return self._expected_return

    @expected_return.setter
    def expected_return(self, value: float):
        self._expected_return = value

    @property
    def volatility(self) -> float:
        return self.daily_log_returns.std() * np.sqrt(self._period_in_day)  # type: ignore

    def __str__(self) -> str:
        return f"Asset({self.name})"

    def __repr__(self) -> str:
        return self.__str__()
