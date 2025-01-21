import unittest

from pystock import Asset, Portfolio


class TestPortfolio(unittest.TestCase):
    def setUp(self):
        self.appl_stock = Asset("AAPL")
        self.tsla_stock = Asset("TSLA")
        self.msft_stock = Asset("MSFT")

    def test_bad_weighted_portfolio(self):
        with self.assertRaises(ValueError):
            Portfolio([self.appl_stock, self.tsla_stock, self.msft_stock], [0.3, 0.3, 0.3])
