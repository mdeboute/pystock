import unittest

from pystock.asset import Asset


class TestAsset(unittest.TestCase):
    def setUp(self):
        self.asset = Asset("AAPL")

    def test_asset_name(self):
        self.assertEqual(self.asset.name, "Apple Inc.")

    def test_historical_data(self):
        self.asset.fetch_historical_data()
        self.assertIsNotNone(self.asset._historic_data)
        self.assertEqual(self.asset._period_in_day, self.asset._historic_data.shape[0])  # type: ignore
