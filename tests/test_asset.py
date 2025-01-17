import unittest
from asset import Asset

class TestAsset(unittest.TestCase):
    def setUp(self):
        self.asset = Asset("AAPL")

    def test_asset_name(self):
        self.assertEqual(self.asset.name, "Apple Inc.")