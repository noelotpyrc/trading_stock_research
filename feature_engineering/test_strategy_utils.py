import unittest
import pandas as pd
import numpy as np
from feature_engineering import indicators, microstructure, session_utils

class TestAnalytics(unittest.TestCase):
    
    def setUp(self):
        # Create dummy 1-second OHLCV data
        # 10 periods
        dates = pd.date_range(start='2023-01-01 09:30:00', periods=10, freq='1s', tz='UTC')
        self.df = pd.DataFrame({
            'timestamp': dates,
            'open':  [100, 101, 102, 101, 100, 99, 98, 99, 100, 101],
            'high':  [101, 102, 103, 102, 101, 100, 99, 100, 101, 102],
            'low':   [99,  100, 101, 100, 99,  98, 97, 98, 99,  100],
            'close': [100, 102, 102, 100, 99,  98, 97, 99, 101, 101], # Mixed closes
            'vol':   [100, 200, 150, 100, 300, 100, 50, 100, 200, 100]
        })
        
    def test_estimate_delta(self):
        print("\nTesting Delta Estimation...")
        # Case 1: Close = High (Max buying)
        # Row 1: H=102, L=100, C=102. Mid=101. HalfRange=1. Pos=(102-101)/1 = 1. Delta = Vol * 1
        self.df.loc[1, 'close'] = 102
        delta_series = indicators.estimate_delta(self.df)
        self.assertAlmostEqual(delta_series[1], 200.0) # Vol is 200
        
        # Case 2: Close = Low (Max selling)
        # Row 4: H=101, L=99, C=99. Mid=100. HalfRange=1. Pos=(99-100)/1 = -1. Delta = Vol * -1
        self.df.loc[4, 'close'] = 99
        delta_series = indicators.estimate_delta(self.df)
        self.assertAlmostEqual(delta_series[4], -300.0) # Vol is 300
        
        # Case 3: Close = Midpoint (Neutral)
        # Row 0: H=101, L=99, C=100. Mid=100. Pos=0. Delta=0
        self.df.loc[0, 'close'] = 100
        delta_series = indicators.estimate_delta(self.df)
        self.assertAlmostEqual(delta_series[0], 0.0)
        
        print("Delta Estimation Passed.")

    def test_calculate_cvd(self):
        print("\nTesting CVD...")
        # Mock delta for predictability
        self.df['delta'] = [10, 20, -10, -20, 0, 10, 10, -10, 0, 0]
        
        # Test full CVD
        cvd = indicators.calculate_cvd(self.df)
        self.assertEqual(cvd.iloc[-1], 10) # Sum is 10
        
        # Test Anchored CVD (start at index 5)
        start_time = self.df['timestamp'].iloc[5]
        cvd_anchored = indicators.calculate_cvd(self.df, start_time=start_time)
        
        # Should be NaN before index 5
        self.assertTrue(pd.isna(cvd_anchored.iloc[0]))
        # Index 5 should be delta[5] (10)
        self.assertEqual(cvd_anchored.iloc[5], 10)
        # Index 6 should be 10 + 10 = 20
        self.assertEqual(cvd_anchored.iloc[6], 20)
        
        print("CVD Passed.")

    def test_anchored_vwap(self):
        print("\nTesting Anchored VWAP...")
        # Simple case: 2 bars
        # Bar 0: P=100 (Typical), V=100 -> PV=10000
        # Bar 1: P=101 (Typical), V=200 -> PV=20200
        # CumPV = 30200, CumVol = 300 -> VWAP = 100.66
        
        # Overwrite typical price components for simplicity
        self.df['high'] = [100, 101] + [100]*8
        self.df['low'] = [100, 101] + [100]*8
        self.df['close'] = [100, 101] + [100]*8
        self.df['vol'] = [100, 200] + [100]*8
        
        vwap = indicators.calculate_anchored_vwap(self.df)
        self.assertAlmostEqual(vwap.iloc[1], 100.666666, places=5)
        
        print("Anchored VWAP Passed.")

    def test_opening_range(self):
        print("\nTesting Opening Range...")
        start_time = self.df['timestamp'].iloc[0]
        # Window of 2 seconds (Rows 0, 1, 2)
        # Row 0: H=101, L=99
        # Row 1: H=102, L=100
        # Row 2: H=103, L=101
        # Max High = 103, Min Low = 99
        
        # Reset data to original setup
        self.setUp()
        
        h, l = microstructure.get_opening_range(self.df, start_time, duration_seconds=2)
        # Note: duration_seconds=2 means t0 to t0+2s.
        # Timestamps: 09:30:00, 09:30:01, 09:30:02.
        # If inclusive, it covers 3 rows.
        # Logic in microstructure: <= end_time.
        # So 00, 01, 02 are included.
        
        self.assertEqual(h, 103) # Row 2 High
        self.assertEqual(l, 99)  # Row 0 Low
        
        print("Opening Range Passed.")

    def test_volume_spike(self):
        print("\nTesting Volume Spike...")
        # Vol: 100, 200, 150, 100, 300...
        # Mean of first 5 (100+200+150+100+300)/5 = 850/5 = 170
        # Multiplier 1.5
        # 300 > 1.5 * 170 (255) -> True
        
        spikes = microstructure.detect_volume_spike(self.df, window=5, multiplier=1.5)
        self.assertTrue(spikes.iloc[4]) # The 300 vol bar
        
        print("Volume Spike Passed.")

if __name__ == '__main__':
    unittest.main()
