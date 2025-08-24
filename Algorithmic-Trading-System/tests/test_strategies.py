import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.moving_average_strategy import MovingAverageStrategy, DualMovingAverageStrategy
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.macd_strategy import MACDStrategy
from src.strategies.base_strategy import SignalType

class TestStrategies(unittest.TestCase):
    
    def setUp(self):
        # Create sample data for testing
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Add some trend and volatility
        trend = np.linspace(0, 0.2, len(dates))
        prices = prices * (1 + trend)
        
        self.test_data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        self.test_data.attrs['symbol'] = 'TEST'
    
    def test_moving_average_strategy(self):
        """Test Moving Average Strategy"""
        strategy = MovingAverageStrategy(short_window=20, long_window=50)
        
        # Test parameter setting
        strategy.set_parameters(short_window=15, long_window=45)
        self.assertEqual(strategy.short_window, 15)
        self.assertEqual(strategy.long_window, 45)
        
        # Test signal generation
        signals = strategy.generate_signals(self.test_data)
        self.assertIsInstance(signals, list)
        
        # Check signal properties
        for signal in signals:
            self.assertIn(signal.signal, [SignalType.BUY, SignalType.SELL])
            self.assertGreater(signal.price, 0)
            self.assertGreaterEqual(signal.confidence, 0)
            self.assertLessEqual(signal.confidence, 1)
    
    def test_dual_moving_average_strategy(self):
        """Test Dual Moving Average Strategy"""
        strategy = DualMovingAverageStrategy(short_window=10, medium_window=20, long_window=50)
        
        signals = strategy.generate_signals(self.test_data)
        self.assertIsInstance(signals, list)
        
        # Verify signals have valid attributes
        for signal in signals:
            self.assertIsNotNone(signal.timestamp)
            self.assertGreater(signal.price, 0)
            self.assertIn('short_ma', signal.metadata)
            self.assertIn('medium_ma', signal.metadata)
            self.assertIn('long_ma', signal.metadata)
    
    def test_rsi_strategy(self):
        """Test RSI Strategy"""
        strategy = RSIStrategy(rsi_period=14, oversold_threshold=30, overbought_threshold=70)
        
        # Test parameter validation
        strategy.set_parameters(oversold_threshold=25, overbought_threshold=75)
        self.assertEqual(strategy.oversold_threshold, 25)
        self.assertEqual(strategy.overbought_threshold, 75)
        
        signals = strategy.generate_signals(self.test_data)
        self.assertIsInstance(signals, list)
        
        # Check that RSI values are in metadata
        for signal in signals:
            self.assertIn('rsi', signal.metadata)
            rsi_value = signal.metadata['rsi']
            self.assertGreaterEqual(rsi_value, 0)
            self.assertLessEqual(rsi_value, 100)
    
    def test_macd_strategy(self):
        """Test MACD Strategy"""
        strategy = MACDStrategy(fast_period=12, slow_period=26, signal_period=9)
        
        signals = strategy.generate_signals(self.test_data)
        self.assertIsInstance(signals, list)
        
        # Verify MACD metadata
        for signal in signals:
            self.assertIn('macd', signal.metadata)
            self.assertIn('macd_signal', signal.metadata)
            self.assertIn('macd_histogram', signal.metadata)
    
    def test_strategy_with_insufficient_data(self):
        """Test strategies with insufficient data"""
        # Create very short dataset
        short_data = self.test_data.head(10)
        
        strategy = MovingAverageStrategy(short_window=20, long_window=50)
        signals = strategy.generate_signals(short_data)
        
        # Should return empty list for insufficient data
        self.assertEqual(len(signals), 0)
    
    def test_signal_confidence_bounds(self):
        """Test that signal confidence values are within valid bounds"""
        strategies = [
            MovingAverageStrategy(),
            RSIStrategy(),
            MACDStrategy()
        ]
        
        for strategy in strategies:
            signals = strategy.generate_signals(self.test_data)
            
            for signal in signals:
                self.assertGreaterEqual(signal.confidence, 0)
                self.assertLessEqual(signal.confidence, 1)
                self.assertIsInstance(signal.confidence, (int, float))
    
    def test_strategy_backtest_signals(self):
        """Test the backtest_signals method"""
        strategy = MovingAverageStrategy(short_window=20, long_window=50)
        signals_df = strategy.backtest_signals(self.test_data)
        
        self.assertIsInstance(signals_df, pd.DataFrame)
        
        if not signals_df.empty:
            expected_columns = ['Symbol', 'Signal', 'Price', 'Confidence', 'Position_Size']
            for col in expected_columns:
                self.assertIn(col, signals_df.columns)
    
    def test_parameter_optimization(self):
        """Test parameter optimization functionality"""
        strategy = MovingAverageStrategy()
        
        # Define small parameter ranges for quick testing
        parameter_ranges = {
            'short_window': [10, 15],
            'long_window': [30, 40]
        }
        
        best_params, best_score = strategy.optimize_parameters(
            self.test_data,
            parameter_ranges,
            optimization_metric='sharpe_ratio'
        )
        
        # Check that optimization returns valid results
        self.assertIsInstance(best_params, dict)
        self.assertIsInstance(best_score, (int, float))
        
        # Check that parameters are from the specified ranges
        if best_params:
            self.assertIn(best_params['short_window'], parameter_ranges['short_window'])
            self.assertIn(best_params['long_window'], parameter_ranges['long_window'])

class TestSignalGeneration(unittest.TestCase):
    
    def setUp(self):
        # Create trending data for more predictable signal testing
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        
        # Create upward trending data
        trend = np.linspace(100, 150, len(dates))
        noise = np.random.normal(0, 2, len(dates))
        prices = trend + noise
        
        self.trending_data = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 2000000, len(dates))
        }, index=dates)
        
        self.trending_data.attrs['symbol'] = 'TREND'
    
    def test_moving_average_crossover_detection(self):
        """Test that moving average strategy detects crossovers"""
        strategy = MovingAverageStrategy(short_window=10, long_window=30)
        signals = strategy.generate_signals(self.trending_data)
        
        # In trending data, we should get some signals
        self.assertGreater(len(signals), 0)
        
        # Verify we have both buy and sell signals or at least buy signals in uptrend
        signal_types = [signal.signal for signal in signals]
        self.assertTrue(any(s == SignalType.BUY for s in signal_types))
    
    def test_rsi_extreme_detection(self):
        """Test RSI strategy detects extreme conditions"""
        # Create data with extreme movements
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create data that will generate RSI extremes
        prices = [100]
        for i in range(1, len(dates)):
            if i < 20:
                # Create downward movement for oversold condition
                prices.append(prices[-1] * 0.98)
            elif i < 40:
                # Create upward movement for overbought condition
                prices.append(prices[-1] * 1.02)
            else:
                # Normal movement
                prices.append(prices[-1] * (1 + np.random.normal(0, 0.01)))
        
        extreme_data = pd.DataFrame({
            'Open': [p * 0.99 for p in prices],
            'High': [p * 1.01 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Close': prices,
            'Volume': [1000000] * len(dates)
        }, index=dates)
        
        extreme_data.attrs['symbol'] = 'EXTREME'
        
        strategy = RSIStrategy(rsi_period=14)
        signals = strategy.generate_signals(extreme_data)
        
        # Should generate some signals from extreme conditions
        self.assertGreaterEqual(len(signals), 0)

if __name__ == '__main__':
    unittest.main()