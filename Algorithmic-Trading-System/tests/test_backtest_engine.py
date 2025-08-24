import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.backtest_engine import BacktestEngine, Position, Trade, BacktestResults
from src.strategies.base_strategy import TradingSignal, SignalType

class TestBacktestEngine(unittest.TestCase):
    
    def setUp(self):
        # Create sample market data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        prices = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
        
        self.market_data = {
            'TEST': pd.DataFrame({
                'Open': prices * 0.99,
                'High': prices * 1.02,
                'Low': prices * 0.98,
                'Close': prices,
                'Volume': np.random.randint(1000000, 5000000, len(dates))
            }, index=dates)
        }
        
        # Create sample signals
        self.signals = {
            'TEST': [
                TradingSignal(
                    timestamp=pd.Timestamp('2023-02-01'),
                    symbol='TEST',
                    signal=SignalType.BUY,
                    price=105.0,
                    confidence=0.8
                ),
                TradingSignal(
                    timestamp=pd.Timestamp('2023-03-01'),
                    symbol='TEST',
                    signal=SignalType.SELL,
                    price=110.0,
                    confidence=0.7
                ),
                TradingSignal(
                    timestamp=pd.Timestamp('2023-06-01'),
                    symbol='TEST',
                    signal=SignalType.BUY,
                    price=108.0,
                    confidence=0.9
                )
            ]
        }
        
        self.engine = BacktestEngine(initial_capital=100000)
    
    def test_initialization(self):
        """Test BacktestEngine initialization"""
        self.assertEqual(self.engine.initial_capital, 100000)
        self.assertEqual(self.engine.cash, 100000)
        self.assertEqual(self.engine.commission_rate, 0.001)
        self.assertEqual(len(self.engine.positions), 0)
        self.assertEqual(len(self.engine.trades), 0)
    
    def test_reset(self):
        """Test engine reset functionality"""
        # Modify engine state
        self.engine.cash = 50000
        self.engine.positions['TEST'] = Position('TEST', 100, 105.0, pd.Timestamp.now(), 'long')
        self.engine.trades.append(Trade('TEST', pd.Timestamp.now(), pd.Timestamp.now(), 
                                      105.0, 110.0, 100, 'long', 500, 4.76, 105.0))
        
        # Reset and check
        self.engine.reset()
        
        self.assertEqual(self.engine.cash, 100000)
        self.assertEqual(len(self.engine.positions), 0)
        self.assertEqual(len(self.engine.trades), 0)
    
    def test_position_size_calculation(self):
        """Test position size calculation"""
        price = 100.0
        position_size = self.engine.calculate_position_size(price, confidence=1.0)
        
        # Should be positive integer
        self.assertIsInstance(position_size, int)
        self.assertGreaterEqual(position_size, 0)
        
        # Test with different confidence levels
        size_high_conf = self.engine.calculate_position_size(price, confidence=1.0)
        size_low_conf = self.engine.calculate_position_size(price, confidence=0.5)
        
        # Higher confidence should result in larger position (or equal)
        self.assertGreaterEqual(size_high_conf, size_low_conf)
    
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation"""
        # Initially should equal cash
        portfolio_value = self.engine.calculate_portfolio_value(self.market_data)
        self.assertEqual(portfolio_value, self.engine.initial_capital)
        
        # Add a position and test again
        self.engine.positions['TEST'] = Position('TEST', 100, 105.0, pd.Timestamp.now(), 'long')
        self.engine.current_date = self.market_data['TEST'].index[50]  # Set current date
        
        portfolio_value = self.engine.calculate_portfolio_value(self.market_data)
        expected_value = (self.engine.cash + 
                         100 * self.market_data['TEST'].loc[self.engine.current_date, 'Close'])
        
        self.assertEqual(portfolio_value, expected_value)
    
    def test_trade_execution_buy(self):
        """Test buy trade execution"""
        initial_cash = self.engine.cash
        price = 100.0
        quantity = 100
        date = pd.Timestamp('2023-01-15')
        
        self.engine.execute_trade('TEST', quantity, price, 'buy', date)
        
        # Check cash reduction
        expected_cost = quantity * price * (1 + self.engine.slippage) + \
                       quantity * price * (1 + self.engine.slippage) * self.engine.commission_rate
        self.assertAlmostEqual(self.engine.cash, initial_cash - expected_cost, places=2)
        
        # Check position creation
        self.assertIn('TEST', self.engine.positions)
        position = self.engine.positions['TEST']
        self.assertEqual(position.quantity, quantity)
        self.assertEqual(position.position_type, 'long')
    
    def test_trade_execution_sell(self):
        """Test sell trade execution"""
        # First create a long position
        buy_date = pd.Timestamp('2023-01-15')
        self.engine.execute_trade('TEST', 100, 100.0, 'buy', buy_date)
        
        initial_cash = self.engine.cash
        sell_price = 110.0
        sell_quantity = 50
        sell_date = pd.Timestamp('2023-01-20')
        
        # Execute partial sell
        self.engine.execute_trade('TEST', sell_quantity, sell_price, 'sell', sell_date)
        
        # Check that position was partially closed
        self.assertEqual(self.engine.positions['TEST'].quantity, 50)
        
        # Check that a trade was recorded
        self.assertEqual(len(self.engine.trades), 1)
        trade = self.engine.trades[0]
        self.assertEqual(trade.symbol, 'TEST')
        self.assertEqual(trade.quantity, sell_quantity)
        self.assertGreater(trade.pnl, 0)  # Should be profitable
    
    def test_run_backtest(self):
        """Test complete backtest run"""
        results = self.engine.run_backtest(
            market_data=self.market_data,
            signals=self.signals,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        # Check results structure
        self.assertIsInstance(results, BacktestResults)
        self.assertIsInstance(results.trades, list)
        self.assertIsInstance(results.portfolio_values, pd.Series)
        self.assertIsInstance(results.metrics, dict)
        
        # Should have some trades
        self.assertGreater(len(results.trades), 0)
        
        # Should have portfolio values recorded
        self.assertGreater(len(results.portfolio_values), 0)
        
        # Should have calculated metrics
        self.assertIn('Total_Return_Pct', results.metrics)
        self.assertIn('Sharpe_Ratio', results.metrics)
        self.assertIn('Max_Drawdown_Pct', results.metrics)
    
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation"""
        # Run backtest first
        results = self.engine.run_backtest(
            market_data=self.market_data,
            signals=self.signals
        )
        
        metrics = results.metrics
        
        # Check that key metrics are present
        required_metrics = [
            'Total_Return_Pct', 'Volatility_Pct', 'Sharpe_Ratio',
            'Max_Drawdown_Pct', 'Win_Rate_Pct', 'Total_Trades'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
        
        # Check metric ranges
        self.assertGreaterEqual(metrics['Win_Rate_Pct'], 0)
        self.assertLessEqual(metrics['Win_Rate_Pct'], 100)
        self.assertGreaterEqual(metrics['Total_Trades'], 0)
        self.assertGreaterEqual(metrics['Max_Drawdown_Pct'], 0)
    
    def test_results_to_dataframe(self):
        """Test conversion of results to DataFrame"""
        results = self.engine.run_backtest(
            market_data=self.market_data,
            signals=self.signals
        )
        
        df = results.to_dataframe()
        
        if not df.empty:
            # Check DataFrame structure
            expected_columns = [
                'Symbol', 'Entry_Date', 'Exit_Date', 'Entry_Price',
                'Exit_Price', 'Quantity', 'Position_Type', 'PnL',
                'PnL_Pct', 'Commission', 'Duration_Days'
            ]
            
            for col in expected_columns:
                self.assertIn(col, df.columns)
            
            # Check data types
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['Entry_Date']))
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['Exit_Date']))
            self.assertTrue(pd.api.types.is_numeric_dtype(df['PnL']))

class TestPosition(unittest.TestCase):
    
    def test_position_creation(self):
        """Test Position object creation"""
        position = Position(
            symbol='TEST',
            quantity=100,
            entry_price=105.0,
            entry_date=pd.Timestamp('2023-01-15'),
            position_type='long'
        )
        
        self.assertEqual(position.symbol, 'TEST')
        self.assertEqual(position.quantity, 100)
        self.assertEqual(position.entry_price, 105.0)
        self.assertEqual(position.position_type, 'long')

class TestTrade(unittest.TestCase):
    
    def test_trade_creation(self):
        """Test Trade object creation"""
        trade = Trade(
            symbol='TEST',
            entry_date=pd.Timestamp('2023-01-15'),
            exit_date=pd.Timestamp('2023-01-20'),
            entry_price=100.0,
            exit_price=105.0,
            quantity=100,
            position_type='long',
            pnl=500.0,
            pnl_pct=5.0,
            commission=10.0,
            duration=5
        )
        
        self.assertEqual(trade.symbol, 'TEST')
        self.assertEqual(trade.quantity, 100)
        self.assertEqual(trade.pnl, 500.0)
        self.assertEqual(trade.pnl_pct, 5.0)
        self.assertEqual(trade.duration, 5)

class TestBacktestWithComplexScenarios(unittest.TestCase):
    
    def setUp(self):
        # Create more complex test data
        dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
        np.random.seed(123)
        
        # Simulate price movements with volatility clusters
        returns = np.random.normal(0.001, 0.02, len(dates))
        # Add volatility clustering
        for i in range(10, len(returns)):
            if abs(returns[i-1]) > 0.03:
                returns[i] *= 1.5
        
        prices = 100 * np.exp(np.cumsum(returns))
        
        self.complex_data = {
            'COMPLEX': pd.DataFrame({
                'Open': prices * (1 + np.random.normal(0, 0.002, len(dates))),
                'High': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
                'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
                'Close': prices,
                'Volume': np.random.randint(500000, 3000000, len(dates))
            }, index=dates)
        }
        
        self.engine = BacktestEngine(
            initial_capital=50000,
            commission_rate=0.002,
            slippage=0.0005
        )
    
    def test_frequent_trading_scenario(self):
        """Test scenario with frequent buy/sell signals"""
        # Create frequent signals
        signals = []
        signal_dates = pd.date_range(start='2023-01-15', end='2023-05-15', freq='W')
        
        for i, date in enumerate(signal_dates):
            signal_type = SignalType.BUY if i % 2 == 0 else SignalType.SELL
            price = self.complex_data['COMPLEX'].loc[date, 'Close'] if date in self.complex_data['COMPLEX'].index else 100
            
            signals.append(TradingSignal(
                timestamp=date,
                symbol='COMPLEX',
                signal=signal_type,
                price=price,
                confidence=0.6
            ))
        
        complex_signals = {'COMPLEX': signals}
        
        results = self.engine.run_backtest(
            market_data=self.complex_data,
            signals=complex_signals
        )
        
        # Should handle frequent trading
        self.assertGreater(len(results.trades), 5)  # Should have multiple trades
        self.assertIsNotNone(results.metrics)
        
        # Portfolio value should be recorded throughout
        self.assertGreater(len(results.portfolio_values), 50)
    
    def test_no_signals_scenario(self):
        """Test backtest with no trading signals"""
        empty_signals = {'COMPLEX': []}
        
        results = self.engine.run_backtest(
            market_data=self.complex_data,
            signals=empty_signals
        )
        
        # Should handle empty signals gracefully
        self.assertEqual(len(results.trades), 0)
        self.assertGreater(len(results.portfolio_values), 0)
        
        # Final portfolio value should equal initial capital (minus any fees)
        final_value = results.portfolio_values.iloc[-1]
        self.assertAlmostEqual(final_value, self.engine.initial_capital, delta=100)

if __name__ == '__main__':
    unittest.main()