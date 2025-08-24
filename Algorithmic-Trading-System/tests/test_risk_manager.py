import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.risk_management.risk_manager import RiskManager, RiskMetrics, RiskLevel

class TestRiskManager(unittest.TestCase):
    
    def setUp(self):
        self.risk_manager = RiskManager(
            max_position_size=0.1,
            max_portfolio_risk=0.02,
            max_daily_loss=0.05,
            var_confidence=0.95
        )
        
        # Create sample returns data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        
        # Generate correlated returns for multiple assets
        base_returns = np.random.normal(0.001, 0.02, len(dates))
        
        self.returns_data = {
            'ASSET1': pd.Series(base_returns + np.random.normal(0, 0.005, len(dates)), index=dates),
            'ASSET2': pd.Series(base_returns * 0.8 + np.random.normal(0, 0.015, len(dates)), index=dates),
            'ASSET3': pd.Series(np.random.normal(0.0005, 0.025, len(dates)), index=dates)
        }
        
        self.positions = {
            'ASSET1': 30000,
            'ASSET2': 25000,
            'ASSET3': 15000
        }
    
    def test_initialization(self):
        """Test RiskManager initialization"""
        self.assertEqual(self.risk_manager.max_position_size, 0.1)
        self.assertEqual(self.risk_manager.max_portfolio_risk, 0.02)
        self.assertEqual(self.risk_manager.max_daily_loss, 0.05)
        self.assertEqual(self.risk_manager.var_confidence, 0.95)
    
    def test_position_size_calculation(self):
        """Test position size calculation using Kelly Criterion"""
        symbol = 'TEST'
        price = 100.0
        portfolio_value = 100000.0
        volatility = 0.2
        confidence = 0.8
        
        position_size = self.risk_manager.calculate_position_size(
            symbol, price, portfolio_value, volatility, confidence
        )
        
        # Should return non-negative integer
        self.assertIsInstance(position_size, int)
        self.assertGreaterEqual(position_size, 0)
        
        # Higher confidence should generally result in larger positions
        size_high_conf = self.risk_manager.calculate_position_size(
            symbol, price, portfolio_value, volatility, 0.9
        )
        size_low_conf = self.risk_manager.calculate_position_size(
            symbol, price, portfolio_value, volatility, 0.6
        )
        
        # This may not always hold due to Kelly formula, but generally true
        self.assertGreaterEqual(size_high_conf, size_low_conf)
    
    def test_var_calculation(self):
        """Test Value at Risk calculation"""
        returns = self.returns_data['ASSET1']
        
        var_95, cvar_95 = self.risk_manager.calculate_var(returns, 0.95)
        var_99, cvar_99 = self.risk_manager.calculate_var(returns, 0.99)
        
        # VaR and CVaR should be positive (since we take absolute values)
        self.assertGreaterEqual(var_95, 0)
        self.assertGreaterEqual(cvar_95, 0)
        self.assertGreaterEqual(var_99, 0)
        self.assertGreaterEqual(cvar_99, 0)
        
        # 99% VaR should be greater than 95% VaR
        self.assertGreaterEqual(var_99, var_95)
        
        # CVaR should be greater than or equal to VaR
        self.assertGreaterEqual(cvar_95, var_95)
        self.assertGreaterEqual(cvar_99, var_99)
    
    def test_var_calculation_insufficient_data(self):
        """Test VaR calculation with insufficient data"""
        short_returns = pd.Series([0.01, -0.02, 0.005])
        var, cvar = self.risk_manager.calculate_var(short_returns)
        
        # Should return 0 for insufficient data
        self.assertEqual(var, 0.0)
        self.assertEqual(cvar, 0.0)
    
    def test_portfolio_risk_calculation(self):
        """Test portfolio risk metrics calculation"""
        risk_metrics = self.risk_manager.calculate_portfolio_risk(
            self.positions,
            self.returns_data
        )
        
        # Check that it returns RiskMetrics object
        self.assertIsInstance(risk_metrics, RiskMetrics)
        
        # Check that metrics are reasonable
        self.assertGreaterEqual(risk_metrics.var_95, 0)
        self.assertGreaterEqual(risk_metrics.volatility, 0)
        self.assertIsInstance(risk_metrics.sharpe_ratio, (int, float))
        self.assertGreaterEqual(risk_metrics.max_drawdown, 0)
        
        # Correlation risk should be between -1 and 1
        self.assertGreaterEqual(risk_metrics.correlation_risk, -1)
        self.assertLessEqual(risk_metrics.correlation_risk, 1)
    
    def test_portfolio_risk_empty_data(self):
        """Test portfolio risk calculation with empty data"""
        risk_metrics = self.risk_manager.calculate_portfolio_risk({}, {})
        
        # Should return zero metrics
        self.assertEqual(risk_metrics.var_95, 0)
        self.assertEqual(risk_metrics.volatility, 0)
        self.assertEqual(risk_metrics.sharpe_ratio, 0)
    
    def test_risk_level_assessment(self):
        """Test risk level assessment"""
        # Create different risk scenarios
        low_risk = RiskMetrics(
            var_95=0.01, var_99=0.015, cvar_95=0.012,
            volatility=0.1, beta=0.8, sharpe_ratio=1.5,
            max_drawdown=0.05, correlation_risk=0.3
        )
        
        high_risk = RiskMetrics(
            var_95=0.08, var_99=0.12, cvar_95=0.1,
            volatility=0.5, beta=1.8, sharpe_ratio=-0.5,
            max_drawdown=0.4, correlation_risk=0.9
        )
        
        low_level = self.risk_manager.assess_risk_level(low_risk)
        high_level = self.risk_manager.assess_risk_level(high_risk)
        
        # Low risk should result in lower risk level
        self.assertIn(low_level, [RiskLevel.LOW, RiskLevel.MEDIUM])
        self.assertIn(high_level, [RiskLevel.HIGH, RiskLevel.EXTREME])
        
        # High risk level should be higher than low risk level
        self.assertGreaterEqual(high_level.value, low_level.value)
    
    def test_risk_limits_checking(self):
        """Test risk limits checking"""
        portfolio_value = 100000
        daily_pnl = -2000  # 2% loss
        
        # Test with positions within limits
        safe_positions = {'ASSET1': 5000, 'ASSET2': 3000}  # 5% and 3% of portfolio
        checks = self.risk_manager.check_risk_limits(safe_positions, portfolio_value, daily_pnl)
        
        self.assertTrue(checks['position_size_ok'])
        self.assertTrue(checks['daily_loss_ok'])
        
        # Test with positions exceeding limits
        large_positions = {'ASSET1': 15000}  # 15% of portfolio (exceeds 10% limit)
        checks = self.risk_manager.check_risk_limits(large_positions, portfolio_value, daily_pnl)
        
        self.assertFalse(checks['position_size_ok'])
        
        # Test with excessive daily loss
        large_loss = -6000  # 6% loss (exceeds 5% limit)
        checks = self.risk_manager.check_risk_limits(safe_positions, portfolio_value, large_loss)
        
        self.assertFalse(checks['daily_loss_ok'])
    
    def test_concentration_risk_checking(self):
        """Test concentration risk assessment"""
        portfolio_value = 100000
        
        # Concentrated portfolio (< 5 positions, large single position)
        concentrated_positions = {
            'ASSET1': 45000,  # 45% of portfolio
            'ASSET2': 5000
        }
        
        checks = self.risk_manager.check_risk_limits(
            concentrated_positions, portfolio_value, 0
        )
        
        self.assertFalse(checks['concentration_ok'])
        
        # Diversified portfolio
        diversified_positions = {
            f'ASSET{i}': 10000 for i in range(1, 8)  # 7 positions, 10% each
        }
        
        checks = self.risk_manager.check_risk_limits(
            diversified_positions, portfolio_value, 0
        )
        
        self.assertTrue(checks['concentration_ok'])
    
    def test_risk_recommendations(self):
        """Test risk recommendations generation"""
        # High risk scenario
        high_risk_metrics = RiskMetrics(
            var_95=0.06, var_99=0.1, cvar_95=0.08,
            volatility=0.35, beta=1.5, sharpe_ratio=0.3,
            max_drawdown=0.25, correlation_risk=0.8
        )
        
        high_risk_level = RiskLevel.HIGH
        failed_checks = {
            'position_size_ok': False,
            'portfolio_risk_ok': True,
            'daily_loss_ok': False,
            'concentration_ok': False
        }
        
        recommendations = self.risk_manager.generate_risk_recommendations(
            high_risk_metrics, high_risk_level, failed_checks
        )
        
        # Should generate multiple recommendations
        self.assertGreater(len(recommendations), 3)
        
        # Should include specific recommendations for failed checks
        rec_text = ' '.join(recommendations).lower()
        self.assertIn('position', rec_text)  # Position size issue
        self.assertIn('loss', rec_text)  # Daily loss issue
        self.assertIn('diversif', rec_text)  # Concentration issue
    
    def test_optimal_hedge_calculation(self):
        """Test optimal hedge calculation"""
        portfolio_beta = 1.2
        portfolio_value = 100000
        hedge_instrument_beta = -1.0
        
        hedge_notional = self.risk_manager.calculate_optimal_hedge(
            portfolio_beta, portfolio_value, hedge_instrument_beta
        )
        
        # Hedge should neutralize portfolio beta
        expected_hedge = -portfolio_beta / hedge_instrument_beta * portfolio_value
        self.assertAlmostEqual(hedge_notional, expected_hedge, places=2)
        
        # Test with different hedge instrument
        spy_beta = -0.8  # SPY inverse ETF
        hedge_notional_spy = self.risk_manager.calculate_optimal_hedge(
            portfolio_beta, portfolio_value, spy_beta
        )
        
        # Should be different from perfect hedge
        self.assertNotEqual(hedge_notional, hedge_notional_spy)
    
    def test_stress_testing(self):
        """Test stress testing functionality"""
        # Test with default scenarios
        stress_results = self.risk_manager.stress_test(
            self.positions,
            self.returns_data
        )
        
        # Should return results for default scenarios
        self.assertGreater(len(stress_results), 0)
        
        # All results should be numeric
        for scenario, impact in stress_results.items():
            self.assertIsInstance(impact, (int, float))
        
        # Market crash scenarios should generally result in negative impacts
        if 'market_crash_20pct' in stress_results:
            self.assertLess(stress_results['market_crash_20pct'], 0)
        
        # Test with custom scenarios
        custom_scenarios = {
            'mild_correction': -0.05,
            'interest_rate_shock': 0.01
        }
        
        custom_results = self.risk_manager.stress_test(
            self.positions,
            self.returns_data,
            custom_scenarios
        )
        
        self.assertEqual(len(custom_results), len(custom_scenarios))
        for scenario in custom_scenarios:
            self.assertIn(scenario, custom_results)

class TestRiskMetrics(unittest.TestCase):
    
    def test_risk_metrics_creation(self):
        """Test RiskMetrics dataclass creation"""
        metrics = RiskMetrics(
            var_95=0.05,
            var_99=0.08,
            cvar_95=0.06,
            volatility=0.2,
            beta=1.1,
            sharpe_ratio=1.2,
            max_drawdown=0.1,
            correlation_risk=0.4
        )
        
        self.assertEqual(metrics.var_95, 0.05)
        self.assertEqual(metrics.volatility, 0.2)
        self.assertEqual(metrics.sharpe_ratio, 1.2)
        self.assertEqual(metrics.correlation_risk, 0.4)

class TestRiskLevel(unittest.TestCase):
    
    def test_risk_level_enum(self):
        """Test RiskLevel enumeration"""
        self.assertEqual(RiskLevel.LOW.value, 1)
        self.assertEqual(RiskLevel.MEDIUM.value, 2)
        self.assertEqual(RiskLevel.HIGH.value, 3)
        self.assertEqual(RiskLevel.EXTREME.value, 4)
        
        # Test ordering
        self.assertLess(RiskLevel.LOW, RiskLevel.HIGH)
        self.assertGreater(RiskLevel.EXTREME, RiskLevel.MEDIUM)

if __name__ == '__main__':
    unittest.main()