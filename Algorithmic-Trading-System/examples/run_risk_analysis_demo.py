#!/usr/bin/env python3
"""
Advanced Risk Management and Analysis Demo
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.risk_management.advanced_risk_models import AdvancedRiskModel, RealTimeRiskMonitor, RiskAlert
from src.data.data_fetcher import DataFetcher

def create_sample_portfolio_data():
    """Create sample portfolio returns for demonstration"""
    print("📊 Generating sample portfolio data...")
    
    # Create synthetic return data for demonstration
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # Generate correlated returns for 5 assets
    n_assets = 5
    n_days = len(dates)
    
    # Base returns with different characteristics
    mean_returns = np.array([0.0008, 0.0010, 0.0006, 0.0012, 0.0015])  # Daily returns
    volatilities = np.array([0.015, 0.020, 0.012, 0.025, 0.030])  # Daily volatilities
    
    # Create correlation matrix
    correlation_matrix = np.array([
        [1.00, 0.60, 0.40, 0.30, 0.50],
        [0.60, 1.00, 0.50, 0.40, 0.60],
        [0.40, 0.50, 1.00, 0.35, 0.45],
        [0.30, 0.40, 0.35, 1.00, 0.55],
        [0.50, 0.60, 0.45, 0.55, 1.00]
    ])
    
    # Generate correlated returns
    random_normals = np.random.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=correlation_matrix,
        size=n_days
    )
    
    # Scale by volatilities and add drift
    returns_data = {}
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    for i, symbol in enumerate(symbols):
        returns = mean_returns[i] + volatilities[i] * random_normals[:, i]
        returns_data[symbol] = returns
    
    returns_df = pd.DataFrame(returns_data, index=dates)
    
    print(f"✅ Generated {len(returns_df)} days of return data for {len(symbols)} assets")
    print(f"📅 Date range: {returns_df.index[0].strftime('%Y-%m-%d')} to {returns_df.index[-1].strftime('%Y-%m-%d')}")
    
    return returns_df

def demonstrate_risk_metrics(returns_df: pd.DataFrame):
    """Demonstrate comprehensive risk metric calculations"""
    print("\n" + "="*70)
    print("📈 ADVANCED RISK METRICS ANALYSIS")
    print("="*70)
    
    # Initialize risk model
    risk_model = AdvancedRiskModel(confidence_level=0.05, lookback_period=252)
    
    # Portfolio weights (equal weight for demo)
    n_assets = len(returns_df.columns)
    equal_weights = np.array([1/n_assets] * n_assets)
    
    print(f"\n🎯 Portfolio Composition:")
    for i, symbol in enumerate(returns_df.columns):
        print(f"   {symbol}: {equal_weights[i]:.1%}")
    
    # Calculate portfolio-level risk metrics
    print(f"\n⚡ Calculating Risk Metrics...")
    portfolio_metrics = risk_model.calculate_portfolio_risk(returns_df, equal_weights)
    
    print(f"\n📊 PORTFOLIO RISK ANALYSIS")
    print("-" * 40)
    
    # Display key metrics
    metrics_to_show = [
        ('Annual Return', 'annual_return', '{:.2%}'),
        ('Annual Volatility', 'annual_volatility', '{:.2%}'),
        ('Sharpe Ratio', 'sharpe_ratio', '{:.3f}'),
        ('Sortino Ratio', 'sortino_ratio', '{:.3f}'),
        ('Calmar Ratio', 'calmar_ratio', '{:.3f}'),
        ('Maximum Drawdown', 'max_drawdown', '{:.2%}'),
        ('VaR (95%)', 'var_95', '{:.2%}'),
        ('CVaR (95%)', 'cvar_95', '{:.2%}'),
        ('Skewness', 'skewness', '{:.3f}'),
        ('Kurtosis', 'kurtosis', '{:.3f}'),
        ('Win Rate', 'win_rate', '{:.1%}'),
        ('Average Correlation', 'avg_correlation', '{:.3f}'),
        ('Concentration Risk', 'concentration_risk', '{:.3f}'),
        ('Effective # Assets', 'effective_num_assets', '{:.1f}')
    ]
    
    for display_name, metric_key, format_str in metrics_to_show:
        value = portfolio_metrics.get(metric_key)
        if value is not None:
            print(f"{display_name:<20}: {format_str.format(value)}")
    
    return risk_model, portfolio_metrics, equal_weights

def demonstrate_stress_testing(risk_model: AdvancedRiskModel, returns_df: pd.DataFrame, weights: np.ndarray):
    """Demonstrate portfolio stress testing"""
    print(f"\n🚨 STRESS TESTING ANALYSIS")
    print("-" * 40)
    
    stress_results = risk_model.stress_test_portfolio(returns_df, weights)
    
    for scenario, results in stress_results.items():
        print(f"\n📉 {scenario.replace('_', ' ').title()} Scenario:")
        print(f"   Portfolio Return: {results['portfolio_return']:.2%}")
        print(f"   Portfolio Vol:    {results['portfolio_vol']:.2%}")
        print(f"   Max Drawdown:     {results['max_drawdown']:.2%}")
        print(f"   VaR (95%):        {results['var_95']:.2%}")

def demonstrate_monte_carlo_simulation(risk_model: AdvancedRiskModel, returns_df: pd.DataFrame, weights: np.ndarray):
    """Demonstrate Monte Carlo risk simulation"""
    print(f"\n🎲 MONTE CARLO SIMULATION")
    print("-" * 40)
    
    print("🔄 Running Monte Carlo simulation (10,000 scenarios)...")
    mc_results = risk_model.monte_carlo_risk_simulation(
        returns_df, weights, num_simulations=10000, horizon_days=252
    )
    
    print(f"\n📈 Monte Carlo Results (1-year horizon):")
    print(f"   Expected Return:        {mc_results['expected_return']:.2%}")
    print(f"   VaR (95%):             {mc_results['var_95']:.2%}")
    print(f"   VaR (99%):             {mc_results['var_99']:.2%}")
    print(f"   Probability of Loss:    {mc_results['probability_of_loss']:.1%}")
    print(f"   Prob of 20%+ Loss:      {mc_results['probability_large_loss']:.1%}")
    print(f"   Expected Max Drawdown:  {mc_results['expected_max_drawdown']:.2%}")
    
    return mc_results

def demonstrate_risk_parity(risk_model: AdvancedRiskModel, returns_df: pd.DataFrame):
    """Demonstrate risk parity optimization"""
    print(f"\n⚖️  RISK PARITY ANALYSIS")
    print("-" * 40)
    
    print("🔄 Calculating risk parity weights...")
    rp_weights = risk_model.calculate_risk_parity_weights(returns_df)
    
    print(f"\n📊 Risk Parity vs Equal Weight Allocation:")
    equal_weights = np.array([1/len(returns_df.columns)] * len(returns_df.columns))
    
    print(f"{'Asset':<8} {'Equal Weight':<12} {'Risk Parity':<12}")
    print("-" * 35)
    
    for i, symbol in enumerate(returns_df.columns):
        print(f"{symbol:<8} {equal_weights[i]:<12.1%} {rp_weights[i]:<12.1%}")
    
    # Compare performance
    equal_weight_returns = (returns_df * equal_weights).sum(axis=1)
    rp_returns = (returns_df * rp_weights).sum(axis=1)
    
    eq_metrics = risk_model.calculate_risk_adjusted_metrics(equal_weight_returns)
    rp_metrics = risk_model.calculate_risk_adjusted_metrics(rp_returns)
    
    print(f"\n📈 Performance Comparison:")
    comparison_metrics = [
        ('Annual Return', 'annual_return', '{:.2%}'),
        ('Annual Volatility', 'annual_volatility', '{:.2%}'),
        ('Sharpe Ratio', 'sharpe_ratio', '{:.3f}'),
        ('Max Drawdown', 'max_drawdown', '{:.2%}')
    ]
    
    print(f"{'Metric':<18} {'Equal Weight':<15} {'Risk Parity':<15}")
    print("-" * 50)
    
    for display_name, metric_key, format_str in comparison_metrics:
        eq_val = eq_metrics.get(metric_key, 0)
        rp_val = rp_metrics.get(metric_key, 0)
        print(f"{display_name:<18} {format_str.format(eq_val):<15} {format_str.format(rp_val):<15}")

def demonstrate_risk_alerts(risk_model: AdvancedRiskModel, returns_df: pd.DataFrame, weights: np.ndarray):
    """Demonstrate risk limit monitoring"""
    print(f"\n🚨 RISK LIMIT MONITORING")
    print("-" * 40)
    
    # Calculate current metrics
    portfolio_returns = (returns_df * weights).sum(axis=1)
    current_metrics = risk_model.calculate_risk_adjusted_metrics(portfolio_returns)
    
    # Check against risk limits
    alerts = risk_model.check_risk_limits(portfolio_returns, current_metrics, weights)
    
    if alerts:
        print(f"\n⚠️  RISK ALERTS DETECTED ({len(alerts)} alerts):")
        for i, alert in enumerate(alerts, 1):
            severity_emoji = {
                'low': '🟡', 'medium': '🟠', 'high': '🔴', 'critical': '🚨'
            }
            emoji = severity_emoji.get(alert.severity, '⚪')
            print(f"   {emoji} Alert {i}: {alert.message}")
            print(f"      Severity: {alert.severity.upper()}")
            print(f"      Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("✅ No risk limit breaches detected")

def demonstrate_comprehensive_report(risk_model: AdvancedRiskModel, returns_df: pd.DataFrame, weights: np.ndarray):
    """Generate and display comprehensive risk report"""
    print(f"\n📋 COMPREHENSIVE RISK REPORT")
    print("-" * 40)
    
    print("🔄 Generating comprehensive risk report...")
    report = risk_model.generate_risk_report(returns_df, weights)
    
    print(f"\n📊 PORTFOLIO SUMMARY")
    print(f"   Timestamp: {report['timestamp']}")
    print(f"   Number of Strategies: {report['num_strategies']}")
    
    print(f"\n⚠️  RISK ALERTS:")
    if report['risk_alerts']:
        for alert in report['risk_alerts']:
            print(f"   • {alert['message']} (Severity: {alert['severity']})")
    else:
        print("   ✅ No active risk alerts")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"   • {rec}")
    
    return report

def create_risk_visualizations(risk_model: AdvancedRiskModel, returns_df: pd.DataFrame, 
                             weights: np.ndarray, mc_results: dict):
    """Create comprehensive risk analysis visualizations"""
    print(f"\n📈 CREATING RISK ANALYSIS VISUALIZATIONS")
    print("-" * 40)
    
    # Create the risk analysis plots
    try:
        print("🎨 Generating risk analysis dashboard...")
        risk_model.plot_risk_analysis(returns_df, weights)
        
        # Additional Monte Carlo visualization
        plt.figure(figsize=(15, 10))
        
        # Monte Carlo simulation paths (sample)
        paths = mc_results['simulated_paths']
        sample_paths = paths[np.random.choice(len(paths), 100, replace=False)]
        
        plt.subplot(2, 2, 1)
        for path in sample_paths:
            plt.plot(path, alpha=0.1, color='blue')
        plt.plot(np.mean(paths, axis=0), color='red', linewidth=2, label='Average Path')
        plt.title('Monte Carlo Simulation Paths (Sample)')
        plt.xlabel('Days')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        
        # Final returns distribution
        plt.subplot(2, 2, 2)
        plt.hist(mc_results['final_returns'], bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(mc_results['var_95'], color='red', linestyle='--', label=f"VaR 95%: {mc_results['var_95']:.2%}")
        plt.axvline(mc_results['var_99'], color='orange', linestyle='--', label=f"VaR 99%: {mc_results['var_99']:.2%}")
        plt.title('Final Returns Distribution')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Drawdown distribution
        plt.subplot(2, 2, 3)
        plt.hist(mc_results['simulated_max_drawdowns'], bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(mc_results['expected_max_drawdown'], color='red', linestyle='--', 
                   label=f"Expected: {mc_results['expected_max_drawdown']:.2%}")
        plt.title('Maximum Drawdown Distribution')
        plt.xlabel('Max Drawdown')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Risk metrics comparison
        plt.subplot(2, 2, 4)
        portfolio_returns = (returns_df * weights).sum(axis=1)
        risk_metrics = risk_model.calculate_risk_adjusted_metrics(portfolio_returns)
        
        metrics_names = ['Annual Return', 'Annual Vol', 'Sharpe Ratio', 'Max Drawdown']
        metrics_values = [
            risk_metrics.get('annual_return', 0),
            risk_metrics.get('annual_volatility', 0),
            risk_metrics.get('sharpe_ratio', 0),
            abs(risk_metrics.get('max_drawdown', 0))
        ]
        
        bars = plt.bar(range(len(metrics_names)), metrics_values)
        plt.xticks(range(len(metrics_names)), metrics_names, rotation=45)
        plt.title('Key Risk Metrics')
        plt.ylabel('Value')
        
        # Color bars based on values
        for i, bar in enumerate(bars):
            if i == 2:  # Sharpe ratio - green if > 1
                bar.set_color('green' if metrics_values[i] > 1 else 'orange')
            elif i == 3:  # Max drawdown - red if > 10%
                bar.set_color('red' if metrics_values[i] > 0.1 else 'green')
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Risk analysis visualizations completed")
        
    except Exception as e:
        print(f"⚠️  Error creating visualizations: {e}")

def demonstrate_real_time_monitoring():
    """Demonstrate real-time risk monitoring setup"""
    print(f"\n⏰ REAL-TIME RISK MONITORING DEMO")
    print("-" * 40)
    
    # Create risk model and monitor
    risk_model = AdvancedRiskModel()
    monitor = RealTimeRiskMonitor(risk_model)
    
    # Add custom risk alert callback
    def risk_alert_handler(alert: RiskAlert):
        severity_icons = {'low': '🟡', 'medium': '🟠', 'high': '🔴', 'critical': '🚨'}
        icon = severity_icons.get(alert.severity, '⚪')
        print(f"   {icon} RISK ALERT: {alert.message}")
    
    monitor.add_risk_callback(risk_alert_handler)
    
    print("🔧 Real-time risk monitor configured with custom alert handler")
    print("📡 In a live trading environment, this would:")
    print("   • Monitor positions and prices in real-time")
    print("   • Calculate risk metrics continuously")
    print("   • Trigger alerts when limits are breached")
    print("   • Log all risk events for analysis")
    print("   • Enable automated risk management responses")
    
    return monitor

def main():
    """Main demonstration function"""
    print("=" * 80)
    print("🚀 ADVANCED RISK MANAGEMENT & ANALYSIS DEMONSTRATION")
    print("=" * 80)
    print("This demo showcases comprehensive risk management capabilities")
    print("for algorithmic trading systems including:")
    print("• Portfolio risk metrics (VaR, CVaR, Sharpe, Sortino, etc.)")
    print("• Stress testing and scenario analysis") 
    print("• Monte Carlo simulation")
    print("• Risk parity optimization")
    print("• Real-time risk monitoring")
    print("• Comprehensive risk reporting")
    print("=" * 80)
    
    # Generate sample data
    returns_df = create_sample_portfolio_data()
    
    # Demonstrate risk metrics
    risk_model, portfolio_metrics, weights = demonstrate_risk_metrics(returns_df)
    
    # Demonstrate stress testing
    demonstrate_stress_testing(risk_model, returns_df, weights)
    
    # Demonstrate Monte Carlo simulation
    mc_results = demonstrate_monte_carlo_simulation(risk_model, returns_df, weights)
    
    # Demonstrate risk parity
    demonstrate_risk_parity(risk_model, returns_df)
    
    # Demonstrate risk alerts
    demonstrate_risk_alerts(risk_model, returns_df, weights)
    
    # Generate comprehensive report
    report = demonstrate_comprehensive_report(risk_model, returns_df, weights)
    
    # Create visualizations
    create_risk_visualizations(risk_model, returns_df, weights, mc_results)
    
    # Demonstrate real-time monitoring
    monitor = demonstrate_real_time_monitoring()
    
    print(f"\n" + "="*80)
    print("✅ ADVANCED RISK MANAGEMENT DEMONSTRATION COMPLETED")
    print("="*80)
    print("🎯 Key Features Demonstrated:")
    print("   ✓ Comprehensive portfolio risk analysis")
    print("   ✓ Advanced statistical risk measures")
    print("   ✓ Stress testing across multiple scenarios")
    print("   ✓ Monte Carlo simulation for forward-looking risk")
    print("   ✓ Risk parity portfolio optimization")
    print("   ✓ Risk limit monitoring and alerting")
    print("   ✓ Professional risk reporting")
    print("   ✓ Advanced risk visualizations")
    print("   ✓ Real-time monitoring framework")
    print("\n💼 This system is now ready for professional algorithmic trading")
    print("   risk management in institutional or personal trading environments!")
    print("="*80)

if __name__ == "__main__":
    main()