#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.data.data_fetcher import DataFetcher
from src.portfolio_optimization.portfolio_optimizer import (
    MeanVarianceOptimizer, BlackLittermanOptimizer, 
    RiskParityOptimizer, FactorOptimizer
)
from src.portfolio_optimization.multi_objective_optimizer import (
    MultiObjectiveOptimizer, DynamicPortfolioOptimizer
)
from src.portfolio_optimization.alternative_data_sources import (
    EconomicDataProvider, SentimentDataProvider, MarketRegimeDetector,
    AlternativeDataIntegrator
)
from src.visualization.visualizer import TradingVisualizer

def main():
    print("📊 Advanced Portfolio Optimization Demo")
    print("=" * 70)
    
    # Configuration
    SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'PG', 'XOM', 'UNH']
    START_DATE = '2020-01-01'
    END_DATE = '2023-12-31'
    
    print(f"📈 Universe: {len(SYMBOLS)} assets")
    print(f"📅 Period: {START_DATE} to {END_DATE}")
    
    # Step 1: Fetch Market Data
    print("\n📥 Fetching market data...")
    data_fetcher = DataFetcher()
    
    try:
        market_data = data_fetcher.fetch_market_data(SYMBOLS, START_DATE, END_DATE)
        
        if not market_data:
            print("❌ Using synthetic data for demo...")
            market_data = generate_synthetic_data(SYMBOLS, START_DATE, END_DATE)
        else:
            print(f"✅ Fetched data for {len(market_data)} assets")
        
    except Exception as e:
        print(f"❌ Data fetch error: {e}")
        print("Using synthetic data for demo...")
        market_data = generate_synthetic_data(SYMBOLS, START_DATE, END_DATE)
    
    # Convert to returns
    returns_data = {}
    for symbol, data in market_data.items():
        returns_data[symbol] = data['Close'].pct_change().dropna()
    
    returns_df = pd.DataFrame(returns_data)
    returns_df = returns_df.dropna()
    
    print(f"Returns matrix shape: {returns_df.shape}")
    
    # Step 2: Basic Portfolio Optimization Methods
    print("\n🔧 Running Portfolio Optimization Methods...")
    
    optimization_results = {}
    
    # 1. Mean Variance Optimization
    print("\n  📊 Mean-Variance Optimization...")
    mv_optimizer = MeanVarianceOptimizer(risk_free_rate=0.02)
    
    try:
        # Maximize Sharpe ratio
        mv_result = mv_optimizer.optimize_portfolio(returns_df)
        optimization_results['Mean-Variance'] = mv_result
        
        print(f"    Return: {mv_result['expected_return']:.3f}")
        print(f"    Risk: {mv_result['volatility']:.3f}")
        print(f"    Sharpe: {mv_result['sharpe_ratio']:.3f}")
        print(f"    Top holdings: {mv_result['weights'].nlargest(3).to_dict()}")
        
        # Generate efficient frontier
        frontier_df = mv_optimizer.efficient_frontier(returns_df, num_points=20)
        print(f"    Generated {len(frontier_df)} efficient frontier points")
        
    except Exception as e:
        print(f"    ❌ Mean-Variance optimization failed: {e}")
    
    # 2. Risk Parity Optimization
    print("\n  ⚖️ Risk Parity Optimization...")
    rp_optimizer = RiskParityOptimizer()
    
    try:
        rp_result = rp_optimizer.optimize_risk_parity(returns_df)
        optimization_results['Risk-Parity'] = rp_result
        
        print(f"    Return: {rp_result['expected_return']:.3f}")
        print(f"    Risk: {rp_result['volatility']:.3f}")
        print(f"    Diversification Ratio: {rp_result['diversification_ratio']:.3f}")
        print(f"    Risk contributions: {rp_result['risk_contributions'].round(3).to_dict()}")
        
    except Exception as e:
        print(f"    ❌ Risk Parity optimization failed: {e}")
    
    # 3. Black-Litterman Optimization
    print("\n  🎯 Black-Litterman Optimization...")
    bl_optimizer = BlackLittermanOptimizer()
    
    try:
        # Create market cap weights (simplified - equal weights for demo)
        market_weights = pd.Series(1/len(SYMBOLS), index=SYMBOLS[:len(returns_df.columns)])
        
        # Create some views
        views = {
            'tech_outperform': {
                'assets': ['AAPL', 'MSFT', 'GOOGL'] if all(x in returns_df.columns for x in ['AAPL', 'MSFT', 'GOOGL']) else [returns_df.columns[0]],
                'return': 0.15,
                'confidence': 0.7
            },
            'energy_underperform': {
                'assets': ['XOM'] if 'XOM' in returns_df.columns else [returns_df.columns[-1]],
                'return': 0.05,
                'confidence': 0.6
            }
        }
        
        bl_result = bl_optimizer.optimize_with_views(
            returns_df, market_weights, views, risk_aversion=3
        )
        optimization_results['Black-Litterman'] = bl_result
        
        print(f"    Return: {bl_result['expected_return']:.3f}")
        print(f"    Risk: {bl_result['volatility']:.3f}")
        print(f"    Sharpe: {bl_result['sharpe_ratio']:.3f}")
        
    except Exception as e:
        print(f"    ❌ Black-Litterman optimization failed: {e}")
    
    # 4. Factor-Based Optimization
    print("\n  🏭 Factor-Based Optimization...")
    factor_optimizer = FactorOptimizer()
    
    try:
        # Estimate factor model using PCA
        factor_model = factor_optimizer.estimate_factor_model(returns_df)
        
        print(f"    Factors extracted: {len(factor_model['factor_loadings'].columns)}")
        if factor_model.get('explained_variance'):
            print(f"    Explained variance: {factor_model['explained_variance']:.3f}")
        
        # Optimize based on factor model
        factor_result = factor_optimizer.optimize_factor_portfolio(returns_df)
        optimization_results['Factor-Based'] = factor_result
        
        print(f"    Return: {factor_result['expected_return']:.3f}")
        print(f"    Risk: {factor_result['volatility']:.3f}")
        
    except Exception as e:
        print(f"    ❌ Factor optimization failed: {e}")
    
    # Step 3: Multi-Objective Optimization
    print("\n🎯 Multi-Objective Optimization...")
    mo_optimizer = MultiObjectiveOptimizer()
    
    try:
        # Pareto optimization
        pareto_result = mo_optimizer.pareto_optimization(
            returns_df,
            objectives=['return', 'risk', 'drawdown'],
            num_portfolios=50
        )
        
        print(f"  📈 Pareto frontier: {pareto_result['num_portfolios']} optimal portfolios")
        
        # Scalarized multi-objective
        scalarized_result = mo_optimizer.scalarization_optimization(
            returns_df,
            objectives={'return': 0.6, 'risk': 0.4},
            method='weighted_sum'
        )
        optimization_results['Multi-Objective'] = scalarized_result
        
        print(f"  📊 Weighted sum result: Sharpe = {scalarized_result['sharpe_ratio']:.3f}")
        
        # Robust optimization
        robust_result = mo_optimizer.robust_optimization(
            returns_df,
            uncertainty_sets={'returns': 0.1, 'covariance': 0.05},
            confidence_level=0.95
        )
        optimization_results['Robust'] = robust_result
        
        print(f"  🛡️ Robust optimization: Sharpe = {robust_result['sharpe_ratio']:.3f}")
        
    except Exception as e:
        print(f"  ❌ Multi-objective optimization failed: {e}")
    
    # Step 4: Alternative Data Integration
    print("\n🌐 Alternative Data Integration...")
    alt_data_integrator = AlternativeDataIntegrator()
    
    try:
        # Create features with alternative data
        sample_market_data = market_data[list(market_data.keys())[0]]  # Use first asset as proxy
        
        enhanced_features = alt_data_integrator.create_alternative_features(
            sample_market_data, START_DATE, END_DATE
        )
        
        print(f"  📊 Enhanced features: {len(enhanced_features.columns)} total features")
        
        # Market regime detection
        regime_detector = MarketRegimeDetector()
        economic_data = alt_data_integrator.economic_provider.get_economic_indicators(
            START_DATE, END_DATE
        )
        
        regime_info = regime_detector.detect_regime(
            sample_market_data, economic_data
        )
        
        print(f"  🎭 Detected regime: {regime_info['primary_regime']} (confidence: {regime_info['confidence']:.2f})")
        
        # Sentiment data
        sentiment_provider = SentimentDataProvider()
        fear_greed = sentiment_provider.get_fear_greed_index()
        
        print(f"  😰 Fear & Greed Index: {fear_greed['value']:.0f} ({fear_greed['text']})")
        
    except Exception as e:
        print(f"  ❌ Alternative data integration failed: {e}")
    
    # Step 5: Dynamic Optimization
    print("\n🔄 Dynamic Portfolio Optimization...")
    dynamic_optimizer = DynamicPortfolioOptimizer(
        rebalance_frequency='quarterly',
        lookback_window=252
    )
    
    try:
        dynamic_result = dynamic_optimizer.dynamic_optimization(
            returns_df,
            optimizer_type='mean_variance'
        )
        
        if 'error' not in dynamic_result:
            print(f"  📊 Rebalancing periods: {len(dynamic_result['rebalance_dates'])}")
            print(f"  📈 Total return: {dynamic_result['performance']['total_return']:.3f}")
            print(f"  📊 Sharpe ratio: {dynamic_result['performance']['sharpe_ratio']:.3f}")
            print(f"  📉 Max drawdown: {dynamic_result['performance']['max_drawdown']:.3f}")
        else:
            print(f"  ❌ Dynamic optimization failed")
        
    except Exception as e:
        print(f"  ❌ Dynamic optimization failed: {e}")
    
    # Step 6: Performance Comparison
    print("\n📊 Portfolio Performance Comparison")
    print("=" * 50)
    
    if optimization_results:
        comparison_df = pd.DataFrame({
            name: {
                'Expected Return': result.get('expected_return', 0),
                'Volatility': result.get('volatility', 0),
                'Sharpe Ratio': result.get('sharpe_ratio', 0)
            }
            for name, result in optimization_results.items()
        }).T
        
        print(comparison_df.round(4))
        
        # Find best Sharpe ratio
        best_sharpe = comparison_df['Sharpe Ratio'].max()
        best_method = comparison_df['Sharpe Ratio'].idxmax()
        
        print(f"\n🏆 Best Sharpe Ratio: {best_method} ({best_sharpe:.3f})")
    
    # Step 7: Visualization
    print("\n📈 Creating Visualizations...")
    
    try:
        os.makedirs('results', exist_ok=True)
        visualizer = TradingVisualizer()
        
        # Create portfolio comparison chart
        if optimization_results:
            # Convert optimization results to format expected by visualizer
            strategy_results = {}
            for name, result in optimization_results.items():
                strategy_results[name] = {
                    'metrics': {
                        'Total_Return_Pct': result.get('expected_return', 0) * 100,
                        'Sharpe_Ratio': result.get('sharpe_ratio', 0),
                        'Volatility_Pct': result.get('volatility', 0) * 100
                    }
                }
            
            comparison_fig = visualizer.plot_strategy_comparison(
                strategy_results,
                save_path='results/portfolio_optimization_comparison.html'
            )
            print("  ✅ Portfolio comparison chart saved")
        
        # Efficient frontier plot (if available)
        if 'Mean-Variance' in optimization_results and 'frontier_df' in locals():
            print("  ✅ Efficient frontier data available")
        
    except Exception as e:
        print(f"  ❌ Visualization failed: {e}")
    
    # Step 8: Summary & Insights
    print(f"\n📋 PORTFOLIO OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    print(f"🔬 Methods Tested: {len(optimization_results)}")
    
    if optimization_results:
        print(f"🏆 Best Method: {best_method}")
        print(f"📊 Best Sharpe: {best_sharpe:.3f}")
        
        # Key insights
        print(f"\n💡 Key Insights:")
        print(f"   • Mean-variance optimization provides theoretical maximum Sharpe ratio")
        print(f"   • Risk parity offers better diversification across assets")
        print(f"   • Black-Litterman incorporates market views and equilibrium assumptions")
        print(f"   • Multi-objective optimization balances competing objectives")
        print(f"   • Dynamic rebalancing adapts to changing market conditions")
        print(f"   • Alternative data provides regime-aware allocation adjustments")
    
    print(f"\n📁 Results saved to 'results/' directory")
    print(f"   • Portfolio comparison: results/portfolio_optimization_comparison.html")
    
    print(f"\n🎯 Advanced Features Demonstrated:")
    print(f"   ✅ Mean-Variance Optimization (Markowitz)")
    print(f"   ✅ Black-Litterman Model with investor views")
    print(f"   ✅ Risk Parity and Equal Risk Contribution")
    print(f"   ✅ Factor-based optimization with PCA")
    print(f"   ✅ Multi-objective optimization (Pareto frontier)")
    print(f"   ✅ Robust optimization under uncertainty")
    print(f"   ✅ Alternative data integration")
    print(f"   ✅ Market regime detection")
    print(f"   ✅ Dynamic rebalancing strategies")
    
    print(f"\n🎉 Portfolio Optimization Demo Complete!")

def generate_synthetic_data(symbols: List[str], start_date: str, end_date: str) -> Dict:
    """Generate synthetic market data for demo"""
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    synthetic_data = {}
    
    np.random.seed(42)  # For reproducible results
    
    for symbol in symbols:
        # Generate correlated returns
        returns = np.random.normal(0.0008, 0.02, len(dates))  # ~20% annual vol
        
        # Add some correlation structure
        if symbol in ['AAPL', 'MSFT', 'GOOGL']:  # Tech correlation
            returns += np.random.normal(0, 0.005, len(dates))
        elif symbol in ['JPM', 'XOM']:  # Value correlation
            returns += np.random.normal(0, 0.003, len(dates))
        
        # Generate price series
        prices = 100 * np.exp(np.cumsum(returns))
        
        synthetic_data[symbol] = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
    
    return synthetic_data

if __name__ == "__main__":
    main()