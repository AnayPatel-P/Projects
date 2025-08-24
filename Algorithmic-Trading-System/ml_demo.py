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
from src.data.preprocessor import DataPreprocessor
from src.strategies.ml_strategy import MLPredictionStrategy, MLEnsembleStrategy
from src.strategies.moving_average_strategy import MovingAverageStrategy
from src.backtesting.backtest_engine import BacktestEngine
from src.visualization.visualizer import TradingVisualizer
from src.ml.feature_engineering import AdvancedFeatureEngineer
from src.ml.models import ModelEvaluator

def main():
    print("🤖 Machine Learning Trading Strategy Demo")
    print("=" * 60)
    
    # Configuration
    SYMBOL = 'AAPL'
    START_DATE = '2020-01-01'
    END_DATE = '2023-12-31'
    INITIAL_CAPITAL = 100000
    
    print(f"📊 Symbol: {SYMBOL}")
    print(f"📅 Period: {START_DATE} to {END_DATE}")
    print(f"💰 Capital: ${INITIAL_CAPITAL:,}")
    
    # Step 1: Fetch and preprocess data
    print("\n📥 Fetching market data...")
    data_fetcher = DataFetcher()
    
    try:
        market_data = data_fetcher.fetch_market_data([SYMBOL], START_DATE, END_DATE)
        if not market_data or SYMBOL not in market_data:
            print("❌ Failed to fetch data. Using synthetic data for demo...")
            # Create synthetic data for demo
            dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
            np.random.seed(42)
            prices = 100 + np.cumsum(np.random.normal(0.1, 2, len(dates)))
            
            market_data = {
                SYMBOL: pd.DataFrame({
                    'Open': prices * (1 + np.random.normal(0, 0.01, len(dates))),
                    'High': prices * (1 + np.abs(np.random.normal(0, 0.015, len(dates)))),
                    'Low': prices * (1 - np.abs(np.random.normal(0, 0.015, len(dates)))),
                    'Close': prices,
                    'Volume': np.random.randint(1000000, 5000000, len(dates))
                }, index=dates)
            }
            print("✅ Using synthetic data for demo")
        else:
            print(f"✅ Fetched {len(market_data[SYMBOL])} days of data")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Preprocess data
    print("\n🔧 Preprocessing data...")
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process_multiple_symbols(
        market_data, 
        add_technical_indicators=True,
        add_features=True
    )
    
    # Set symbol attribute for strategies
    symbol_data = processed_data[SYMBOL].copy()
    symbol_data.attrs['symbol'] = SYMBOL
    
    # Step 2: Feature Engineering Demo
    print("\n🧠 Demonstrating Advanced Feature Engineering...")
    feature_engineer = AdvancedFeatureEngineer()
    
    try:
        # Show original features
        print(f"Original features: {len(symbol_data.columns)}")
        
        # Create advanced features
        featured_data = feature_engineer.engineer_all_features(symbol_data)
        print(f"Enhanced features: {len(featured_data.columns)}")
        
        # Show sample of new features
        new_features = [col for col in featured_data.columns if col not in symbol_data.columns]
        print(f"New features (sample): {new_features[:10]}")
        
    except Exception as e:
        print(f"⚠️ Feature engineering warning: {e}")
        featured_data = symbol_data
    
    # Step 3: ML Strategy Comparison
    print("\n🚀 Setting up ML strategies...")
    
    strategies = {
        'Traditional MA': MovingAverageStrategy(short_window=20, long_window=50),
        'ML Ensemble': MLPredictionStrategy(
            model_type='ensemble',
            prediction_horizon=1,
            confidence_threshold=0.5,
            feature_selection_k=30
        )
    }
    
    # Add LSTM if we have enough data
    if len(featured_data) > 200:
        strategies['ML LSTM'] = MLPredictionStrategy(
            model_type='lstm',
            prediction_horizon=1,
            confidence_threshold=0.6,
            feature_selection_k=40
        )
    
    # Step 4: Generate signals and backtest
    print("\n📡 Generating signals and backtesting...")
    backtest_engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
    evaluator = ModelEvaluator()
    
    strategy_results = {}
    
    for strategy_name, strategy in strategies.items():
        print(f"\n  🔄 Testing {strategy_name}...")
        
        try:
            # Generate signals
            if 'ML' in strategy_name:
                # For ML strategies, we need more sophisticated signal generation
                signals = strategy.generate_signals(featured_data)
            else:
                # Traditional strategy
                signals = strategy.generate_signals(symbol_data)
            
            print(f"    Generated {len(signals)} signals")
            
            if not signals:
                print(f"    ⚠️ No signals generated for {strategy_name}")
                continue
            
            # Backtest
            signals_dict = {SYMBOL: signals}
            
            results = backtest_engine.run_backtest(
                market_data={SYMBOL: featured_data},
                signals=signals_dict,
                start_date=START_DATE,
                end_date=END_DATE
            )
            
            strategy_results[strategy_name] = {
                'results': results,
                'metrics': results.metrics,
                'signals_count': len(signals)
            }
            
            # Print results
            metrics = results.metrics
            print(f"    💰 Return: {metrics.get('Total_Return_Pct', 0):.2f}%")
            print(f"    📊 Sharpe: {metrics.get('Sharpe_Ratio', 0):.2f}")
            print(f"    📉 Max DD: {metrics.get('Max_Drawdown_Pct', 0):.2f}%")
            print(f"    🎯 Win Rate: {metrics.get('Win_Rate_Pct', 0):.1f}%")
            print(f"    🔢 Trades: {metrics.get('Total_Trades', 0)}")
            
        except Exception as e:
            print(f"    ❌ Error in {strategy_name}: {str(e)}")
            continue
    
    # Step 5: Model Performance Analysis
    print("\n📈 Performance Analysis...")
    
    if len(strategy_results) >= 2:
        # Compare strategies
        comparison_data = {}
        for name, result in strategy_results.items():
            comparison_data[name] = result['metrics']
        
        comparison_df = pd.DataFrame(comparison_data).T
        print("\nStrategy Comparison:")
        print(comparison_df.round(3))
        
        # Find best ML strategy
        ml_strategies = {k: v for k, v in strategy_results.items() if 'ML' in k}
        if ml_strategies:
            best_ml = max(ml_strategies.items(), 
                         key=lambda x: x[1]['metrics'].get('Total_Return_Pct', -999))
            print(f"\n🏆 Best ML Strategy: {best_ml[0]}")
            print(f"   Return: {best_ml[1]['metrics'].get('Total_Return_Pct', 0):.2f}%")
    
    # Step 6: Feature Importance Analysis
    print("\n🔍 Feature Importance Analysis...")
    
    try:
        # Get ML strategy for feature analysis
        ml_strategy = None
        for name, strategy in strategies.items():
            if 'ML' in name and hasattr(strategy, 'model') and strategy.model:
                ml_strategy = strategy
                break
        
        if ml_strategy and hasattr(ml_strategy.model, 'get_feature_importance'):
            importance = ml_strategy.model.get_feature_importance(
                ml_strategy.selected_features or featured_data.columns.tolist()
            )
            
            if not importance.empty:
                print("\nTop 10 Most Important Features:")
                print(importance.head(10))
            else:
                print("Feature importance not available for this model type")
        else:
            print("Feature importance analysis not available")
            
    except Exception as e:
        print(f"Feature importance analysis failed: {e}")
    
    # Step 7: Visualization
    print("\n📊 Creating visualizations...")
    visualizer = TradingVisualizer()
    
    try:
        os.makedirs('results', exist_ok=True)
        
        # Strategy comparison
        if strategy_results:
            comparison_fig = visualizer.plot_strategy_comparison(
                strategy_results,
                save_path='results/ml_strategy_comparison.html'
            )
            print("  ✅ ML strategy comparison saved")
        
        # Best strategy performance
        if strategy_results:
            best_strategy = max(strategy_results.items(), 
                              key=lambda x: x[1]['metrics'].get('Total_Return_Pct', -999))
            
            portfolio_fig = visualizer.plot_portfolio_performance(
                best_strategy[1]['results'].portfolio_values,
                trades=best_strategy[1]['results'].to_dataframe(),
                save_path=f'results/ml_best_strategy_performance.html'
            )
            print(f"  ✅ {best_strategy[0]} performance chart saved")
    
    except Exception as e:
        print(f"  ❌ Visualization error: {e}")
    
    # Step 8: Summary
    print(f"\n📋 ML DEMO SUMMARY")
    print("=" * 50)
    
    if strategy_results:
        best_strategy_name = max(strategy_results.items(), 
                               key=lambda x: x[1]['metrics'].get('Total_Return_Pct', -999))[0]
        best_metrics = strategy_results[best_strategy_name]['metrics']
        
        print(f"🏆 Best Strategy: {best_strategy_name}")
        print(f"💰 Best Return: {best_metrics.get('Total_Return_Pct', 0):.2f}%")
        print(f"📊 Best Sharpe: {best_metrics.get('Sharpe_Ratio', 0):.2f}")
        print(f"🎯 Win Rate: {best_metrics.get('Win_Rate_Pct', 0):.1f}%")
        
        # ML vs Traditional comparison
        ml_returns = []
        traditional_returns = []
        
        for name, result in strategy_results.items():
            return_pct = result['metrics'].get('Total_Return_Pct', 0)
            if 'ML' in name:
                ml_returns.append(return_pct)
            else:
                traditional_returns.append(return_pct)
        
        if ml_returns and traditional_returns:
            avg_ml = np.mean(ml_returns)
            avg_traditional = np.mean(traditional_returns)
            improvement = avg_ml - avg_traditional
            
            print(f"\n📈 ML vs Traditional:")
            print(f"   ML Average Return: {avg_ml:.2f}%")
            print(f"   Traditional Return: {avg_traditional:.2f}%")
            print(f"   Improvement: {improvement:+.2f}%")
        
        print(f"\n📁 Results saved to 'results/' directory")
        print(f"   • ML strategy comparison: results/ml_strategy_comparison.html")
        print(f"   • Best strategy performance: results/ml_best_strategy_performance.html")
    else:
        print("❌ No successful strategy results")
    
    print("\n🎉 ML Demo completed!")
    print("\n💡 Next Steps:")
    print("   • Experiment with different ML models (XGBoost, LSTM, Transformer)")
    print("   • Try ensemble methods combining multiple models") 
    print("   • Implement walk-forward validation")
    print("   • Add alternative data sources (sentiment, news, etc.)")
    print("   • Implement real-time prediction capabilities")

if __name__ == "__main__":
    main()