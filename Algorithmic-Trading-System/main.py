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
from src.strategies.moving_average_strategy import MovingAverageStrategy, DualMovingAverageStrategy
from src.strategies.rsi_strategy import RSIStrategy, RSIDivergenceStrategy
from src.strategies.macd_strategy import MACDStrategy, MACDZeroCrossStrategy, MACDDivergenceStrategy
from src.backtesting.backtest_engine import BacktestEngine
from src.risk_management.risk_manager import RiskManager
from src.visualization.visualizer import TradingVisualizer

def main():
    print("🚀 Algorithmic Trading System")
    print("=" * 50)
    
    # Configuration
    SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    START_DATE = '2020-01-01'
    END_DATE = '2023-12-31'
    INITIAL_CAPITAL = 100000
    
    print(f"📊 Fetching data for symbols: {', '.join(SYMBOLS)}")
    print(f"📅 Date range: {START_DATE} to {END_DATE}")
    print(f"💰 Initial capital: ${INITIAL_CAPITAL:,}")
    
    # Step 1: Fetch Data
    data_fetcher = DataFetcher()
    try:
        market_data = data_fetcher.fetch_market_data(SYMBOLS, START_DATE, END_DATE)
        if not market_data:
            print("❌ No data fetched. Please check your internet connection and try again.")
            return
        print(f"✅ Successfully fetched data for {len(market_data)} symbols")
    except Exception as e:
        print(f"❌ Error fetching data: {str(e)}")
        return
    
    # Step 2: Preprocess Data
    print("\n🔧 Preprocessing data and calculating technical indicators...")
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process_multiple_symbols(
        market_data, 
        add_technical_indicators=True,
        add_features=True
    )
    
    # Step 3: Initialize Strategies
    strategies = {
        'Moving Average': MovingAverageStrategy(short_window=20, long_window=50),
        'Dual MA': DualMovingAverageStrategy(short_window=10, medium_window=20, long_window=50),
        'RSI': RSIStrategy(rsi_period=14, oversold_threshold=30, overbought_threshold=70),
        'MACD': MACDStrategy(fast_period=12, slow_period=26, signal_period=9),
        'MACD Zero Cross': MACDZeroCrossStrategy()
    }
    
    # Step 4: Generate Signals for All Strategies
    print("\n📡 Generating trading signals...")
    all_strategy_signals = {}
    
    for strategy_name, strategy in strategies.items():
        print(f"  🔄 {strategy_name}...")
        strategy_signals = {}
        
        for symbol in SYMBOLS:
            if symbol in processed_data:
                # Set symbol attribute for signal generation
                symbol_data = processed_data[symbol].copy()
                symbol_data.attrs['symbol'] = symbol
                signals = strategy.generate_signals(symbol_data)
                strategy_signals[symbol] = signals
        
        all_strategy_signals[strategy_name] = strategy_signals
        total_signals = sum(len(signals) for signals in strategy_signals.values())
        print(f"    ✅ Generated {total_signals} signals across all symbols")
    
    # Step 5: Backtesting
    print("\n🧪 Running backtests...")
    backtest_engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
    risk_manager = RiskManager()
    
    strategy_results = {}
    
    for strategy_name, strategy_signals in all_strategy_signals.items():
        print(f"  📈 Backtesting {strategy_name}...")
        
        try:
            results = backtest_engine.run_backtest(
                market_data=processed_data,
                signals=strategy_signals,
                start_date=START_DATE,
                end_date=END_DATE
            )
            
            strategy_results[strategy_name] = {
                'results': results,
                'metrics': results.metrics
            }
            
            # Print key metrics
            metrics = results.metrics
            print(f"    💰 Total Return: {metrics.get('Total_Return_Pct', 0):.2f}%")
            print(f"    📊 Sharpe Ratio: {metrics.get('Sharpe_Ratio', 0):.2f}")
            print(f"    📉 Max Drawdown: {metrics.get('Max_Drawdown_Pct', 0):.2f}%")
            print(f"    🎯 Win Rate: {metrics.get('Win_Rate_Pct', 0):.1f}%")
            print(f"    🔢 Total Trades: {metrics.get('Total_Trades', 0)}")
            
        except Exception as e:
            print(f"    ❌ Error in backtesting {strategy_name}: {str(e)}")
            continue
    
    # Step 6: Risk Analysis
    print("\n⚠️  Performing risk analysis...")
    for strategy_name, result_data in strategy_results.items():
        if result_data['results'].portfolio_values.empty:
            continue
            
        portfolio_returns = result_data['results'].portfolio_values.pct_change().dropna()
        
        # Create dummy position data for risk analysis
        positions = {'Portfolio': result_data['results'].portfolio_values.iloc[-1] - INITIAL_CAPITAL}
        returns_data = {'Portfolio': portfolio_returns}
        
        risk_metrics = risk_manager.calculate_portfolio_risk(positions, returns_data)
        risk_level = risk_manager.assess_risk_level(risk_metrics)
        
        print(f"  🔍 {strategy_name} Risk Level: {risk_level.name}")
    
    # Step 7: Visualization
    print("\n📊 Creating visualizations...")
    visualizer = TradingVisualizer()
    
    try:
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Strategy comparison chart
        comparison_fig = visualizer.plot_strategy_comparison(
            strategy_results,
            save_path='results/strategy_comparison.html'
        )
        print("  ✅ Strategy comparison chart saved to results/strategy_comparison.html")
        
        # Individual strategy analysis for best performing strategy
        best_strategy = max(strategy_results.items(), 
                          key=lambda x: x[1]['metrics'].get('Total_Return_Pct', -999))
        
        if best_strategy[1]['results'].portfolio_values is not None and not best_strategy[1]['results'].portfolio_values.empty:
            portfolio_fig = visualizer.plot_portfolio_performance(
                best_strategy[1]['results'].portfolio_values,
                trades=best_strategy[1]['results'].to_dataframe(),
                save_path=f'results/{best_strategy[0].lower().replace(" ", "_")}_performance.html'
            )
            print(f"  ✅ {best_strategy[0]} performance chart saved")
        
        # Risk metrics visualization
        risk_fig = visualizer.plot_risk_metrics(
            best_strategy[1]['metrics'],
            save_path='results/risk_analysis.html'
        )
        print("  ✅ Risk analysis chart saved to results/risk_analysis.html")
        
        # Performance report
        report_path = visualizer.create_performance_report(
            best_strategy[1],
            save_path='results/performance_report.html'
        )
        print(f"  ✅ Comprehensive performance report saved to {report_path}")
        
    except Exception as e:
        print(f"  ❌ Error creating visualizations: {str(e)}")
    
    # Step 8: Summary
    print("\n📋 SUMMARY")
    print("=" * 50)
    
    if strategy_results:
        best_strategy_name = max(strategy_results.items(), 
                               key=lambda x: x[1]['metrics'].get('Total_Return_Pct', -999))[0]
        best_metrics = strategy_results[best_strategy_name]['metrics']
        
        print(f"🏆 Best Performing Strategy: {best_strategy_name}")
        print(f"💰 Best Return: {best_metrics.get('Total_Return_Pct', 0):.2f}%")
        print(f"📊 Best Sharpe: {best_metrics.get('Sharpe_Ratio', 0):.2f}")
        print(f"🎯 Win Rate: {best_metrics.get('Win_Rate_Pct', 0):.1f}%")
        
        print(f"\n📁 Results saved to 'results/' directory:")
        print(f"   • Strategy comparison: results/strategy_comparison.html")
        print(f"   • Performance charts: results/")
        print(f"   • Risk analysis: results/risk_analysis.html")
        print(f"   • Full report: results/performance_report.html")
    else:
        print("❌ No successful backtests completed")
    
    print("\n🎉 Analysis complete!")

if __name__ == "__main__":
    main()