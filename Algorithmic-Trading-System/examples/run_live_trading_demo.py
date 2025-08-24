#!/usr/bin/env python3
"""
Live trading demo with real-time dashboard
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import threading
from datetime import datetime
import pandas as pd

from src.real_time.live_strategy_engine import LiveStrategyEngine
from src.real_time.paper_trading import PaperTradingAccount
from src.real_time.live_data_feed import LiveDataFeed
from src.dashboard.live_dashboard import LiveTradingDashboard
from src.strategies.moving_average_strategy import MovingAverageStrategy
from src.strategies.rsi_strategy import RSIStrategy

def main():
    print("=" * 60)
    print("Live Algorithmic Trading Demo with Real-Time Dashboard")
    print("=" * 60)
    
    # Initialize paper trading account
    paper_account = PaperTradingAccount(initial_balance=100000, commission_rate=0.001)
    
    # Initialize live data feed
    data_feed = LiveDataFeed()
    
    # Initialize strategy engine
    strategy_engine = LiveStrategyEngine(paper_account)
    strategy_engine.set_data_feed(data_feed)
    
    # Add trading strategies
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    # Moving Average Strategy
    ma_strategy = MovingAverageStrategy(
        short_window=10,
        long_window=30
    )
    strategy_engine.add_strategy('MA_10_30', ma_strategy, symbols[:3])
    
    # RSI Strategy
    rsi_strategy = RSIStrategy(
        rsi_period=14,
        oversold_threshold=30,
        overbought_threshold=70
    )
    strategy_engine.add_strategy('RSI_14', rsi_strategy, symbols[2:])
    
    print(f"Added strategies for symbols: {symbols}")
    print("Starting live trading engine...")
    
    # Start the strategy engine
    strategy_engine.start_engine(symbols)
    
    # Wait a moment for data to accumulate
    print("Waiting for initial data accumulation (10 seconds)...")
    time.sleep(10)
    
    # Initialize and start dashboard
    print("Starting real-time dashboard...")
    dashboard = LiveTradingDashboard(strategy_engine)
    
    def run_dashboard():
        dashboard.run_server(debug=False, port=8050)
    
    # Start dashboard in separate thread
    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()
    
    print("\n" + "=" * 60)
    print("LIVE TRADING SYSTEM ACTIVE")
    print("=" * 60)
    print("Dashboard: http://localhost:8050")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        # Monitor and display periodic updates
        start_time = datetime.now()
        
        while True:
            time.sleep(30)  # Update every 30 seconds
            
            # Get current status
            status = strategy_engine.get_engine_status()
            account_summary = status['account_summary']
            
            runtime = datetime.now() - start_time
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Runtime: {runtime}")
            print(f"Portfolio Value: ${account_summary['portfolio_value']:,.2f}")
            print(f"Total Return: {account_summary['total_return_pct']:+.2f}%")
            print(f"Cash Balance: ${account_summary['cash_balance']:,.2f}")
            print(f"Recent Signals: {len(status['recent_signals'])}")
            
            # Show strategy performance
            for name, strategy_info in status['strategies'].items():
                print(f"  {name}: {strategy_info['signal_count']} signals, {strategy_info['trades']} trades")
            
            print("-" * 40)
            
    except KeyboardInterrupt:
        print("\nShutting down live trading system...")
        strategy_engine.stop_engine()
        print("System stopped successfully.")

if __name__ == "__main__":
    main()