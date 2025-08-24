#!/usr/bin/env python3
"""
Run Streamlit dashboard for live trading system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import subprocess
import threading
import time
from datetime import datetime

from src.real_time.live_strategy_engine import LiveStrategyEngine
from src.real_time.paper_trading import PaperTradingAccount
from src.real_time.live_data_feed import LiveDataFeed
from src.strategies.moving_average_strategy import MovingAverageStrategy
from src.strategies.rsi_strategy import RSIStrategy

def setup_trading_system():
    """Setup and start the trading system"""
    print("Setting up live trading system...")
    
    # Initialize components
    paper_account = PaperTradingAccount(initial_balance=100000, commission_rate=0.001)
    data_feed = LiveDataFeed()
    strategy_engine = LiveStrategyEngine(paper_account)
    strategy_engine.set_data_feed(data_feed)
    
    # Add strategies
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    ma_strategy = MovingAverageStrategy(short_window=10, long_window=30, confidence_threshold=0.6)
    strategy_engine.add_strategy('MA_10_30', ma_strategy, symbols[:3])
    
    rsi_strategy = RSIStrategy(period=14, oversold_threshold=30, overbought_threshold=70, confidence_threshold=0.5)
    strategy_engine.add_strategy('RSI_14', rsi_strategy, symbols[2:])
    
    # Start the engine
    strategy_engine.start_engine(symbols)
    
    print("✅ Trading system started successfully")
    return strategy_engine

def run_streamlit_app():
    """Run the Streamlit app"""
    dashboard_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'dashboard', 'streamlit_dashboard.py')
    
    try:
        subprocess.run([
            'streamlit', 'run', dashboard_path,
            '--server.port=8501',
            '--server.address=0.0.0.0',
            '--theme.base=light'
        ])
    except KeyboardInterrupt:
        print("\nStreamlit dashboard stopped.")
    except Exception as e:
        print(f"Error running Streamlit: {e}")

def main():
    print("=" * 60)
    print("Live Trading Dashboard - Streamlit Version")
    print("=" * 60)
    
    # Setup trading system
    strategy_engine = setup_trading_system()
    
    print("\nWaiting for data accumulation (15 seconds)...")
    time.sleep(15)
    
    print("\n" + "=" * 60)
    print("LAUNCHING STREAMLIT DASHBOARD")
    print("=" * 60)
    print("Dashboard will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        # Start Streamlit dashboard
        run_streamlit_app()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
        strategy_engine.stop_engine()
        print("✅ System stopped successfully")

if __name__ == "__main__":
    main()