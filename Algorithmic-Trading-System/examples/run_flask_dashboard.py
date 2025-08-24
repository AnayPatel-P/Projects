#!/usr/bin/env python3
"""
Run Flask web dashboard for live trading system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import threading
import time
from datetime import datetime

from src.real_time.live_strategy_engine import LiveStrategyEngine
from src.real_time.paper_trading import PaperTradingAccount
from src.real_time.live_data_feed import LiveDataFeed
from src.dashboard.flask_api import TradingSystemAPI
from src.strategies.moving_average_strategy import MovingAverageStrategy
from src.strategies.rsi_strategy import RSIStrategy

def setup_trading_system():
    """Setup and start the trading system"""
    print("🚀 Initializing live trading system...")
    
    # Initialize components
    paper_account = PaperTradingAccount(initial_balance=100000, commission_rate=0.001)
    data_feed = LiveDataFeed()
    strategy_engine = LiveStrategyEngine(paper_account)
    strategy_engine.set_data_feed(data_feed)
    
    # Add strategies
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    # Moving Average Strategy
    ma_strategy = MovingAverageStrategy(
        short_window=10,
        long_window=30,
        confidence_threshold=0.6
    )
    strategy_engine.add_strategy('MA_10_30', ma_strategy, symbols[:3])
    
    # RSI Strategy
    rsi_strategy = RSIStrategy(
        period=14,
        oversold_threshold=30,
        overbought_threshold=70,
        confidence_threshold=0.5
    )
    strategy_engine.add_strategy('RSI_14', rsi_strategy, symbols[2:])
    
    print(f"📊 Added strategies for symbols: {symbols}")
    
    # Start the engine
    strategy_engine.start_engine(symbols)
    print("✅ Trading system started successfully")
    
    return strategy_engine

def main():
    print("=" * 70)
    print("🌐 FLASK WEB DASHBOARD FOR ALGORITHMIC TRADING")
    print("=" * 70)
    
    # Setup trading system
    strategy_engine = setup_trading_system()
    
    print("\n⏳ Waiting for data accumulation (15 seconds)...")
    time.sleep(15)
    
    # Initialize Flask API
    print("🌐 Starting Flask web server...")
    api = TradingSystemAPI(strategy_engine)
    
    print("\n" + "=" * 70)
    print("🎯 FLASK DASHBOARD ACTIVE")
    print("=" * 70)
    print("🌐 Web Dashboard:   http://localhost:5000")
    print("🔗 API Endpoints:")
    print("   • Status:        http://localhost:5000/api/status")
    print("   • Account:       http://localhost:5000/api/account")
    print("   • Positions:     http://localhost:5000/api/positions")
    print("   • Trades:        http://localhost:5000/api/trades")
    print("   • Signals:       http://localhost:5000/api/signals")
    print("   • Performance:   http://localhost:5000/api/performance")
    print("   • Health:        http://localhost:5000/api/health")
    print("=" * 70)
    print("Press Ctrl+C to stop")
    print("=" * 70)
    
    try:
        # Start Flask server
        api.run_server(debug=False, host='127.0.0.1', port=5000)
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        strategy_engine.stop_engine()
        print("✅ System stopped successfully")
    except Exception as e:
        print(f"❌ Error: {e}")
        strategy_engine.stop_engine()

if __name__ == "__main__":
    main()