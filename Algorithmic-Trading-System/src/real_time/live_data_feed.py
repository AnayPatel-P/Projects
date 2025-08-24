import asyncio
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Optional
import threading
import time
from datetime import datetime, timedelta
import queue
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class LiveDataFeed:
    """
    Real-time data feed manager with multiple data sources
    """
    
    def __init__(self):
        self.subscribers = []
        self.is_running = False
        self.data_queue = queue.Queue()
        self.current_prices = {}
        self.price_history = {}
        self.connection_status = {}
        
    def subscribe(self, callback: Callable):
        """Subscribe to real-time price updates"""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from real-time updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def _notify_subscribers(self, data: Dict):
        """Notify all subscribers of new data"""
        for callback in self.subscribers:
            try:
                callback(data)
            except Exception as e:
                print(f"Error notifying subscriber: {e}")
    
    def start_feed(self, symbols: List[str], source: str = 'yahoo'):
        """Start the real-time data feed"""
        if self.is_running:
            print("Data feed is already running")
            return
        
        self.is_running = True
        
        if source == 'yahoo':
            self._start_yahoo_feed(symbols)
        elif source == 'websocket':
            self._start_websocket_feed(symbols)
        else:
            self._start_simulated_feed(symbols)
    
    def stop_feed(self):
        """Stop the real-time data feed"""
        self.is_running = False
        print("Data feed stopped")
    
    def _start_yahoo_feed(self, symbols: List[str]):
        """Start Yahoo Finance real-time feed (polling-based)"""
        
        def yahoo_worker():
            while self.is_running:
                try:
                    for symbol in symbols:
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        
                        if 'regularMarketPrice' in info:
                            price = info['regularMarketPrice']
                            volume = info.get('regularMarketVolume', 0)
                            
                            data = {
                                'symbol': symbol,
                                'price': price,
                                'volume': volume,
                                'timestamp': datetime.now(),
                                'source': 'yahoo'
                            }
                            
                            self.current_prices[symbol] = price
                            self._update_price_history(symbol, price)
                            self._notify_subscribers(data)
                    
                    time.sleep(1)  # Update every second
                    
                except Exception as e:
                    print(f"Yahoo feed error: {e}")
                    time.sleep(5)  # Wait before retrying
        
        thread = threading.Thread(target=yahoo_worker, daemon=True)
        thread.start()
        print(f"Started Yahoo Finance feed for {symbols}")
    
    def _start_websocket_feed(self, symbols: List[str]):
        """Start WebSocket-based real-time feed (demo implementation)"""
        
        def websocket_worker():
            # This is a simplified WebSocket implementation
            # In production, you'd connect to a real WebSocket API like Alpaca, IEX, etc.
            
            while self.is_running:
                try:
                    # Simulate WebSocket data
                    for symbol in symbols:
                        # Generate realistic price movements
                        if symbol in self.current_prices:
                            last_price = self.current_prices[symbol]
                            # Random walk with mean reversion
                            change = np.random.normal(0, 0.001) * last_price
                            new_price = last_price + change
                        else:
                            # Initialize with a reasonable price
                            new_price = 100 + np.random.normal(0, 10)
                        
                        data = {
                            'symbol': symbol,
                            'price': new_price,
                            'volume': np.random.randint(1000, 10000),
                            'timestamp': datetime.now(),
                            'source': 'websocket'
                        }
                        
                        self.current_prices[symbol] = new_price
                        self._update_price_history(symbol, new_price)
                        self._notify_subscribers(data)
                    
                    time.sleep(0.1)  # Update every 100ms
                    
                except Exception as e:
                    print(f"WebSocket feed error: {e}")
                    time.sleep(1)
        
        thread = threading.Thread(target=websocket_worker, daemon=True)
        thread.start()
        print(f"Started WebSocket feed for {symbols}")
    
    def _start_simulated_feed(self, symbols: List[str]):
        """Start simulated real-time feed for demo purposes"""
        
        def simulation_worker():
            # Initialize prices
            for symbol in symbols:
                self.current_prices[symbol] = 100 + np.random.normal(0, 20)
            
            while self.is_running:
                try:
                    for symbol in symbols:
                        # Generate realistic intraday price movement
                        last_price = self.current_prices[symbol]
                        
                        # Add market hours logic
                        now = datetime.now()
                        is_market_hours = 9.5 <= now.hour + now.minute/60 <= 16
                        
                        if is_market_hours:
                            # Normal trading volatility
                            volatility = 0.002
                            volume_base = 5000
                        else:
                            # After-hours reduced volatility
                            volatility = 0.0005
                            volume_base = 1000
                        
                        # Price movement with mean reversion
                        drift = -0.00001 * (last_price - 100)  # Mean revert to 100
                        noise = np.random.normal(0, volatility) * last_price
                        new_price = max(last_price + drift + noise, 0.01)  # Prevent negative prices
                        
                        # Volume with realistic patterns
                        volume = max(int(volume_base * (1 + np.random.exponential(0.5))), 100)
                        
                        data = {
                            'symbol': symbol,
                            'price': new_price,
                            'bid': new_price * 0.9995,
                            'ask': new_price * 1.0005,
                            'volume': volume,
                            'timestamp': datetime.now(),
                            'source': 'simulated',
                            'market_hours': is_market_hours
                        }
                        
                        self.current_prices[symbol] = new_price
                        self._update_price_history(symbol, new_price)
                        self._notify_subscribers(data)
                    
                    time.sleep(0.5)  # Update every 500ms
                    
                except Exception as e:
                    print(f"Simulation feed error: {e}")
                    time.sleep(1)
        
        thread = threading.Thread(target=simulation_worker, daemon=True)
        thread.start()
        print(f"Started simulated feed for {symbols}")
    
    def _update_price_history(self, symbol: str, price: float):
        """Update price history for technical analysis"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append({
            'price': price,
            'timestamp': datetime.now()
        })
        
        # Keep only last 1000 prices
        if len(self.price_history[symbol]) > 1000:
            self.price_history[symbol] = self.price_history[symbol][-1000:]
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        return self.current_prices.get(symbol)
    
    def get_price_history(self, symbol: str, periods: int = 100) -> pd.DataFrame:
        """Get recent price history as DataFrame"""
        if symbol not in self.price_history:
            return pd.DataFrame()
        
        history = self.price_history[symbol][-periods:]
        
        if not history:
            return pd.DataFrame()
        
        df = pd.DataFrame(history)
        df.set_index('timestamp', inplace=True)
        return df

class MarketDataManager:
    """
    Manages multiple data feeds and provides unified interface
    """
    
    def __init__(self):
        self.feeds = {}
        self.aggregated_data = {}
        self.subscribers = []
        
    def add_feed(self, name: str, feed: LiveDataFeed):
        """Add a data feed"""
        self.feeds[name] = feed
        feed.subscribe(self._on_data_update)
    
    def remove_feed(self, name: str):
        """Remove a data feed"""
        if name in self.feeds:
            self.feeds[name].stop_feed()
            del self.feeds[name]
    
    def _on_data_update(self, data: Dict):
        """Handle data update from any feed"""
        symbol = data['symbol']
        
        # Update aggregated data
        if symbol not in self.aggregated_data:
            self.aggregated_data[symbol] = {}
        
        self.aggregated_data[symbol].update(data)
        
        # Notify subscribers
        for callback in self.subscribers:
            try:
                callback(data)
            except Exception as e:
                print(f"Error in market data callback: {e}")
    
    def subscribe(self, callback: Callable):
        """Subscribe to market data updates"""
        self.subscribers.append(callback)
    
    def get_current_data(self, symbol: str) -> Optional[Dict]:
        """Get current market data for symbol"""
        return self.aggregated_data.get(symbol)
    
    def get_all_current_data(self) -> Dict:
        """Get current data for all symbols"""
        return self.aggregated_data.copy()
    
    def start_all_feeds(self):
        """Start all registered feeds"""
        for name, feed in self.feeds.items():
            if not feed.is_running:
                print(f"Starting feed: {name}")
                # Note: symbols need to be passed when starting
    
    def stop_all_feeds(self):
        """Stop all registered feeds"""
        for name, feed in self.feeds.items():
            feed.stop_feed()

class OrderBookSimulator:
    """
    Simulate realistic order book for demo purposes
    """
    
    def __init__(self, symbol: str, base_price: float = 100):
        self.symbol = symbol
        self.base_price = base_price
        self.bids = {}  # price -> quantity
        self.asks = {}  # price -> quantity
        self.last_price = base_price
        self.update_order_book()
    
    def update_order_book(self):
        """Generate realistic bid/ask spread and depth"""
        spread = self.last_price * 0.001  # 0.1% spread
        
        # Clear old orders
        self.bids = {}
        self.asks = {}
        
        # Generate bid side (below current price)
        for i in range(10):
            price = self.last_price - spread/2 - (i * spread/10)
            quantity = np.random.randint(100, 1000) * (10 - i)  # More quantity at better prices
            self.bids[round(price, 2)] = quantity
        
        # Generate ask side (above current price)
        for i in range(10):
            price = self.last_price + spread/2 + (i * spread/10)
            quantity = np.random.randint(100, 1000) * (10 - i)
            self.asks[round(price, 2)] = quantity
    
    def get_best_bid_ask(self) -> tuple:
        """Get best bid and ask prices"""
        if self.bids and self.asks:
            best_bid = max(self.bids.keys())
            best_ask = min(self.asks.keys())
            return best_bid, best_ask
        return None, None
    
    def get_order_book_snapshot(self) -> Dict:
        """Get current order book snapshot"""
        return {
            'symbol': self.symbol,
            'bids': sorted(self.bids.items(), reverse=True)[:5],  # Top 5 bids
            'asks': sorted(self.asks.items())[:5],  # Top 5 asks
            'timestamp': datetime.now()
        }
    
    def update_price(self, new_price: float):
        """Update last price and regenerate order book"""
        self.last_price = new_price
        self.update_order_book()

class TechnicalIndicatorCalculator:
    """
    Calculate technical indicators in real-time
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.price_buffers = {}
        
    def add_price(self, symbol: str, price: float, timestamp: datetime = None):
        """Add new price data point"""
        if timestamp is None:
            timestamp = datetime.now()
            
        if symbol not in self.price_buffers:
            self.price_buffers[symbol] = []
        
        self.price_buffers[symbol].append({
            'price': price,
            'timestamp': timestamp
        })
        
        # Keep only recent data
        if len(self.price_buffers[symbol]) > self.window_size:
            self.price_buffers[symbol] = self.price_buffers[symbol][-self.window_size:]
    
    def calculate_sma(self, symbol: str, period: int = 20) -> Optional[float]:
        """Calculate Simple Moving Average"""
        if symbol not in self.price_buffers:
            return None
        
        prices = [p['price'] for p in self.price_buffers[symbol]]
        
        if len(prices) < period:
            return None
        
        return sum(prices[-period:]) / period
    
    def calculate_ema(self, symbol: str, period: int = 20) -> Optional[float]:
        """Calculate Exponential Moving Average"""
        if symbol not in self.price_buffers:
            return None
        
        prices = [p['price'] for p in self.price_buffers[symbol]]
        
        if len(prices) < period:
            return None
        
        # Simple EMA calculation
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def calculate_rsi(self, symbol: str, period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index"""
        if symbol not in self.price_buffers:
            return None
        
        prices = [p['price'] for p in self.price_buffers[symbol]]
        
        if len(prices) < period + 1:
            return None
        
        # Calculate price changes
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Separate gains and losses
        gains = [change if change > 0 else 0 for change in changes[-period:]]
        losses = [-change if change < 0 else 0 for change in changes[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_all_indicators(self, symbol: str) -> Dict:
        """Get all available indicators for symbol"""
        indicators = {
            'sma_20': self.calculate_sma(symbol, 20),
            'sma_50': self.calculate_sma(symbol, 50),
            'ema_12': self.calculate_ema(symbol, 12),
            'ema_26': self.calculate_ema(symbol, 26),
            'rsi_14': self.calculate_rsi(symbol, 14),
            'timestamp': datetime.now()
        }
        
        return {k: v for k, v in indicators.items() if v is not None}