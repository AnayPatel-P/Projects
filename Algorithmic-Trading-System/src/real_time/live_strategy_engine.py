import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np
import logging

from .paper_trading import PaperTradingAccount, OrderSide, OrderType
from .live_data_feed import LiveDataFeed, TechnicalIndicatorCalculator
from ..strategies.base_strategy import BaseStrategy, SignalType

class LiveStrategyEngine:
    """
    Real-time strategy execution engine with paper trading
    """
    
    def __init__(self, paper_account: PaperTradingAccount):
        self.paper_account = paper_account
        self.data_feed = None
        self.strategies = {}
        self.strategy_states = {}
        self.indicator_calculator = TechnicalIndicatorCalculator()
        
        # Engine state
        self.is_running = False
        self.last_signals = {}
        self.signal_history = []
        self.performance_callbacks = []
        
        # Risk management
        self.max_position_size = 0.1  # 10% max position size
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        self.position_limits = {}
        
        # Logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for the strategy engine"""
        self.logger = logging.getLogger('LiveStrategyEngine')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def set_data_feed(self, data_feed: LiveDataFeed):
        """Set the live data feed"""
        self.data_feed = data_feed
        self.data_feed.subscribe(self._on_market_data)
        # Note: paper_account market data updates handled in _on_market_data method
        
    def add_strategy(self, name: str, strategy: BaseStrategy, symbols: List[str]):
        """Add a trading strategy"""
        self.strategies[name] = {
            'strategy': strategy,
            'symbols': symbols,
            'last_signals': {},
            'signal_count': 0,
            'last_execution': None
        }
        
        self.strategy_states[name] = {
            'active': True,
            'pnl': 0.0,
            'trades': 0,
            'positions': {}
        }
        
        self.logger.info(f"Added strategy: {name} for symbols: {symbols}")
    
    def remove_strategy(self, name: str):
        """Remove a trading strategy"""
        if name in self.strategies:
            del self.strategies[name]
            del self.strategy_states[name]
            self.logger.info(f"Removed strategy: {name}")
    
    def start_engine(self, symbols: List[str]):
        """Start the live trading engine"""
        if self.is_running:
            self.logger.warning("Engine is already running")
            return
        
        if not self.data_feed:
            raise ValueError("Data feed not set. Use set_data_feed() first.")
        
        self.is_running = True
        
        # Start data feed
        self.data_feed.start_feed(symbols, source='simulated')
        
        # Start strategy execution loop
        self.strategy_thread = threading.Thread(target=self._strategy_execution_loop, daemon=True)
        self.strategy_thread.start()
        
        # Start performance monitoring
        self.performance_thread = threading.Thread(target=self._performance_monitoring_loop, daemon=True)
        self.performance_thread.start()
        
        self.logger.info(f"Live strategy engine started for symbols: {symbols}")
    
    def stop_engine(self):
        """Stop the live trading engine"""
        self.is_running = False
        
        if self.data_feed:
            self.data_feed.stop_feed()
        
        self.logger.info("Live strategy engine stopped")
    
    def _on_market_data(self, data: Dict):
        """Handle incoming market data"""
        symbol = data['symbol']
        price = data['price']
        timestamp = data['timestamp']
        
        # Update technical indicators
        self.indicator_calculator.add_price(symbol, price, timestamp)
        
        # Update paper account
        self.paper_account.update_market_data(data)
        
        # Trigger strategy evaluation
        self._evaluate_strategies(symbol, data)
    
    def _evaluate_strategies(self, symbol: str, market_data: Dict):
        """Evaluate all strategies for the given symbol"""
        for strategy_name, strategy_info in self.strategies.items():
            if not self.strategy_states[strategy_name]['active']:
                continue
                
            if symbol not in strategy_info['symbols']:
                continue
            
            try:
                # Get historical data for strategy
                price_history = self.data_feed.get_price_history(symbol, periods=200)
                
                if len(price_history) < 50:  # Need minimum history
                    continue
                
                # Add technical indicators to price history
                indicators = self.indicator_calculator.get_all_indicators(symbol)
                
                # Create market data DataFrame for strategy
                strategy_data = self._prepare_strategy_data(price_history, indicators)
                strategy_data.attrs['symbol'] = symbol
                
                # Generate signals
                signals = strategy_info['strategy'].generate_signals(strategy_data)
                
                # Process new signals
                for signal in signals:
                    if self._is_new_signal(strategy_name, symbol, signal):
                        self._process_signal(strategy_name, signal)
                        
            except Exception as e:
                self.logger.error(f"Error evaluating strategy {strategy_name} for {symbol}: {e}")
    
    def _prepare_strategy_data(self, price_history: pd.DataFrame, indicators: Dict) -> pd.DataFrame:
        """Prepare data format expected by strategies"""
        if price_history.empty:
            return pd.DataFrame()
        
        # Create OHLCV data from price history
        # For real-time, we approximate OHLCV from price points
        data = pd.DataFrame(index=price_history.index)
        
        # Simple approximation - in production, you'd get actual OHLCV bars
        data['Close'] = price_history['price']
        data['Open'] = price_history['price'].shift(1).fillna(price_history['price'])
        data['High'] = price_history['price'].rolling(window=5, min_periods=1).max()
        data['Low'] = price_history['price'].rolling(window=5, min_periods=1).min()
        data['Volume'] = 100000  # Dummy volume
        
        # Add technical indicators
        for indicator, value in indicators.items():
            if indicator != 'timestamp':
                data[indicator] = value
        
        return data.ffill().bfill()
    
    def _is_new_signal(self, strategy_name: str, symbol: str, signal) -> bool:
        """Check if this is a new signal (not duplicate)"""
        last_signal = self.strategies[strategy_name]['last_signals'].get(symbol)
        
        if last_signal is None:
            return True
        
        # Check if signal is different from last signal
        time_diff = (signal.timestamp - last_signal.timestamp).total_seconds()
        
        # Consider it new if:
        # 1. Different signal type, or
        # 2. Same type but sufficient time has passed (avoid spam)
        if (signal.signal != last_signal.signal or 
            time_diff > 300):  # 5 minutes minimum between same signals
            return True
        
        return False
    
    def _process_signal(self, strategy_name: str, signal):
        """Process a trading signal"""
        symbol = signal.symbol
        
        # Update last signal
        self.strategies[strategy_name]['last_signals'][symbol] = signal
        self.strategies[strategy_name]['signal_count'] += 1
        
        # Add to signal history
        self.signal_history.append({
            'strategy': strategy_name,
            'symbol': symbol,
            'signal': signal.signal.name,
            'price': signal.price,
            'confidence': signal.confidence,
            'timestamp': signal.timestamp
        })
        
        # Keep only recent history
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
        
        self.logger.info(
            f"Signal: {strategy_name} - {symbol} - {signal.signal.name} "
            f"@ {signal.price:.2f} (confidence: {signal.confidence:.2f})"
        )
        
        # Execute trade if signal is strong enough and passes risk checks
        if self._should_execute_signal(strategy_name, signal):
            self._execute_signal(strategy_name, signal)
    
    def _should_execute_signal(self, strategy_name: str, signal) -> bool:
        """Determine if signal should be executed based on risk management"""
        
        # Check confidence threshold
        if signal.confidence < 0.5:
            self.logger.debug(f"Signal rejected: low confidence ({signal.confidence:.2f})")
            return False
        
        # Check daily loss limit
        account_summary = self.paper_account.get_account_summary()
        daily_pnl_pct = account_summary['total_pnl'] / account_summary['initial_balance'] * 100
        
        if daily_pnl_pct < -self.daily_loss_limit * 100:
            self.logger.warning(f"Daily loss limit reached ({daily_pnl_pct:.2f}%)")
            return False
        
        # Check position size limits
        current_positions = self.paper_account.get_positions()
        portfolio_value = account_summary['portfolio_value']
        
        symbol_position = current_positions.get(signal.symbol, {})
        current_market_value = symbol_position.get('market_value', 0)
        position_pct = abs(current_market_value) / portfolio_value
        
        if position_pct > self.max_position_size:
            self.logger.debug(f"Position size limit exceeded for {signal.symbol}")
            return False
        
        # Check if we have enough buying power (for buy signals)
        if signal.signal == SignalType.BUY:
            target_position_value = portfolio_value * self.max_position_size * signal.confidence
            required_cash = target_position_value - current_market_value
            
            if required_cash > account_summary['cash_balance']:
                self.logger.debug(f"Insufficient buying power for {signal.symbol}")
                return False
        
        # Check if we have shares to sell (for sell signals)
        if signal.signal == SignalType.SELL:
            current_quantity = symbol_position.get('quantity', 0)
            if current_quantity <= 0:
                self.logger.debug(f"No position to sell for {signal.symbol}")
                return False
        
        return True
    
    def _execute_signal(self, strategy_name: str, signal):
        """Execute a trading signal"""
        try:
            account_summary = self.paper_account.get_account_summary()
            portfolio_value = account_summary['portfolio_value']
            
            if signal.signal == SignalType.BUY:
                # Calculate position size based on confidence and risk management
                target_value = portfolio_value * self.max_position_size * signal.confidence
                quantity = int(target_value / signal.price)
                
                if quantity > 0:
                    order_id = self.paper_account.submit_order(
                        symbol=signal.symbol,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=quantity
                    )
                    
                    self.logger.info(f"BUY order submitted: {signal.symbol} x{quantity} @ market (Order ID: {order_id})")
                    self._update_strategy_state(strategy_name, 'buy', signal.symbol, quantity)
            
            elif signal.signal == SignalType.SELL:
                # Sell current position
                positions = self.paper_account.get_positions()
                current_position = positions.get(signal.symbol, {})
                quantity = current_position.get('quantity', 0)
                
                if quantity > 0:
                    order_id = self.paper_account.submit_order(
                        symbol=signal.symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=quantity
                    )
                    
                    self.logger.info(f"SELL order submitted: {signal.symbol} x{quantity} @ market (Order ID: {order_id})")
                    self._update_strategy_state(strategy_name, 'sell', signal.symbol, quantity)
            
        except Exception as e:
            self.logger.error(f"Error executing signal for {signal.symbol}: {e}")
    
    def _update_strategy_state(self, strategy_name: str, action: str, symbol: str, quantity: int):
        """Update strategy state after trade execution"""
        state = self.strategy_states[strategy_name]
        state['trades'] += 1
        state['last_execution'] = datetime.now()
        
        if symbol not in state['positions']:
            state['positions'][symbol] = 0
        
        if action == 'buy':
            state['positions'][symbol] += quantity
        elif action == 'sell':
            state['positions'][symbol] -= quantity
    
    def _strategy_execution_loop(self):
        """Main strategy execution loop"""
        while self.is_running:
            try:
                # Strategy-specific tasks can be added here
                # For now, just monitor and log status
                
                if len(self.strategies) > 0 and len(self.signal_history) > 0:
                    recent_signals = [s for s in self.signal_history 
                                    if (datetime.now() - s['timestamp']).total_seconds() < 300]
                    
                    if len(recent_signals) > 0:
                        self.logger.debug(f"Recent signals (5min): {len(recent_signals)}")
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in strategy execution loop: {e}")
                time.sleep(5)
    
    def _performance_monitoring_loop(self):
        """Performance monitoring and reporting loop"""
        while self.is_running:
            try:
                # Calculate and log performance metrics
                account_summary = self.paper_account.get_account_summary()
                performance_metrics = self.paper_account.get_performance_metrics()
                
                # Log performance every 5 minutes
                self.logger.info(
                    f"Portfolio: ${account_summary['portfolio_value']:,.2f} "
                    f"({account_summary['total_return_pct']:+.2f}%) | "
                    f"Positions: {performance_metrics.get('current_positions', 0)} | "
                    f"Trades: {performance_metrics.get('total_trades', 0)}"
                )
                
                # Notify performance callbacks
                for callback in self.performance_callbacks:
                    try:
                        callback({
                            'account_summary': account_summary,
                            'performance_metrics': performance_metrics,
                            'strategy_states': self.strategy_states,
                            'recent_signals': self.signal_history[-10:] if self.signal_history else []
                        })
                    except Exception as e:
                        self.logger.error(f"Error in performance callback: {e}")
                
                time.sleep(300)  # Report every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                time.sleep(30)
    
    def add_performance_callback(self, callback: Callable):
        """Add a callback for performance updates"""
        self.performance_callbacks.append(callback)
    
    def get_engine_status(self) -> Dict:
        """Get current engine status"""
        return {
            'is_running': self.is_running,
            'strategies': {
                name: {
                    'active': self.strategy_states[name]['active'],
                    'symbols': info['symbols'],
                    'signal_count': info['signal_count'],
                    'trades': self.strategy_states[name]['trades'],
                    'positions': self.strategy_states[name]['positions']
                }
                for name, info in self.strategies.items()
            },
            'account_summary': self.paper_account.get_account_summary(),
            'recent_signals': self.signal_history[-20:] if self.signal_history else [],
            'timestamp': datetime.now().isoformat()
        }
    
    def activate_strategy(self, strategy_name: str):
        """Activate a strategy"""
        if strategy_name in self.strategy_states:
            self.strategy_states[strategy_name]['active'] = True
            self.logger.info(f"Strategy activated: {strategy_name}")
    
    def deactivate_strategy(self, strategy_name: str):
        """Deactivate a strategy"""
        if strategy_name in self.strategy_states:
            self.strategy_states[strategy_name]['active'] = False
            self.logger.info(f"Strategy deactivated: {strategy_name}")
    
    def reset_strategy_state(self, strategy_name: str):
        """Reset strategy state"""
        if strategy_name in self.strategy_states:
            self.strategy_states[strategy_name] = {
                'active': True,
                'pnl': 0.0,
                'trades': 0,
                'positions': {}
            }
            self.strategies[strategy_name]['signal_count'] = 0
            self.strategies[strategy_name]['last_signals'] = {}
            
            self.logger.info(f"Strategy state reset: {strategy_name}")
    
    def export_performance_data(self) -> Dict:
        """Export comprehensive performance data"""
        return {
            'engine_status': self.get_engine_status(),
            'account_data': self.paper_account.export_performance_data(),
            'signal_history': self.signal_history,
            'strategy_performance': {
                name: {
                    'state': state,
                    'info': self.strategies[name]
                }
                for name, state in self.strategy_states.items()
            }
        }