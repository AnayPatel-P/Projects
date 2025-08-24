import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from enum import Enum
import json

from ..real_time.live_strategy_engine import LiveStrategyEngine
from ..real_time.paper_trading import PaperTradingAccount
from ..real_time.live_data_feed import LiveDataFeed
from ..risk_management.advanced_risk_models import AdvancedRiskModel, RealTimeRiskMonitor
from ..derivatives.options_strategies import OptionsPortfolioManager
from ..portfolio_optimization.portfolio_optimizer import PortfolioOptimizer
from ..ml.models import MLTradingModel
from ..dashboard.live_dashboard import LiveTradingDashboard

class SystemStatus(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"
    SHUTDOWN = "shutdown"

@dataclass
class SystemConfiguration:
    initial_balance: float = 100000.0
    commission_rate: float = 0.001
    risk_free_rate: float = 0.05
    update_interval: int = 5  # seconds
    auto_rebalance: bool = True
    rebalance_interval: int = 3600  # seconds
    enable_options: bool = True
    enable_ml: bool = True
    enable_dashboard: bool = True
    dashboard_port: int = 8050
    max_position_size: float = 0.20  # 20% max per position
    stop_loss_pct: float = 0.05  # 5% stop loss
    symbols: List[str] = field(default_factory=lambda: ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'])

class IntegratedTradingSystem:
    """
    Comprehensive integrated algorithmic trading system
    """
    
    def __init__(self, config: SystemConfiguration):
        self.config = config
        self.status = SystemStatus.STOPPED
        self.start_time = None
        self.components = {}
        self.callbacks = []
        self.performance_history = []
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.initialize_components()
        
    def setup_logging(self):
        """Setup system-wide logging"""
        self.logger = logging.getLogger('IntegratedTradingSystem')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # File handler
            file_handler = logging.FileHandler('trading_system.log')
            file_handler.setLevel(logging.DEBUG)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
    
    def initialize_components(self):
        """Initialize all system components"""
        try:
            self.status = SystemStatus.STARTING
            self.logger.info("Initializing trading system components...")
            
            # 1. Paper Trading Account
            self.paper_account = PaperTradingAccount(
                initial_balance=self.config.initial_balance,
                commission_rate=self.config.commission_rate
            )
            self.components['paper_account'] = self.paper_account
            self.logger.info("✓ Paper trading account initialized")
            
            # 2. Live Data Feed
            self.data_feed = LiveDataFeed()
            self.components['data_feed'] = self.data_feed
            self.logger.info("✓ Live data feed initialized")
            
            # 3. Strategy Engine
            self.strategy_engine = LiveStrategyEngine(self.paper_account)
            self.strategy_engine.set_data_feed(self.data_feed)
            self.strategy_engine.max_position_size = self.config.max_position_size
            self.strategy_engine.daily_loss_limit = self.config.stop_loss_pct
            self.components['strategy_engine'] = self.strategy_engine
            self.logger.info("✓ Strategy engine initialized")
            
            # 4. Risk Management
            self.risk_model = AdvancedRiskModel()
            self.risk_monitor = RealTimeRiskMonitor(self.risk_model)
            self.risk_monitor.add_risk_callback(self._handle_risk_alert)
            self.components['risk_model'] = self.risk_model
            self.components['risk_monitor'] = self.risk_monitor
            self.logger.info("✓ Risk management system initialized")
            
            # 5. Portfolio Optimizer
            self.portfolio_optimizer = PortfolioOptimizer()
            self.components['portfolio_optimizer'] = self.portfolio_optimizer
            self.logger.info("✓ Portfolio optimizer initialized")
            
            # 6. Options Portfolio (if enabled)
            if self.config.enable_options:
                self.options_portfolio = OptionsPortfolioManager()
                self.components['options_portfolio'] = self.options_portfolio
                self.logger.info("✓ Options portfolio manager initialized")
            
            # 7. ML Models (if enabled)
            if self.config.enable_ml:
                self.ml_model = MLTradingModel()
                self.components['ml_model'] = self.ml_model
                self.logger.info("✓ ML trading models initialized")
            
            # 8. Dashboard (if enabled)
            if self.config.enable_dashboard:
                self.dashboard = LiveTradingDashboard(self.strategy_engine)
                self.components['dashboard'] = self.dashboard
                self.logger.info("✓ Live dashboard initialized")
            
            self.logger.info(f"All components initialized successfully ({len(self.components)} components)")
            
        except Exception as e:
            self.status = SystemStatus.ERROR
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    def add_callback(self, callback: Callable):
        """Add system event callback"""
        self.callbacks.append(callback)
    
    def _handle_risk_alert(self, alert):
        """Handle risk management alerts"""
        self.logger.warning(f"Risk Alert: {alert.message} (Severity: {alert.severity})")
        
        # Implement automatic responses to critical alerts
        if alert.severity == 'critical':
            self.logger.critical("Critical risk alert - implementing emergency procedures")
            self._emergency_risk_response()
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback('risk_alert', alert)
            except Exception as e:
                self.logger.error(f"Error in risk alert callback: {e}")
    
    def _emergency_risk_response(self):
        """Emergency risk response procedures"""
        try:
            # Reduce position sizes
            positions = self.paper_account.get_positions()
            for symbol, position in positions.items():
                if position['quantity'] > 0:
                    # Sell 50% of position
                    sell_quantity = position['quantity'] // 2
                    if sell_quantity > 0:
                        self.paper_account.submit_order(
                            symbol=symbol,
                            side='sell',
                            order_type='market',
                            quantity=sell_quantity
                        )
                        self.logger.info(f"Emergency sale: {symbol} x{sell_quantity}")
            
        except Exception as e:
            self.logger.error(f"Error in emergency risk response: {e}")
    
    def add_strategy(self, name: str, strategy, symbols: List[str]):
        """Add trading strategy to the system"""
        if 'strategy_engine' not in self.components:
            raise ValueError("Strategy engine not initialized")
        
        self.strategy_engine.add_strategy(name, strategy, symbols)
        self.logger.info(f"Strategy added: {name} for symbols {symbols}")
    
    def start_system(self):
        """Start the complete trading system"""
        try:
            if self.status == SystemStatus.RUNNING:
                self.logger.warning("System is already running")
                return
            
            self.logger.info("Starting integrated trading system...")
            self.status = SystemStatus.STARTING
            self.start_time = datetime.now()
            
            # Start data feed
            self.data_feed.start_feed(self.config.symbols, source='simulated')
            
            # Start strategy engine
            self.strategy_engine.start_engine(self.config.symbols)
            
            # Start risk monitoring
            self.risk_monitor.start_monitoring(update_interval=60)
            
            # Start dashboard (in background thread)
            if self.config.enable_dashboard:
                self.dashboard_thread = threading.Thread(
                    target=self._run_dashboard,
                    daemon=True
                )
                self.dashboard_thread.start()
            
            # Start main system loop
            self.system_thread = threading.Thread(
                target=self._system_main_loop,
                daemon=True
            )
            self.system_thread.start()
            
            # Start auto-rebalancing (if enabled)
            if self.config.auto_rebalance:
                self.rebalance_thread = threading.Thread(
                    target=self._auto_rebalance_loop,
                    daemon=True
                )
                self.rebalance_thread.start()
            
            self.status = SystemStatus.RUNNING
            self.logger.info("✅ Integrated trading system started successfully")
            
            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback('system_started', {'timestamp': self.start_time})
                except Exception as e:
                    self.logger.error(f"Error in system start callback: {e}")
                    
        except Exception as e:
            self.status = SystemStatus.ERROR
            self.logger.error(f"Error starting system: {e}")
            raise
    
    def stop_system(self):
        """Stop the complete trading system"""
        try:
            self.logger.info("Stopping integrated trading system...")
            self.status = SystemStatus.SHUTDOWN
            
            # Stop all components
            if hasattr(self, 'strategy_engine'):
                self.strategy_engine.stop_engine()
            
            if hasattr(self, 'data_feed'):
                self.data_feed.stop_feed()
            
            if hasattr(self, 'risk_monitor'):
                self.risk_monitor.stop_monitoring()
            
            self.logger.info("✅ Integrated trading system stopped")
            
            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback('system_stopped', {'timestamp': datetime.now()})
                except Exception as e:
                    self.logger.error(f"Error in system stop callback: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error stopping system: {e}")
    
    def _run_dashboard(self):
        """Run dashboard in background thread"""
        try:
            self.dashboard.run_server(
                debug=False,
                port=self.config.dashboard_port,
                host='127.0.0.1'
            )
        except Exception as e:
            self.logger.error(f"Dashboard error: {e}")
    
    def _system_main_loop(self):
        """Main system monitoring and coordination loop"""
        while self.status == SystemStatus.RUNNING:
            try:
                # Update risk monitoring with current positions
                positions = self.paper_account.get_positions()
                self.risk_monitor.update_positions(positions)
                
                # Update performance history
                account_summary = self.paper_account.get_account_summary()
                self.performance_history.append({
                    'timestamp': datetime.now(),
                    'portfolio_value': account_summary['portfolio_value'],
                    'total_return_pct': account_summary['total_return_pct'],
                    'cash_balance': account_summary['cash_balance'],
                    'num_positions': len(positions)
                })
                
                # Keep only recent history
                if len(self.performance_history) > 1000:
                    self.performance_history = self.performance_history[-1000:]
                
                # Log system status periodically
                if len(self.performance_history) % 12 == 0:  # Every minute (assuming 5-second intervals)
                    self._log_system_status()
                
                time.sleep(self.config.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in main system loop: {e}")
                time.sleep(self.config.update_interval)
    
    def _auto_rebalance_loop(self):
        """Automatic portfolio rebalancing loop"""
        while self.status == SystemStatus.RUNNING:
            try:
                # Wait for rebalance interval
                time.sleep(self.config.rebalance_interval)
                
                if self.status != SystemStatus.RUNNING:
                    break
                
                # Perform rebalancing
                self._perform_rebalancing()
                
            except Exception as e:
                self.logger.error(f"Error in auto-rebalance loop: {e}")
    
    def _perform_rebalancing(self):
        """Perform portfolio rebalancing"""
        try:
            self.logger.info("Performing portfolio rebalancing...")
            
            # Get current positions and historical returns
            positions = self.paper_account.get_positions()
            
            if not positions:
                self.logger.info("No positions to rebalance")
                return
            
            # Create returns DataFrame for optimization
            # In a real implementation, you would fetch historical price data
            symbols = list(positions.keys())
            
            # For demo, use the recent performance history to estimate returns
            if len(self.performance_history) > 50:
                # Calculate target weights using portfolio optimizer
                # This is a simplified example
                equal_weights = {symbol: 1.0/len(symbols) for symbol in symbols}
                
                self.logger.info(f"Target allocation: {equal_weights}")
                
                # Execute rebalancing trades (simplified)
                account_summary = self.paper_account.get_account_summary()
                total_value = account_summary['portfolio_value']
                
                for symbol, target_weight in equal_weights.items():
                    current_position = positions.get(symbol, {})
                    current_value = current_position.get('market_value', 0)
                    target_value = total_value * target_weight
                    
                    difference = target_value - current_value
                    
                    # Only rebalance if difference is significant
                    if abs(difference) > total_value * 0.02:  # 2% threshold
                        self.logger.info(f"Rebalancing {symbol}: ${current_value:.2f} -> ${target_value:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error performing rebalancing: {e}")
    
    def _log_system_status(self):
        """Log current system status"""
        try:
            account_summary = self.paper_account.get_account_summary()
            engine_status = self.strategy_engine.get_engine_status()
            
            uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
            
            self.logger.info(f"System Status - Uptime: {uptime}, "
                           f"Portfolio: ${account_summary['portfolio_value']:,.2f} "
                           f"({account_summary['total_return_pct']:+.2f}%), "
                           f"Strategies: {len(engine_status['strategies'])}, "
                           f"Recent Signals: {len(engine_status['recent_signals'])}")
                           
        except Exception as e:
            self.logger.error(f"Error logging system status: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status_data = {
                'system_status': self.status.value,
                'uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                'components': {
                    name: 'active' if hasattr(component, 'is_running') and component.is_running
                    else 'initialized' for name, component in self.components.items()
                },
                'configuration': {
                    'initial_balance': self.config.initial_balance,
                    'symbols': self.config.symbols,
                    'auto_rebalance': self.config.auto_rebalance,
                    'enable_options': self.config.enable_options,
                    'enable_ml': self.config.enable_ml,
                    'enable_dashboard': self.config.enable_dashboard
                }
            }
            
            # Add component-specific status
            if hasattr(self, 'paper_account'):
                account_summary = self.paper_account.get_account_summary()
                status_data['account'] = account_summary
            
            if hasattr(self, 'strategy_engine'):
                engine_status = self.strategy_engine.get_engine_status()
                status_data['strategies'] = engine_status
            
            if hasattr(self, 'performance_history') and self.performance_history:
                latest_performance = self.performance_history[-1]
                status_data['performance'] = latest_performance
            
            return status_data
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'system_status': 'error', 'error': str(e)}
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            if not hasattr(self, 'paper_account'):
                return {}
            
            # Get account and performance data
            account_summary = self.paper_account.get_account_summary()
            performance_data = self.paper_account.export_performance_data()
            
            # Calculate additional metrics
            if self.performance_history and len(self.performance_history) > 1:
                values = [p['portfolio_value'] for p in self.performance_history]
                returns = pd.Series(values).pct_change().dropna()
                
                # Risk-adjusted metrics
                if len(returns) > 30:
                    portfolio_metrics = self.risk_model.calculate_risk_adjusted_metrics(returns)
                else:
                    portfolio_metrics = {}
                
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'system_uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                    'account_summary': account_summary,
                    'performance_metrics': portfolio_metrics,
                    'performance_data': performance_data,
                    'system_statistics': {
                        'total_strategies': len(self.strategy_engine.strategies) if hasattr(self, 'strategy_engine') else 0,
                        'data_points': len(self.performance_history),
                        'components_active': len(self.components)
                    }
                }
                
                return report
            
            return {
                'timestamp': datetime.now().isoformat(),
                'account_summary': account_summary,
                'message': 'Insufficient data for comprehensive analysis'
            }
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def save_system_state(self, filepath: str):
        """Save current system state to file"""
        try:
            state_data = {
                'timestamp': datetime.now().isoformat(),
                'configuration': self.config.__dict__,
                'system_status': self.get_system_status(),
                'performance_history': self.performance_history[-100:],  # Save last 100 points
            }
            
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            self.logger.info(f"System state saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving system state: {e}")
    
    def load_system_state(self, filepath: str):
        """Load system state from file"""
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)
            
            # Restore performance history
            if 'performance_history' in state_data:
                self.performance_history = state_data['performance_history']
            
            self.logger.info(f"System state loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading system state: {e}")

def create_default_system(symbols: Optional[List[str]] = None) -> IntegratedTradingSystem:
    """Create a default integrated trading system"""
    if symbols is None:
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    config = SystemConfiguration(
        initial_balance=100000.0,
        symbols=symbols,
        enable_dashboard=True,
        auto_rebalance=True,
        rebalance_interval=3600  # 1 hour
    )
    
    return IntegratedTradingSystem(config)

def run_integrated_demo():
    """Run a complete demonstration of the integrated system"""
    from ..strategies.moving_average_strategy import MovingAverageStrategy
    from ..strategies.rsi_strategy import RSIStrategy
    
    # Create system
    system = create_default_system()
    
    # Add strategies
    ma_strategy = MovingAverageStrategy(short_window=10, long_window=30)
    rsi_strategy = RSIStrategy(period=14, oversold_threshold=30, overbought_threshold=70)
    
    system.add_strategy('MA_10_30', ma_strategy, ['AAPL', 'GOOGL', 'MSFT'])
    system.add_strategy('RSI_14', rsi_strategy, ['TSLA', 'NVDA'])
    
    # Add system monitoring callback
    def system_monitor(event_type, data):
        print(f"System Event: {event_type} - {data}")
    
    system.add_callback(system_monitor)
    
    try:
        # Start the system
        system.start_system()
        
        print("Integrated trading system started successfully!")
        print(f"Dashboard available at: http://localhost:{system.config.dashboard_port}")
        print("Press Ctrl+C to stop the system")
        
        # Run for demonstration
        while True:
            time.sleep(30)
            status = system.get_system_status()
            print(f"\nSystem Status: {status['system_status']}")
            if 'account' in status:
                print(f"Portfolio Value: ${status['account']['portfolio_value']:,.2f}")
                print(f"Total Return: {status['account']['total_return_pct']:+.2f}%")
    
    except KeyboardInterrupt:
        print("\nShutting down system...")
        system.stop_system()
        print("System shutdown complete")

if __name__ == "__main__":
    run_integrated_demo()