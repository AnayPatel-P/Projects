import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'timestamp': self.timestamp.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }

@dataclass
class Position:
    symbol: str
    quantity: int = 0
    avg_cost: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_market_value(self, current_price: float):
        """Update market value and unrealized PnL"""
        self.market_value = self.quantity * current_price
        if self.quantity != 0:
            self.unrealized_pnl = (current_price - self.avg_cost) * self.quantity
        else:
            self.unrealized_pnl = 0.0
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_cost': self.avg_cost,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'last_updated': self.last_updated.isoformat()
        }

@dataclass
class Trade:
    id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: datetime = field(default_factory=datetime.now)
    commission: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'commission': self.commission
        }

class PaperTradingAccount:
    """
    Paper trading account with realistic order execution simulation
    """
    
    def __init__(self, initial_balance: float = 100000, commission_rate: float = 0.001):
        self.account_id = str(uuid.uuid4())
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance
        self.commission_rate = commission_rate
        
        # Trading state
        self.positions = {}  # symbol -> Position
        self.orders = {}     # order_id -> Order
        self.trades = {}     # trade_id -> Trade
        self.order_history = []
        self.trade_history = []
        
        # Performance tracking
        self.portfolio_value_history = []
        self.daily_pnl_history = []
        
        # Market data subscription
        self.current_prices = {}
        self.market_data_callbacks = []
        
    def get_account_summary(self) -> Dict:
        """Get current account summary"""
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        
        portfolio_value = self.cash_balance + total_market_value
        
        return {
            'account_id': self.account_id,
            'cash_balance': self.cash_balance,
            'market_value': total_market_value,
            'portfolio_value': portfolio_value,
            'unrealized_pnl': total_unrealized_pnl,
            'realized_pnl': total_realized_pnl,
            'total_pnl': total_unrealized_pnl + total_realized_pnl,
            'initial_balance': self.initial_balance,
            'total_return_pct': ((portfolio_value / self.initial_balance) - 1) * 100,
            'buying_power': self.cash_balance,  # Simplified - no margin
            'timestamp': datetime.now().isoformat()
        }
    
    def update_market_data(self, data: Dict):
        """Update current market prices"""
        symbol = data['symbol']
        price = data['price']
        
        self.current_prices[symbol] = price
        
        # Update position market values
        if symbol in self.positions:
            self.positions[symbol].update_market_value(price)
        
        # Check for order triggers
        self._check_order_triggers(symbol, price)
        
        # Record portfolio value
        self._record_portfolio_value()
    
    def submit_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                    quantity: int, price: Optional[float] = None,
                    stop_price: Optional[float] = None) -> str:
        """Submit a trading order"""
        
        # Validate order
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and price is None:
            raise ValueError(f"{order_type.value} orders require a price")
        
        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and stop_price is None:
            raise ValueError(f"{order_type.value} orders require a stop price")
        
        # Check buying power for buy orders
        if side == OrderSide.BUY:
            required_cash = quantity * (price or self.current_prices.get(symbol, 0))
            if required_cash > self.cash_balance:
                raise ValueError("Insufficient buying power")
        
        # Check position for sell orders
        if side == OrderSide.SELL:
            current_position = self.positions.get(symbol, Position(symbol)).quantity
            if quantity > current_position:
                raise ValueError("Insufficient shares to sell")
        
        # Create order
        order_id = str(uuid.uuid4())
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )
        
        self.orders[order_id] = order
        self.order_history.append(order)
        
        # Attempt immediate execution for market orders
        if order_type == OrderType.MARKET:
            self._execute_market_order(order)
        
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.status != OrderStatus.PENDING:
            return False
        
        order.status = OrderStatus.CANCELLED
        order.last_updated = datetime.now()
        
        return True
    
    def _execute_market_order(self, order: Order):
        """Execute a market order immediately"""
        current_price = self.current_prices.get(order.symbol)
        
        if current_price is None:
            order.status = OrderStatus.REJECTED
            return
        
        # Add realistic slippage
        slippage = 0.001  # 0.1%
        if order.side == OrderSide.BUY:
            execution_price = current_price * (1 + slippage)
        else:
            execution_price = current_price * (1 - slippage)
        
        self._fill_order(order, order.quantity, execution_price)
    
    def _check_order_triggers(self, symbol: str, current_price: float):
        """Check if any orders should be triggered"""
        symbol_orders = [o for o in self.orders.values() 
                        if o.symbol == symbol and o.status == OrderStatus.PENDING]
        
        for order in symbol_orders:
            should_trigger = False
            
            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and current_price <= order.price:
                    should_trigger = True
                elif order.side == OrderSide.SELL and current_price >= order.price:
                    should_trigger = True
            
            elif order.order_type == OrderType.STOP:
                if order.side == OrderSide.BUY and current_price >= order.stop_price:
                    should_trigger = True
                elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                    should_trigger = True
            
            elif order.order_type == OrderType.STOP_LIMIT:
                if order.side == OrderSide.BUY and current_price >= order.stop_price:
                    # Convert to limit order
                    order.order_type = OrderType.LIMIT
                elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                    # Convert to limit order
                    order.order_type = OrderType.LIMIT
            
            if should_trigger:
                if order.order_type == OrderType.LIMIT:
                    self._fill_order(order, order.quantity, order.price)
                else:  # STOP order becomes market order
                    self._fill_order(order, order.quantity, current_price)
    
    def _fill_order(self, order: Order, quantity: int, price: float):
        """Fill an order completely or partially"""
        commission = quantity * price * self.commission_rate
        
        # Create trade
        trade_id = str(uuid.uuid4())
        trade = Trade(
            id=trade_id,
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            commission=commission
        )
        
        self.trades[trade_id] = trade
        self.trade_history.append(trade)
        
        # Update order
        order.filled_quantity += quantity
        order.avg_fill_price = (order.avg_fill_price * (order.filled_quantity - quantity) + 
                              price * quantity) / order.filled_quantity
        
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        order.last_updated = datetime.now()
        
        # Update position
        self._update_position(order.symbol, order.side, quantity, price, commission)
        
        # Update cash balance
        if order.side == OrderSide.BUY:
            self.cash_balance -= (quantity * price + commission)
        else:
            self.cash_balance += (quantity * price - commission)
    
    def _update_position(self, symbol: str, side: OrderSide, quantity: int, 
                        price: float, commission: float):
        """Update position after trade execution"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        
        position = self.positions[symbol]
        
        if side == OrderSide.BUY:
            # Calculate new average cost
            total_cost = position.quantity * position.avg_cost + quantity * price
            position.quantity += quantity
            position.avg_cost = total_cost / position.quantity if position.quantity > 0 else 0
            
        else:  # SELL
            # Calculate realized PnL
            realized_pnl = (price - position.avg_cost) * quantity - commission
            position.realized_pnl += realized_pnl
            position.quantity -= quantity
            
            if position.quantity == 0:
                position.avg_cost = 0
        
        # Update market value
        current_price = self.current_prices.get(symbol, price)
        position.update_market_value(current_price)
    
    def _record_portfolio_value(self):
        """Record current portfolio value for performance tracking"""
        summary = self.get_account_summary()
        
        self.portfolio_value_history.append({
            'timestamp': datetime.now(),
            'portfolio_value': summary['portfolio_value'],
            'cash_balance': summary['cash_balance'],
            'market_value': summary['market_value'],
            'unrealized_pnl': summary['unrealized_pnl']
        })
        
        # Keep only last 1000 records
        if len(self.portfolio_value_history) > 1000:
            self.portfolio_value_history = self.portfolio_value_history[-1000:]
    
    def get_positions(self) -> Dict[str, Dict]:
        """Get all current positions"""
        return {symbol: pos.to_dict() for symbol, pos in self.positions.items() 
                if pos.quantity != 0}
    
    def get_orders(self, status_filter: Optional[OrderStatus] = None) -> List[Dict]:
        """Get orders, optionally filtered by status"""
        orders = list(self.orders.values())
        
        if status_filter:
            orders = [o for o in orders if o.status == status_filter]
        
        return [order.to_dict() for order in sorted(orders, 
                key=lambda x: x.timestamp, reverse=True)]
    
    def get_trades(self, limit: Optional[int] = 100) -> List[Dict]:
        """Get trade history"""
        trades = sorted(self.trade_history, key=lambda x: x.timestamp, reverse=True)
        
        if limit:
            trades = trades[:limit]
        
        return [trade.to_dict() for trade in trades]
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if len(self.portfolio_value_history) < 2:
            return {}
        
        values = [p['portfolio_value'] for p in self.portfolio_value_history]
        timestamps = [p['timestamp'] for p in self.portfolio_value_history]
        
        # Convert to pandas for easier calculation
        df = pd.DataFrame({
            'value': values,
            'timestamp': timestamps
        }).set_index('timestamp')
        
        # Calculate returns
        df['returns'] = df['value'].pct_change()
        
        # Performance metrics
        total_return = (values[-1] / self.initial_balance - 1) * 100
        
        if len(df['returns'].dropna()) > 1:
            volatility = df['returns'].std() * np.sqrt(252 * 24 * 60)  # Assuming minute-level data
            sharpe_ratio = df['returns'].mean() / df['returns'].std() * np.sqrt(252 * 24 * 60) if df['returns'].std() != 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
        
        # Drawdown calculation
        cumulative = df['value'] / df['value'].expanding().max()
        drawdown = (1 - cumulative) * 100
        max_drawdown = drawdown.max()
        
        return {
            'total_return_pct': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'total_trades': len(self.trade_history),
            'winning_trades': len([t for t in self.trade_history if self._is_winning_trade(t)]),
            'current_positions': len([p for p in self.positions.values() if p.quantity != 0]),
            'cash_utilization': (self.initial_balance - self.cash_balance) / self.initial_balance * 100
        }
    
    def _is_winning_trade(self, trade: Trade) -> bool:
        """Determine if a trade was winning (simplified)"""
        # This is a simplified check - in reality, you'd need to track the full trade lifecycle
        position = self.positions.get(trade.symbol)
        if position:
            return position.realized_pnl > 0
        return False
    
    def reset_account(self):
        """Reset account to initial state"""
        self.cash_balance = self.initial_balance
        self.positions = {}
        self.orders = {}
        self.trades = {}
        self.order_history = []
        self.trade_history = []
        self.portfolio_value_history = []
        self.daily_pnl_history = []
        
        print("Paper trading account reset")
    
    def export_performance_data(self) -> Dict:
        """Export all performance data for analysis"""
        return {
            'account_summary': self.get_account_summary(),
            'positions': self.get_positions(),
            'orders': self.get_orders(),
            'trades': self.get_trades(),
            'portfolio_history': self.portfolio_value_history,
            'performance_metrics': self.get_performance_metrics()
        }