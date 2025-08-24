import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    entry_date: pd.Timestamp
    position_type: str  # 'long' or 'short'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    position_type: str
    pnl: float
    pnl_pct: float
    commission: float = 0.0
    duration: int = 0

@dataclass
class BacktestResults:
    trades: List[Trade] = field(default_factory=list)
    portfolio_values: pd.Series = field(default_factory=pd.Series)
    positions: Dict[str, Position] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([{
            'Symbol': trade.symbol,
            'Entry_Date': trade.entry_date,
            'Exit_Date': trade.exit_date,
            'Entry_Price': trade.entry_price,
            'Exit_Price': trade.exit_price,
            'Quantity': trade.quantity,
            'Position_Type': trade.position_type,
            'PnL': trade.pnl,
            'PnL_Pct': trade.pnl_pct,
            'Commission': trade.commission,
            'Duration_Days': trade.duration
        } for trade in self.trades])

class BacktestEngine:
    def __init__(self, 
                 initial_capital: float = 100000,
                 commission_rate: float = 0.001,
                 slippage: float = 0.0001,
                 max_position_size: float = 0.1):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.max_position_size = max_position_size
        
        # State variables
        self.cash = initial_capital
        self.positions = {}
        self.portfolio_values = []
        self.trades = []
        self.current_date = None
        
    def reset(self):
        self.cash = self.initial_capital
        self.positions = {}
        self.portfolio_values = []
        self.trades = []
        self.current_date = None
    
    def calculate_position_size(self, price: float, confidence: float = 1.0) -> float:
        max_position_value = self.cash * self.max_position_size * confidence
        shares = int(max_position_value / price)
        return max(shares, 0)
    
    def calculate_portfolio_value(self, market_data: Dict[str, pd.DataFrame]) -> float:
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in market_data and self.current_date in market_data[symbol].index:
                current_price = market_data[symbol].loc[self.current_date, 'Close']
                position_value = position.quantity * current_price
                total_value += position_value
        
        return total_value
    
    def execute_trade(self, symbol: str, quantity: float, price: float, 
                     trade_type: str, date: pd.Timestamp):
        # Apply slippage
        if trade_type == 'buy':
            execution_price = price * (1 + self.slippage)
        else:
            execution_price = price * (1 - self.slippage)
        
        # Calculate commission
        commission = abs(quantity * execution_price * self.commission_rate)
        total_cost = abs(quantity * execution_price) + commission
        
        if trade_type == 'buy':
            if total_cost <= self.cash:
                self.cash -= total_cost
                
                if symbol in self.positions:
                    # Average in if same direction
                    existing_pos = self.positions[symbol]
                    if existing_pos.quantity > 0:  # Same direction (long)
                        total_quantity = existing_pos.quantity + quantity
                        avg_price = ((existing_pos.quantity * existing_pos.entry_price) + 
                                   (quantity * execution_price)) / total_quantity
                        self.positions[symbol].quantity = total_quantity
                        self.positions[symbol].entry_price = avg_price
                    else:
                        # Opposite direction - close existing and open new
                        self._close_position(symbol, execution_price, date)
                        if quantity > abs(existing_pos.quantity):
                            remaining_quantity = quantity - abs(existing_pos.quantity)
                            self.positions[symbol] = Position(
                                symbol=symbol,
                                quantity=remaining_quantity,
                                entry_price=execution_price,
                                entry_date=date,
                                position_type='long'
                            )
                else:
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=quantity,
                        entry_price=execution_price,
                        entry_date=date,
                        position_type='long'
                    )
                    
        elif trade_type == 'sell':
            if symbol in self.positions:
                existing_pos = self.positions[symbol]
                
                if existing_pos.quantity >= quantity:
                    # Partial or full close of long position
                    pnl = quantity * (execution_price - existing_pos.entry_price) - commission
                    
                    trade = Trade(
                        symbol=symbol,
                        entry_date=existing_pos.entry_date,
                        exit_date=date,
                        entry_price=existing_pos.entry_price,
                        exit_price=execution_price,
                        quantity=quantity,
                        position_type='long',
                        pnl=pnl,
                        pnl_pct=(execution_price / existing_pos.entry_price - 1) * 100,
                        commission=commission,
                        duration=(date - existing_pos.entry_date).days
                    )
                    self.trades.append(trade)
                    
                    self.cash += quantity * execution_price - commission
                    
                    if existing_pos.quantity == quantity:
                        del self.positions[symbol]
                    else:
                        self.positions[symbol].quantity -= quantity
                        
                elif existing_pos.quantity < 0:
                    # Adding to short position
                    self.positions[symbol].quantity -= quantity
                    avg_price = ((abs(existing_pos.quantity) * existing_pos.entry_price) + 
                               (quantity * execution_price)) / (abs(existing_pos.quantity) + quantity)
                    self.positions[symbol].entry_price = avg_price
                    
            else:
                # Open short position
                self.cash += quantity * execution_price - commission
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=-quantity,
                    entry_price=execution_price,
                    entry_date=date,
                    position_type='short'
                )
    
    def _close_position(self, symbol: str, price: float, date: pd.Timestamp):
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        commission = abs(position.quantity * price * self.commission_rate)
        
        if position.quantity > 0:  # Close long position
            pnl = position.quantity * (price - position.entry_price) - commission
            self.cash += position.quantity * price - commission
        else:  # Close short position
            pnl = abs(position.quantity) * (position.entry_price - price) - commission
            self.cash += abs(position.quantity) * (position.entry_price - price) - commission
        
        trade = Trade(
            symbol=symbol,
            entry_date=position.entry_date,
            exit_date=date,
            entry_price=position.entry_price,
            exit_price=price,
            quantity=abs(position.quantity),
            position_type=position.position_type,
            pnl=pnl,
            pnl_pct=(price / position.entry_price - 1) * 100 if position.quantity > 0 
                    else (position.entry_price / price - 1) * 100,
            commission=commission,
            duration=(date - position.entry_date).days
        )
        self.trades.append(trade)
        del self.positions[symbol]
    
    def run_backtest(self, 
                    market_data: Dict[str, pd.DataFrame],
                    signals: Dict[str, List],
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> BacktestResults:
        
        self.reset()
        
        # Get date range
        all_dates = set()
        for symbol_data in market_data.values():
            all_dates.update(symbol_data.index)
        
        all_dates = sorted(list(all_dates))
        
        if start_date:
            start_date = pd.to_datetime(start_date)
            all_dates = [d for d in all_dates if d >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            all_dates = [d for d in all_dates if d <= end_date]
        
        # Create signal lookup
        signal_lookup = {}
        for symbol, symbol_signals in signals.items():
            signal_lookup[symbol] = {}
            for signal in symbol_signals:
                signal_lookup[symbol][signal.timestamp] = signal
        
        # Run simulation
        for date in all_dates:
            self.current_date = date
            
            # Process signals for this date
            for symbol in market_data.keys():
                if (symbol in signal_lookup and 
                    date in signal_lookup[symbol] and
                    date in market_data[symbol].index):
                    
                    signal = signal_lookup[symbol][date]
                    current_price = market_data[symbol].loc[date, 'Close']
                    
                    if signal.signal.value == 1:  # Buy signal
                        quantity = self.calculate_position_size(current_price, signal.confidence)
                        if quantity > 0:
                            self.execute_trade(symbol, quantity, current_price, 'buy', date)
                    
                    elif signal.signal.value == -1:  # Sell signal
                        if symbol in self.positions and self.positions[symbol].quantity > 0:
                            quantity = self.positions[symbol].quantity
                            self.execute_trade(symbol, quantity, current_price, 'sell', date)
            
            # Record portfolio value
            portfolio_value = self.calculate_portfolio_value(market_data)
            self.portfolio_values.append({
                'Date': date,
                'Portfolio_Value': portfolio_value,
                'Cash': self.cash
            })
        
        # Close all remaining positions
        final_date = all_dates[-1] if all_dates else None
        if final_date:
            for symbol in list(self.positions.keys()):
                if symbol in market_data and final_date in market_data[symbol].index:
                    final_price = market_data[symbol].loc[final_date, 'Close']
                    self._close_position(symbol, final_price, final_date)
        
        # Create results
        portfolio_df = pd.DataFrame(self.portfolio_values)
        if not portfolio_df.empty:
            portfolio_df.set_index('Date', inplace=True)
            portfolio_series = portfolio_df['Portfolio_Value']
        else:
            portfolio_series = pd.Series()
        
        results = BacktestResults(
            trades=self.trades.copy(),
            portfolio_values=portfolio_series,
            positions=self.positions.copy()
        )
        
        # Calculate performance metrics
        results.metrics = self.calculate_performance_metrics(results)
        
        return results
    
    def calculate_performance_metrics(self, results: BacktestResults) -> Dict[str, float]:
        if results.portfolio_values.empty or len(results.trades) == 0:
            return {}
        
        portfolio_values = results.portfolio_values
        trades_df = results.to_dataframe()
        
        # Basic metrics
        total_return = (portfolio_values.iloc[-1] / self.initial_capital - 1) * 100
        
        # Calculate daily returns
        daily_returns = portfolio_values.pct_change().dropna()
        
        # Risk metrics
        volatility = daily_returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() != 0 else 0
        
        # Drawdown calculation
        cumulative = portfolio_values / portfolio_values.expanding().max()
        drawdowns = (1 - cumulative) * 100
        max_drawdown = drawdowns.max()
        
        # Trade statistics
        winning_trades = trades_df[trades_df['PnL'] > 0]
        losing_trades = trades_df[trades_df['PnL'] < 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        avg_win = winning_trades['PnL'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['PnL'].mean() if len(losing_trades) > 0 else 0
        profit_factor = abs(winning_trades['PnL'].sum() / losing_trades['PnL'].sum()) if len(losing_trades) > 0 and losing_trades['PnL'].sum() != 0 else np.inf
        
        return {
            'Total_Return_Pct': total_return,
            'Volatility_Pct': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown_Pct': max_drawdown,
            'Win_Rate_Pct': win_rate,
            'Total_Trades': len(trades_df),
            'Winning_Trades': len(winning_trades),
            'Losing_Trades': len(losing_trades),
            'Average_Win': avg_win,
            'Average_Loss': avg_loss,
            'Profit_Factor': profit_factor,
            'Final_Portfolio_Value': portfolio_values.iloc[-1]
        }