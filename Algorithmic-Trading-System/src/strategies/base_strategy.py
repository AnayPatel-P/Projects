from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0

@dataclass
class TradingSignal:
    timestamp: pd.Timestamp
    symbol: str
    signal: SignalType
    price: float
    confidence: float = 0.5
    position_size: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict = None

class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name
        self.signals = []
        self.parameters = {}
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        pass
    
    @abstractmethod
    def set_parameters(self, **kwargs):
        pass
    
    def get_parameters(self) -> Dict:
        return self.parameters
    
    def backtest_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = self.generate_signals(data)
        
        signal_df = pd.DataFrame([{
            'Date': signal.timestamp,
            'Symbol': signal.symbol,
            'Signal': signal.signal.value,
            'Price': signal.price,
            'Confidence': signal.confidence,
            'Position_Size': signal.position_size
        } for signal in signals])
        
        if not signal_df.empty:
            signal_df.set_index('Date', inplace=True)
        
        return signal_df
    
    def optimize_parameters(self, data: pd.DataFrame, parameter_ranges: Dict, 
                          optimization_metric: str = 'sharpe_ratio'):
        best_params = None
        best_score = -np.inf
        
        # Simple grid search optimization
        from itertools import product
        
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        for param_combination in product(*param_values):
            params = dict(zip(param_names, param_combination))
            self.set_parameters(**params)
            
            try:
                # Generate signals and calculate performance
                signals_df = self.backtest_signals(data)
                score = self._calculate_performance_metric(data, signals_df, optimization_metric)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            except Exception as e:
                continue
        
        if best_params:
            self.set_parameters(**best_params)
        
        return best_params, best_score
    
    def _calculate_performance_metric(self, data: pd.DataFrame, signals_df: pd.DataFrame, 
                                    metric: str) -> float:
        if signals_df.empty:
            return -np.inf
        
        # Simple performance calculation
        returns = []
        position = 0
        
        for date, row in data.iterrows():
            if date in signals_df.index:
                signal = signals_df.loc[date, 'Signal']
                if signal != 0:
                    position = signal
            
            if position != 0:
                daily_return = data.loc[date, 'Close'] / data.loc[date, 'Open'] - 1
                returns.append(daily_return * position)
            else:
                returns.append(0)
        
        returns = np.array(returns)
        
        if metric == 'sharpe_ratio':
            return np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        elif metric == 'total_return':
            return np.sum(returns)
        elif metric == 'max_drawdown':
            cumulative = np.cumprod(1 + returns)
            drawdown = (cumulative / np.maximum.accumulate(cumulative) - 1)
            return -np.min(drawdown)  # Negative because we want to minimize drawdown
        
        return 0