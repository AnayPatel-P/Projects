import pandas as pd
import numpy as np
from typing import List
from .base_strategy import BaseStrategy, TradingSignal, SignalType

class MovingAverageStrategy(BaseStrategy):
    def __init__(self, short_window: int = 20, long_window: int = 50):
        super().__init__("Moving Average Crossover")
        self.short_window = short_window
        self.long_window = long_window
        self.parameters = {
            'short_window': short_window,
            'long_window': long_window
        }
    
    def set_parameters(self, **kwargs):
        self.short_window = kwargs.get('short_window', self.short_window)
        self.long_window = kwargs.get('long_window', self.long_window)
        self.parameters.update(kwargs)
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        if len(data) < self.long_window:
            return []
        
        # Calculate moving averages
        data = data.copy()
        data['SMA_short'] = data['Close'].rolling(window=self.short_window).mean()
        data['SMA_long'] = data['Close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        data['Signal'] = 0
        data.loc[data['SMA_short'] > data['SMA_long'], 'Signal'] = 1
        data.loc[data['SMA_short'] < data['SMA_long'], 'Signal'] = -1
        
        # Detect signal changes
        data['Signal_Change'] = data['Signal'].diff()
        
        signals = []
        for date, row in data.iterrows():
            if pd.notna(row['Signal_Change']) and row['Signal_Change'] != 0:
                if row['Signal'] == 1:
                    signal_type = SignalType.BUY
                elif row['Signal'] == -1:
                    signal_type = SignalType.SELL
                else:
                    continue
                
                # Calculate confidence based on the spread between MAs
                ma_spread = abs(row['SMA_short'] - row['SMA_long']) / row['Close']
                confidence = min(ma_spread * 10, 1.0)  # Scale and cap at 1.0
                
                signal = TradingSignal(
                    timestamp=date,
                    symbol=data.attrs.get('symbol', 'UNKNOWN'),
                    signal=signal_type,
                    price=row['Close'],
                    confidence=confidence,
                    metadata={
                        'short_ma': row['SMA_short'],
                        'long_ma': row['SMA_long'],
                        'ma_spread': ma_spread
                    }
                )
                signals.append(signal)
        
        return signals

class DualMovingAverageStrategy(BaseStrategy):
    def __init__(self, short_window: int = 10, medium_window: int = 20, long_window: int = 50):
        super().__init__("Dual Moving Average")
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.parameters = {
            'short_window': short_window,
            'medium_window': medium_window,
            'long_window': long_window
        }
    
    def set_parameters(self, **kwargs):
        self.short_window = kwargs.get('short_window', self.short_window)
        self.medium_window = kwargs.get('medium_window', self.medium_window)
        self.long_window = kwargs.get('long_window', self.long_window)
        self.parameters.update(kwargs)
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        if len(data) < self.long_window:
            return []
        
        data = data.copy()
        data['SMA_short'] = data['Close'].rolling(window=self.short_window).mean()
        data['SMA_medium'] = data['Close'].rolling(window=self.medium_window).mean()
        data['SMA_long'] = data['Close'].rolling(window=self.long_window).mean()
        
        # Generate signals based on multiple MA conditions
        data['Signal'] = 0
        
        # Strong buy: short > medium > long and all trending up
        buy_condition = (
            (data['SMA_short'] > data['SMA_medium']) &
            (data['SMA_medium'] > data['SMA_long']) &
            (data['SMA_short'] > data['SMA_short'].shift(1)) &
            (data['SMA_medium'] > data['SMA_medium'].shift(1))
        )
        
        # Strong sell: short < medium < long and all trending down
        sell_condition = (
            (data['SMA_short'] < data['SMA_medium']) &
            (data['SMA_medium'] < data['SMA_long']) &
            (data['SMA_short'] < data['SMA_short'].shift(1)) &
            (data['SMA_medium'] < data['SMA_medium'].shift(1))
        )
        
        data.loc[buy_condition, 'Signal'] = 1
        data.loc[sell_condition, 'Signal'] = -1
        
        data['Signal_Change'] = data['Signal'].diff()
        
        signals = []
        for date, row in data.iterrows():
            if pd.notna(row['Signal_Change']) and row['Signal_Change'] != 0:
                if row['Signal'] == 1:
                    signal_type = SignalType.BUY
                elif row['Signal'] == -1:
                    signal_type = SignalType.SELL
                else:
                    continue
                
                # Calculate confidence based on MA alignment strength
                ma_alignment = (
                    abs(row['SMA_short'] - row['SMA_medium']) + 
                    abs(row['SMA_medium'] - row['SMA_long'])
                ) / (2 * row['Close'])
                confidence = min(ma_alignment * 20, 1.0)
                
                signal = TradingSignal(
                    timestamp=date,
                    symbol=data.attrs.get('symbol', 'UNKNOWN'),
                    signal=signal_type,
                    price=row['Close'],
                    confidence=confidence,
                    metadata={
                        'short_ma': row['SMA_short'],
                        'medium_ma': row['SMA_medium'],
                        'long_ma': row['SMA_long']
                    }
                )
                signals.append(signal)
        
        return signals