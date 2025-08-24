import pandas as pd
import numpy as np
import talib
from typing import List
from .base_strategy import BaseStrategy, TradingSignal, SignalType

class RSIStrategy(BaseStrategy):
    def __init__(self, rsi_period: int = 14, oversold_threshold: float = 30, 
                 overbought_threshold: float = 70):
        super().__init__("RSI Mean Reversion")
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.parameters = {
            'rsi_period': rsi_period,
            'oversold_threshold': oversold_threshold,
            'overbought_threshold': overbought_threshold
        }
    
    def set_parameters(self, **kwargs):
        self.rsi_period = kwargs.get('rsi_period', self.rsi_period)
        self.oversold_threshold = kwargs.get('oversold_threshold', self.oversold_threshold)
        self.overbought_threshold = kwargs.get('overbought_threshold', self.overbought_threshold)
        self.parameters.update(kwargs)
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        if len(data) < self.rsi_period + 1:
            return []
        
        data = data.copy()
        data['RSI'] = talib.RSI(data['Close'], timeperiod=self.rsi_period)
        
        # Generate signals
        data['Signal'] = 0
        data.loc[data['RSI'] < self.oversold_threshold, 'Signal'] = 1  # Buy when oversold
        data.loc[data['RSI'] > self.overbought_threshold, 'Signal'] = -1  # Sell when overbought
        
        # Only trigger on crossovers
        data['RSI_prev'] = data['RSI'].shift(1)
        
        buy_signals = (
            (data['RSI'] > self.oversold_threshold) & 
            (data['RSI_prev'] <= self.oversold_threshold)
        )
        
        sell_signals = (
            (data['RSI'] < self.overbought_threshold) & 
            (data['RSI_prev'] >= self.overbought_threshold)
        )
        
        signals = []
        for date, row in data.iterrows():
            if pd.notna(row['RSI']):
                signal_type = None
                
                if buy_signals.loc[date]:
                    signal_type = SignalType.BUY
                elif sell_signals.loc[date]:
                    signal_type = SignalType.SELL
                
                if signal_type:
                    # Calculate confidence based on how extreme the RSI is
                    if signal_type == SignalType.BUY:
                        confidence = (self.oversold_threshold - min(row['RSI'], self.oversold_threshold)) / self.oversold_threshold
                    else:
                        confidence = (max(row['RSI'], self.overbought_threshold) - self.overbought_threshold) / (100 - self.overbought_threshold)
                    
                    confidence = max(0.1, min(1.0, confidence))
                    
                    signal = TradingSignal(
                        timestamp=date,
                        symbol=data.attrs.get('symbol', 'UNKNOWN'),
                        signal=signal_type,
                        price=row['Close'],
                        confidence=confidence,
                        metadata={'rsi': row['RSI']}
                    )
                    signals.append(signal)
        
        return signals

class RSIDivergenceStrategy(BaseStrategy):
    def __init__(self, rsi_period: int = 14, lookback_period: int = 20):
        super().__init__("RSI Divergence")
        self.rsi_period = rsi_period
        self.lookback_period = lookback_period
        self.parameters = {
            'rsi_period': rsi_period,
            'lookback_period': lookback_period
        }
    
    def set_parameters(self, **kwargs):
        self.rsi_period = kwargs.get('rsi_period', self.rsi_period)
        self.lookback_period = kwargs.get('lookback_period', self.lookback_period)
        self.parameters.update(kwargs)
    
    def _detect_divergence(self, prices: pd.Series, rsi: pd.Series, lookback: int) -> int:
        if len(prices) < lookback or len(rsi) < lookback:
            return 0
        
        # Look for peaks and troughs
        price_recent = prices.iloc[-1]
        price_prev = prices.iloc[-lookback]
        rsi_recent = rsi.iloc[-1]
        rsi_prev = rsi.iloc[-lookback]
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        if price_recent < price_prev and rsi_recent > rsi_prev:
            return 1
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        if price_recent > price_prev and rsi_recent < rsi_prev:
            return -1
        
        return 0
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        if len(data) < self.rsi_period + self.lookback_period:
            return []
        
        data = data.copy()
        data['RSI'] = talib.RSI(data['Close'], timeperiod=self.rsi_period)
        
        signals = []
        
        for i in range(self.lookback_period, len(data)):
            current_date = data.index[i]
            
            # Get data window for divergence analysis
            price_window = data['Close'].iloc[i-self.lookback_period:i+1]
            rsi_window = data['RSI'].iloc[i-self.lookback_period:i+1]
            
            divergence = self._detect_divergence(price_window, rsi_window, self.lookback_period)
            
            if divergence != 0:
                signal_type = SignalType.BUY if divergence == 1 else SignalType.SELL
                
                # Calculate confidence based on the strength of divergence
                price_change = abs(price_window.iloc[-1] - price_window.iloc[0]) / price_window.iloc[0]
                rsi_change = abs(rsi_window.iloc[-1] - rsi_window.iloc[0])
                
                confidence = min(price_change * rsi_change / 10, 1.0)
                confidence = max(0.3, confidence)  # Minimum confidence
                
                signal = TradingSignal(
                    timestamp=current_date,
                    symbol=data.attrs.get('symbol', 'UNKNOWN'),
                    signal=signal_type,
                    price=data.loc[current_date, 'Close'],
                    confidence=confidence,
                    metadata={
                        'rsi': data.loc[current_date, 'RSI'],
                        'divergence_type': 'bullish' if divergence == 1 else 'bearish'
                    }
                )
                signals.append(signal)
        
        return signals