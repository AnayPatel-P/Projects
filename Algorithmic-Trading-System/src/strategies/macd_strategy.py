import pandas as pd
import numpy as np
import talib
from typing import List
from .base_strategy import BaseStrategy, TradingSignal, SignalType

class MACDStrategy(BaseStrategy):
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__("MACD Strategy")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period
        }
    
    def set_parameters(self, **kwargs):
        self.fast_period = kwargs.get('fast_period', self.fast_period)
        self.slow_period = kwargs.get('slow_period', self.slow_period)
        self.signal_period = kwargs.get('signal_period', self.signal_period)
        self.parameters.update(kwargs)
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        if len(data) < self.slow_period + self.signal_period:
            return []
        
        data = data.copy()
        
        # Calculate MACD
        data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = talib.MACD(
            data['Close'], 
            fastperiod=self.fast_period,
            slowperiod=self.slow_period, 
            signalperiod=self.signal_period
        )
        
        # Generate signals on MACD line crossovers with signal line
        data['MACD_prev'] = data['MACD'].shift(1)
        data['Signal_prev'] = data['MACD_Signal'].shift(1)
        
        # Buy when MACD crosses above signal line
        buy_condition = (
            (data['MACD'] > data['MACD_Signal']) & 
            (data['MACD_prev'] <= data['Signal_prev'])
        )
        
        # Sell when MACD crosses below signal line
        sell_condition = (
            (data['MACD'] < data['MACD_Signal']) & 
            (data['MACD_prev'] >= data['Signal_prev'])
        )
        
        signals = []
        
        for date, row in data.iterrows():
            if pd.notna(row['MACD']) and pd.notna(row['MACD_Signal']):
                signal_type = None
                
                if buy_condition.loc[date]:
                    signal_type = SignalType.BUY
                elif sell_condition.loc[date]:
                    signal_type = SignalType.SELL
                
                if signal_type:
                    # Calculate confidence based on MACD histogram and distance between lines
                    macd_distance = abs(row['MACD'] - row['MACD_Signal'])
                    histogram_strength = abs(row['MACD_Hist'])
                    
                    confidence = min((macd_distance + histogram_strength) / row['Close'] * 1000, 1.0)
                    confidence = max(0.2, confidence)
                    
                    signal = TradingSignal(
                        timestamp=date,
                        symbol=data.attrs.get('symbol', 'UNKNOWN'),
                        signal=signal_type,
                        price=row['Close'],
                        confidence=confidence,
                        metadata={
                            'macd': row['MACD'],
                            'macd_signal': row['MACD_Signal'],
                            'macd_histogram': row['MACD_Hist']
                        }
                    )
                    signals.append(signal)
        
        return signals

class MACDZeroCrossStrategy(BaseStrategy):
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__("MACD Zero Cross Strategy")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period
        }
    
    def set_parameters(self, **kwargs):
        self.fast_period = kwargs.get('fast_period', self.fast_period)
        self.slow_period = kwargs.get('slow_period', self.slow_period)
        self.signal_period = kwargs.get('signal_period', self.signal_period)
        self.parameters.update(kwargs)
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        if len(data) < self.slow_period + self.signal_period:
            return []
        
        data = data.copy()
        
        # Calculate MACD
        data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = talib.MACD(
            data['Close'], 
            fastperiod=self.fast_period,
            slowperiod=self.slow_period, 
            signalperiod=self.signal_period
        )
        
        # Generate signals on MACD zero line crossovers
        data['MACD_prev'] = data['MACD'].shift(1)
        
        # Buy when MACD crosses above zero
        buy_condition = (data['MACD'] > 0) & (data['MACD_prev'] <= 0)
        
        # Sell when MACD crosses below zero
        sell_condition = (data['MACD'] < 0) & (data['MACD_prev'] >= 0)
        
        signals = []
        
        for date, row in data.iterrows():
            if pd.notna(row['MACD']):
                signal_type = None
                
                if buy_condition.loc[date]:
                    signal_type = SignalType.BUY
                elif sell_condition.loc[date]:
                    signal_type = SignalType.SELL
                
                if signal_type:
                    # Higher confidence when MACD signal line also supports the direction
                    macd_signal_alignment = (
                        (signal_type == SignalType.BUY and row['MACD_Signal'] > 0) or
                        (signal_type == SignalType.SELL and row['MACD_Signal'] < 0)
                    )
                    
                    base_confidence = 0.6
                    if macd_signal_alignment:
                        base_confidence = 0.8
                    
                    # Adjust based on histogram strength
                    histogram_factor = min(abs(row['MACD_Hist']) / row['Close'] * 1000, 0.2)
                    confidence = min(base_confidence + histogram_factor, 1.0)
                    
                    signal = TradingSignal(
                        timestamp=date,
                        symbol=data.attrs.get('symbol', 'UNKNOWN'),
                        signal=signal_type,
                        price=row['Close'],
                        confidence=confidence,
                        metadata={
                            'macd': row['MACD'],
                            'macd_signal': row['MACD_Signal'],
                            'macd_histogram': row['MACD_Hist'],
                            'signal_alignment': macd_signal_alignment
                        }
                    )
                    signals.append(signal)
        
        return signals

class MACDDivergenceStrategy(BaseStrategy):
    def __init__(self, fast_period: int = 12, slow_period: int = 26, 
                 signal_period: int = 9, lookback_period: int = 20):
        super().__init__("MACD Divergence Strategy")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.lookback_period = lookback_period
        self.parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'lookback_period': lookback_period
        }
    
    def set_parameters(self, **kwargs):
        self.fast_period = kwargs.get('fast_period', self.fast_period)
        self.slow_period = kwargs.get('slow_period', self.slow_period)
        self.signal_period = kwargs.get('signal_period', self.signal_period)
        self.lookback_period = kwargs.get('lookback_period', self.lookback_period)
        self.parameters.update(kwargs)
    
    def _detect_divergence(self, prices: pd.Series, macd: pd.Series, lookback: int) -> int:
        if len(prices) < lookback or len(macd) < lookback:
            return 0
        
        # Simple divergence detection
        price_trend = prices.iloc[-1] - prices.iloc[-lookback]
        macd_trend = macd.iloc[-1] - macd.iloc[-lookback]
        
        # Bullish divergence: price down, MACD up
        if price_trend < 0 and macd_trend > 0:
            return 1
        
        # Bearish divergence: price up, MACD down  
        if price_trend > 0 and macd_trend < 0:
            return -1
        
        return 0
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        if len(data) < self.slow_period + self.signal_period + self.lookback_period:
            return []
        
        data = data.copy()
        
        # Calculate MACD
        data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = talib.MACD(
            data['Close'], 
            fastperiod=self.fast_period,
            slowperiod=self.slow_period, 
            signalperiod=self.signal_period
        )
        
        signals = []
        
        for i in range(self.lookback_period, len(data)):
            current_date = data.index[i]
            
            # Get windows for divergence analysis
            price_window = data['Close'].iloc[i-self.lookback_period:i+1]
            macd_window = data['MACD'].iloc[i-self.lookback_period:i+1]
            
            divergence = self._detect_divergence(price_window, macd_window, self.lookback_period)
            
            if divergence != 0:
                signal_type = SignalType.BUY if divergence == 1 else SignalType.SELL
                
                # Calculate confidence based on strength of divergence
                price_change_pct = abs(price_window.iloc[-1] - price_window.iloc[0]) / price_window.iloc[0]
                macd_change = abs(macd_window.iloc[-1] - macd_window.iloc[0])
                
                confidence = min(price_change_pct * macd_change * 100, 1.0)
                confidence = max(0.4, confidence)
                
                signal = TradingSignal(
                    timestamp=current_date,
                    symbol=data.attrs.get('symbol', 'UNKNOWN'),
                    signal=signal_type,
                    price=data.loc[current_date, 'Close'],
                    confidence=confidence,
                    metadata={
                        'macd': data.loc[current_date, 'MACD'],
                        'macd_signal': data.loc[current_date, 'MACD_Signal'],
                        'divergence_type': 'bullish' if divergence == 1 else 'bearish'
                    }
                )
                signals.append(signal)
        
        return signals