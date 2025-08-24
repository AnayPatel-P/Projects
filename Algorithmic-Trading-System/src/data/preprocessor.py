import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = None
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.fillna(method='forward').fillna(method='backward')
        
        # Remove outliers using IQR method
        for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if column in df.columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Price-based indicators
        df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
        df['EMA_12'] = talib.EMA(df['Close'], timeperiod=12)
        df['EMA_26'] = talib.EMA(df['Close'], timeperiod=26)
        
        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['Close'])
        
        # RSI
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        
        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df['Close'])
        
        # Stochastic Oscillator
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])
        
        # Average True Range
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'])
        
        # Volume indicators
        if 'Volume' in df.columns:
            df['OBV'] = talib.OBV(df['Close'], df['Volume'])
            df['Volume_SMA'] = talib.SMA(df['Volume'], timeperiod=20)
        
        # Price changes and returns
        df['Returns'] = df['Close'].pct_change()
        df['Price_Change'] = df['Close'] - df['Open']
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Volume_Price_Trend'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * df['Volume']
        
        # Volatility measures
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Lagged features
        for lag in [1, 5, 10]:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag) if 'Volume' in df.columns else np.nan
            df[f'Returns_lag_{lag}'] = df['Returns'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'Close_rolling_mean_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'Close_rolling_std_{window}'] = df['Close'].rolling(window=window).std()
            df[f'Volume_rolling_mean_{window}'] = df['Volume'].rolling(window=window).mean() if 'Volume' in df.columns else np.nan
        
        # Time-based features
        df['Day_of_week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        
        return df
    
    def normalize_data(self, df: pd.DataFrame, columns: List[str], method: str = 'standard') -> pd.DataFrame:
        df = df.copy()
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        df[columns] = self.scaler.fit_transform(df[columns])
        
        return df
    
    def process_multiple_symbols(self, 
                                data: Dict[str, pd.DataFrame], 
                                add_technical_indicators: bool = True,
                                add_features: bool = True,
                                normalize: bool = False,
                                normalize_columns: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        processed_data = {}
        
        for symbol, df in data.items():
            print(f"Processing {symbol}...")
            
            # Clean data
            df_processed = self.clean_data(df)
            
            # Add technical indicators
            if add_technical_indicators:
                df_processed = self.calculate_technical_indicators(df_processed)
            
            # Add features
            if add_features:
                df_processed = self.create_features(df_processed)
            
            # Normalize
            if normalize and normalize_columns:
                available_columns = [col for col in normalize_columns if col in df_processed.columns]
                if available_columns:
                    df_processed = self.normalize_data(df_processed, available_columns)
            
            processed_data[symbol] = df_processed
            
        return processed_data
    
    def create_target_variable(self, df: pd.DataFrame, 
                              target_type: str = 'returns', 
                              horizon: int = 1) -> pd.DataFrame:
        df = df.copy()
        
        if target_type == 'returns':
            df['Target'] = df['Close'].pct_change(horizon).shift(-horizon)
        elif target_type == 'price':
            df['Target'] = df['Close'].shift(-horizon)
        elif target_type == 'direction':
            df['Target'] = (df['Close'].shift(-horizon) > df['Close']).astype(int)
        else:
            raise ValueError("target_type must be 'returns', 'price', or 'direction'")
        
        return df