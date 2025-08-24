import pandas as pd
import numpy as np
import talib
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    def __init__(self):
        self.scalers = {}
        self.feature_selectors = {}
        self.pca_transformers = {}
        
    def create_time_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Time-based features
        df['Hour'] = df.index.hour if hasattr(df.index, 'hour') else 0
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Day_of_Month'] = df.index.day
        df['Week_of_Year'] = df.index.isocalendar().week if hasattr(df.index, 'isocalendar') else 0
        
        # Market session features
        df['Is_Monday'] = (df['Day_of_Week'] == 0).astype(int)
        df['Is_Friday'] = (df['Day_of_Week'] == 4).astype(int)
        df['Is_Month_End'] = df.index.is_month_end.astype(int)
        df['Is_Month_Start'] = df.index.is_month_start.astype(int)
        df['Is_Quarter_End'] = df.index.is_quarter_end.astype(int)
        
        # Cyclical encoding for time features
        for feature, max_val in [('Hour', 24), ('Day_of_Week', 7), ('Month', 12), ('Day_of_Month', 31)]:
            if feature in df.columns:
                df[f'{feature}_sin'] = np.sin(2 * np.pi * df[feature] / max_val)
                df[f'{feature}_cos'] = np.cos(2 * np.pi * df[feature] / max_val)
        
        return df
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Basic price features
        df['OHLC_Mean'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        df['HL_Mean'] = (df['High'] + df['Low']) / 2
        df['OC_Mean'] = (df['Open'] + df['Close']) / 2
        
        # Price ranges and spreads
        df['True_Range'] = np.maximum(df['High'] - df['Low'], 
                          np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                   abs(df['Low'] - df['Close'].shift(1))))
        
        df['Price_Range'] = df['High'] - df['Low']
        df['Price_Range_Pct'] = df['Price_Range'] / df['Close']
        df['Open_Close_Spread'] = df['Close'] - df['Open']
        df['Open_Close_Spread_Pct'] = df['Open_Close_Spread'] / df['Open']
        
        # Gap analysis
        df['Gap'] = df['Open'] - df['Close'].shift(1)
        df['Gap_Pct'] = df['Gap'] / df['Close'].shift(1)
        df['Gap_Up'] = (df['Gap'] > 0).astype(int)
        df['Gap_Down'] = (df['Gap'] < 0).astype(int)
        
        # Intraday patterns
        df['Upper_Shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df['Lower_Shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
        df['Body_Size'] = abs(df['Close'] - df['Open'])
        
        # Relative positions
        df['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['Open_Position'] = (df['Open'] - df['Low']) / (df['High'] - df['Low'])
        
        return df
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Volume' not in df.columns:
            return df
            
        df = df.copy()
        
        # Volume-price relationships
        df['Volume_Price_Trend'] = df['Volume'] * (df['Close'] - df['Close'].shift(1))
        df['Price_Volume'] = df['Close'] * df['Volume']
        df['VWAP'] = (df['Price_Volume'].rolling(20).sum() / df['Volume'].rolling(20).sum())
        
        # Volume patterns
        df['Volume_SMA_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['Volume_Spike'] = (df['Volume'] > df['Volume'].rolling(20).mean() * 2).astype(int)
        
        # On Balance Volume variations
        df['OBV'] = (df['Volume'] * np.where(df['Close'] > df['Close'].shift(1), 1, 
                    np.where(df['Close'] < df['Close'].shift(1), -1, 0))).cumsum()
        
        # Accumulation/Distribution Line
        df['CLV'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        df['AD_Line'] = (df['CLV'] * df['Volume']).cumsum()
        
        # Money Flow Index
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
        
        money_ratio = (pd.Series(positive_flow).rolling(14).sum() / 
                      pd.Series(negative_flow).rolling(14).sum())
        df['MFI'] = 100 - (100 / (1 + money_ratio))
        
        return df
    
    def create_advanced_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Momentum indicators
        df['ROC_1'] = df['Close'].pct_change(1)
        df['ROC_5'] = df['Close'].pct_change(5)
        df['ROC_10'] = df['Close'].pct_change(10)
        df['ROC_20'] = df['Close'].pct_change(20)
        
        # Multiple timeframe moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = talib.SMA(df['Close'], timeperiod=period)
            df[f'EMA_{period}'] = talib.EMA(df['Close'], timeperiod=period)
            
            # MA ratios
            if period > 5:
                df[f'Close_SMA_{period}_Ratio'] = df['Close'] / df[f'SMA_{period}']
                df[f'SMA_5_SMA_{period}_Ratio'] = df['SMA_5'] / df[f'SMA_{period}']
        
        # Bollinger Bands with multiple periods
        for period in [10, 20, 50]:
            upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=period)
            df[f'BB_Upper_{period}'] = upper
            df[f'BB_Lower_{period}'] = lower
            df[f'BB_Width_{period}'] = (upper - lower) / middle
            df[f'BB_Position_{period}'] = (df['Close'] - lower) / (upper - lower)
        
        # RSI with multiple periods
        for period in [7, 14, 21]:
            df[f'RSI_{period}'] = talib.RSI(df['Close'], timeperiod=period)
            
        # MACD variations
        for fast, slow in [(12, 26), (5, 35), (8, 21)]:
            macd, signal, hist = talib.MACD(df['Close'], fastperiod=fast, slowperiod=slow)
            df[f'MACD_{fast}_{slow}'] = macd
            df[f'MACD_Signal_{fast}_{slow}'] = signal
            df[f'MACD_Hist_{fast}_{slow}'] = hist
        
        # Stochastic variations
        for k_period, d_period in [(14, 3), (21, 5), (5, 3)]:
            k, d = talib.STOCH(df['High'], df['Low'], df['Close'], 
                              fastk_period=k_period, slowd_period=d_period)
            df[f'STOCH_K_{k_period}_{d_period}'] = k
            df[f'STOCH_D_{k_period}_{d_period}'] = d
        
        # Williams %R
        for period in [14, 21]:
            df[f'WILLR_{period}'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=period)
        
        # Commodity Channel Index
        for period in [14, 20]:
            df[f'CCI_{period}'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=period)
        
        # Average True Range
        for period in [14, 21]:
            df[f'ATR_{period}'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=period)
            df[f'ATR_Ratio_{period}'] = df[f'ATR_{period}'] / df['Close']
        
        return df
    
    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Historical volatility
        returns = df['Close'].pct_change()
        
        for window in [5, 10, 20, 30, 60]:
            df[f'Volatility_{window}'] = returns.rolling(window).std() * np.sqrt(252)
            df[f'Volatility_Rank_{window}'] = (
                returns.rolling(window).std().rolling(252).rank(pct=True)
            )
        
        # Parkinson volatility (using high-low)
        for window in [10, 20, 30]:
            df[f'Parkinson_Vol_{window}'] = np.sqrt(
                (np.log(df['High'] / df['Low']) ** 2).rolling(window).mean() * 252 / (4 * np.log(2))
            )
        
        # Garman-Klass volatility
        for window in [10, 20, 30]:
            ln_hl = np.log(df['High'] / df['Low'])
            ln_co = np.log(df['Close'] / df['Open'])
            df[f'GK_Vol_{window}'] = np.sqrt(
                (0.5 * ln_hl**2 - (2*np.log(2) - 1) * ln_co**2).rolling(window).mean() * 252
            )
        
        # Volatility clustering
        df['Vol_Cluster'] = (returns.abs() > returns.rolling(60).quantile(0.95)).astype(int)
        
        return df
    
    def create_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Price efficiency measures
        returns = df['Close'].pct_change()
        
        # Variance ratio
        for lag in [2, 4, 8]:
            var_1 = returns.var()
            var_lag = returns.rolling(lag).sum().var() / lag
            df[f'Variance_Ratio_{lag}'] = var_lag / var_1 if var_1 != 0 else 1
        
        # Hurst exponent (simplified)
        for window in [50, 100]:
            df[f'Hurst_{window}'] = returns.rolling(window).apply(
                lambda x: self._calculate_hurst(x), raw=True
            )
        
        # Fractal dimension
        for window in [20, 50]:
            df[f'Fractal_Dim_{window}'] = df['Close'].rolling(window).apply(
                lambda x: self._calculate_fractal_dimension(x), raw=True
            )
        
        return df
    
    def _calculate_hurst(self, prices: np.ndarray) -> float:
        if len(prices) < 10:
            return 0.5
        
        try:
            lags = range(2, min(len(prices)//2, 20))
            tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag]))) for lag in lags]
            
            if len(tau) < 2:
                return 0.5
                
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return 0.5
    
    def _calculate_fractal_dimension(self, prices: np.ndarray) -> float:
        if len(prices) < 4:
            return 1.5
            
        try:
            N = len(prices)
            n_max = int(np.log2(N))
            
            ns = [2**i for i in range(1, n_max)]
            rs = []
            
            for n in ns:
                segments = N // n
                if segments < 2:
                    continue
                    
                segment_vars = []
                for i in range(segments):
                    segment = prices[i*n:(i+1)*n]
                    if len(segment) >= 2:
                        segment_vars.append(np.var(segment))
                
                if segment_vars:
                    rs.append(np.mean(segment_vars))
            
            if len(rs) < 2:
                return 1.5
                
            poly = np.polyfit(np.log(ns[:len(rs)]), np.log(rs), 1)
            return 2 - poly[0] / 2
        except:
            return 1.5
    
    def create_lagged_features(self, df: pd.DataFrame, 
                             target_columns: List[str] = None,
                             lags: List[int] = None) -> pd.DataFrame:
        if target_columns is None:
            target_columns = ['Close', 'Volume', 'RSI_14', 'MACD_12_26']
        
        if lags is None:
            lags = [1, 2, 3, 5, 10, 20]
        
        df = df.copy()
        
        for col in target_columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def create_rolling_statistics(self, df: pd.DataFrame,
                                 target_columns: List[str] = None,
                                 windows: List[int] = None) -> pd.DataFrame:
        if target_columns is None:
            target_columns = ['Close', 'Volume']
        
        if windows is None:
            windows = [5, 10, 20, 50]
        
        df = df.copy()
        
        for col in target_columns:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_mean_{window}'] = df[col].rolling(window).mean()
                    df[f'{col}_std_{window}'] = df[col].rolling(window).std()
                    df[f'{col}_min_{window}'] = df[col].rolling(window).min()
                    df[f'{col}_max_{window}'] = df[col].rolling(window).max()
                    df[f'{col}_skew_{window}'] = df[col].rolling(window).skew()
                    df[f'{col}_kurt_{window}'] = df[col].rolling(window).kurt()
                    
                    # Percentile features
                    df[f'{col}_q25_{window}'] = df[col].rolling(window).quantile(0.25)
                    df[f'{col}_q75_{window}'] = df[col].rolling(window).quantile(0.75)
                    df[f'{col}_median_{window}'] = df[col].rolling(window).median()
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Engineering comprehensive feature set...")
        
        # Apply all feature engineering steps
        df = self.create_price_features(df)
        df = self.create_volume_features(df)
        df = self.create_time_based_features(df)
        df = self.create_advanced_technical_indicators(df)
        df = self.create_volatility_features(df)
        df = self.create_market_microstructure_features(df)
        df = self.create_lagged_features(df)
        df = self.create_rolling_statistics(df)
        
        print(f"Generated {len(df.columns)} features")
        
        return df
    
    def select_features(self, df: pd.DataFrame, target: pd.Series,
                       method: str = 'mutual_info', k: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        
        # Remove non-numeric columns and handle missing values
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df.fillna(numeric_df.median())
        
        # Align target with features
        common_index = numeric_df.index.intersection(target.index)
        X = numeric_df.loc[common_index]
        y = target.loc[common_index]
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        elif method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=k)
        else:
            raise ValueError("Method must be 'mutual_info' or 'f_regression'")
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        print(f"Selected {len(selected_features)} features using {method}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard',
                      fit_columns: List[str] = None) -> pd.DataFrame:
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Method must be 'standard', 'minmax', or 'robust'")
        
        df = df.copy()
        
        if fit_columns is None:
            fit_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Only scale columns that exist in the dataframe
        columns_to_scale = [col for col in fit_columns if col in df.columns]
        
        if columns_to_scale:
            df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
            self.scalers[method] = scaler
        
        return df
    
    def apply_pca(self, df: pd.DataFrame, n_components: float = 0.95,
                 prefix: str = 'PC') -> Tuple[pd.DataFrame, PCA]:
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df.fillna(numeric_df.median())
        
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(numeric_df)
        
        # Create PCA DataFrame
        pca_columns = [f'{prefix}_{i+1}' for i in range(pca_features.shape[1])]
        pca_df = pd.DataFrame(pca_features, columns=pca_columns, index=df.index)
        
        print(f"PCA reduced {len(numeric_df.columns)} features to {len(pca_columns)} components")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        
        self.pca_transformers[prefix] = pca
        
        return pca_df, pca