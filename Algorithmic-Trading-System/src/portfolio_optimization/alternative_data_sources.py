import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

class EconomicDataProvider:
    """
    Fetch economic indicators and alternative data
    """
    
    def __init__(self):
        self.fred_api_key = os.getenv('FRED_API_KEY')
        self.cache = {}
        
    def get_fred_data(self, series_id: str, start_date: str, end_date: str) -> pd.Series:
        """Fetch data from FRED (Federal Reserve Economic Data)"""
        
        if not self.fred_api_key:
            print("FRED API key not found. Using dummy data.")
            return self._generate_dummy_economic_data(series_id, start_date, end_date)
        
        try:
            import fredapi
            fred = fredapi.Fred(api_key=self.fred_api_key)
            data = fred.get_series(series_id, start_date, end_date)
            return data
        except ImportError:
            print("fredapi not available. Install with: pip install fredapi")
            return self._generate_dummy_economic_data(series_id, start_date, end_date)
        except Exception as e:
            print(f"Error fetching FRED data: {e}")
            return self._generate_dummy_economic_data(series_id, start_date, end_date)
    
    def get_economic_indicators(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch key economic indicators"""
        
        indicators = {
            'GDP': 'GDP',                    # Gross Domestic Product
            'UNEMPLOYMENT': 'UNRATE',        # Unemployment Rate
            'INFLATION': 'CPIAUCNS',         # Consumer Price Index
            'FED_FUNDS': 'FEDFUNDS',         # Federal Funds Rate
            'YIELD_10Y': 'GS10',             # 10-Year Treasury Rate
            'YIELD_2Y': 'GS2',               # 2-Year Treasury Rate
            'VIX': 'VIXCLS',                 # VIX Volatility Index
            'DOLLAR_INDEX': 'DTWEXBGS',      # Dollar Index
            'OIL_PRICE': 'DCOILWTICO',       # Oil Price
            'GOLD_PRICE': 'GOLDAMGBD228NLBM' # Gold Price
        }
        
        economic_data = pd.DataFrame()
        
        for name, series_id in indicators.items():
            try:
                data = self.get_fred_data(series_id, start_date, end_date)
                if not data.empty:
                    economic_data[name] = data
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                print(f"Failed to fetch {name}: {e}")
                continue
        
        return economic_data
    
    def get_sector_etf_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch sector ETF data for sector rotation analysis"""
        
        sector_etfs = {
            'XLK': 'Technology',
            'XLF': 'Financials', 
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrials',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLU': 'Utilities',
            'XLB': 'Materials',
            'XLRE': 'Real Estate'
        }
        
        sector_data = pd.DataFrame()
        
        for etf, sector in sector_etfs.items():
            try:
                ticker = yf.Ticker(etf)
                hist = ticker.history(start=start_date, end=end_date)
                if not hist.empty:
                    sector_data[sector] = hist['Close'].pct_change()
                time.sleep(0.1)
            except Exception as e:
                print(f"Failed to fetch {etf}: {e}")
                continue
        
        return sector_data
    
    def _generate_dummy_economic_data(self, series_id: str, start_date: str, end_date: str) -> pd.Series:
        """Generate dummy economic data for demo purposes"""
        
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Create realistic dummy data based on series type
        if 'GDP' in series_id:
            # GDP growth (quarterly, around 2-3%)
            data = np.random.normal(2.5, 0.5, len(dates))
        elif 'UNRATE' in series_id:
            # Unemployment rate (4-8%)
            data = np.random.normal(5.5, 1.0, len(dates))
            data = np.clip(data, 3, 10)
        elif 'CPI' in series_id:
            # Inflation (1-4%)
            data = np.random.normal(2.5, 0.8, len(dates))
            data = np.clip(data, 0, 6)
        elif 'FEDFUNDS' in series_id:
            # Fed funds rate (0-6%)
            data = np.random.normal(2.0, 1.0, len(dates))
            data = np.clip(data, 0, 6)
        elif 'GS10' in series_id:
            # 10-year treasury (1-5%)
            data = np.random.normal(3.0, 0.8, len(dates))
            data = np.clip(data, 1, 6)
        elif 'VIX' in series_id:
            # VIX (10-40)
            data = np.random.lognormal(2.8, 0.3, len(dates))
            data = np.clip(data, 10, 50)
        else:
            # Generic economic indicator
            data = np.random.normal(100, 10, len(dates))
        
        return pd.Series(data, index=dates, name=series_id)

class SentimentDataProvider:
    """
    Fetch sentiment and alternative data sources
    """
    
    def __init__(self):
        self.cache = {}
        
    def get_fear_greed_index(self) -> Dict:
        """Fetch CNN Fear & Greed Index"""
        
        try:
            # This is a simplified version - in practice, you'd scrape or use an API
            # For demo purposes, we'll generate realistic dummy data
            
            current_time = datetime.now()
            
            # Generate fear/greed index (0-100, where 0 is extreme fear, 100 is extreme greed)
            base_index = 50
            volatility = np.random.normal(0, 15)
            fear_greed = np.clip(base_index + volatility, 0, 100)
            
            return {
                'value': fear_greed,
                'text': self._interpret_fear_greed(fear_greed),
                'timestamp': current_time,
                'components': {
                    'stock_price_momentum': np.random.uniform(20, 80),
                    'stock_price_strength': np.random.uniform(20, 80),
                    'stock_price_breadth': np.random.uniform(20, 80),
                    'put_call_ratios': np.random.uniform(20, 80),
                    'junk_bond_demand': np.random.uniform(20, 80),
                    'market_volatility': np.random.uniform(20, 80),
                    'safe_haven_demand': np.random.uniform(20, 80)
                }
            }
            
        except Exception as e:
            print(f"Error fetching Fear & Greed Index: {e}")
            return {'value': 50, 'text': 'Neutral', 'timestamp': datetime.now()}
    
    def get_crypto_sentiment(self, symbols: List[str] = ['BTC', 'ETH']) -> pd.DataFrame:
        """Get cryptocurrency sentiment indicators"""
        
        # For demo purposes, generate sentiment scores
        sentiment_data = {}
        
        for symbol in symbols:
            # Generate sentiment metrics
            sentiment_data[f'{symbol}_sentiment'] = np.random.uniform(-1, 1)  # -1 to 1 scale
            sentiment_data[f'{symbol}_social_volume'] = np.random.randint(1000, 10000)
            sentiment_data[f'{symbol}_news_sentiment'] = np.random.uniform(-1, 1)
        
        return pd.DataFrame([sentiment_data], index=[datetime.now()])
    
    def get_insider_trading_data(self, symbol: str) -> pd.DataFrame:
        """Get insider trading activity (simplified)"""
        
        # Generate dummy insider trading data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        insider_data = []
        for date in dates:
            if np.random.random() < 0.1:  # 10% chance of insider activity
                activity = {
                    'date': date,
                    'insider_type': np.random.choice(['CEO', 'CFO', 'Director', 'Officer']),
                    'transaction_type': np.random.choice(['Buy', 'Sell']),
                    'shares': np.random.randint(1000, 50000),
                    'price': np.random.uniform(50, 200),
                    'value': 0
                }
                activity['value'] = activity['shares'] * activity['price']
                insider_data.append(activity)
        
        return pd.DataFrame(insider_data)
    
    def get_earnings_sentiment(self, symbols: List[str]) -> pd.DataFrame:
        """Get earnings-related sentiment indicators"""
        
        earnings_data = {}
        
        for symbol in symbols:
            # Generate earnings sentiment metrics
            earnings_data[f'{symbol}_earnings_surprise'] = np.random.normal(0, 0.1)  # Earnings surprise %
            earnings_data[f'{symbol}_estimate_revisions'] = np.random.randint(-5, 5)  # Number of revisions
            earnings_data[f'{symbol}_analyst_rating'] = np.random.uniform(1, 5)  # 1-5 rating scale
            earnings_data[f'{symbol}_price_target_change'] = np.random.normal(0, 0.05)  # Price target change %
        
        return pd.DataFrame([earnings_data], index=[datetime.now()])
    
    def _interpret_fear_greed(self, value: float) -> str:
        """Interpret fear and greed index value"""
        
        if value <= 25:
            return "Extreme Fear"
        elif value <= 45:
            return "Fear"
        elif value <= 55:
            return "Neutral"
        elif value <= 75:
            return "Greed"
        else:
            return "Extreme Greed"

class MarketRegimeDetector:
    """
    Detect market regimes using alternative data
    """
    
    def __init__(self):
        self.regimes = ['Bull Market', 'Bear Market', 'High Volatility', 'Low Volatility', 'Recession', 'Recovery']
        
    def detect_regime(self, market_data: pd.DataFrame, 
                     economic_data: pd.DataFrame,
                     sentiment_data: pd.DataFrame = None) -> Dict:
        """Detect current market regime using multiple data sources"""
        
        regime_scores = {}
        
        # Market-based signals
        if 'Close' in market_data.columns:
            recent_returns = market_data['Close'].pct_change().tail(60)  # Last 60 days
            volatility = recent_returns.std() * np.sqrt(252)
            
            # Trend analysis
            sma_50 = market_data['Close'].rolling(50).mean()
            sma_200 = market_data['Close'].rolling(200).mean()
            
            current_price = market_data['Close'].iloc[-1]
            trend_signal = (current_price > sma_50.iloc[-1]) and (sma_50.iloc[-1] > sma_200.iloc[-1])
            
            if trend_signal:
                regime_scores['Bull Market'] = 0.7
            else:
                regime_scores['Bear Market'] = 0.7
            
            # Volatility regime
            if volatility > 0.25:
                regime_scores['High Volatility'] = 0.8
            else:
                regime_scores['Low Volatility'] = 0.6
        
        # Economic indicators
        if not economic_data.empty:
            # Check unemployment trend
            if 'UNEMPLOYMENT' in economic_data.columns:
                recent_unemployment = economic_data['UNEMPLOYMENT'].dropna().tail(6)  # Last 6 months
                if len(recent_unemployment) >= 2:
                    unemployment_trend = recent_unemployment.iloc[-1] - recent_unemployment.iloc[0]
                    
                    if unemployment_trend > 0.5:  # Rising unemployment
                        regime_scores['Recession'] = regime_scores.get('Recession', 0) + 0.3
                    elif unemployment_trend < -0.5:  # Falling unemployment
                        regime_scores['Recovery'] = regime_scores.get('Recovery', 0) + 0.3
            
            # Check GDP growth
            if 'GDP' in economic_data.columns:
                recent_gdp = economic_data['GDP'].dropna().tail(4)  # Last 4 quarters
                if len(recent_gdp) >= 2:
                    gdp_growth = recent_gdp.iloc[-1]
                    
                    if gdp_growth < 0:
                        regime_scores['Recession'] = regime_scores.get('Recession', 0) + 0.4
                    elif gdp_growth > 3:
                        regime_scores['Recovery'] = regime_scores.get('Recovery', 0) + 0.4
        
        # Sentiment-based signals
        if sentiment_data is not None and not sentiment_data.empty:
            # Add sentiment-based regime detection logic
            pass
        
        # Determine primary regime
        if regime_scores:
            primary_regime = max(regime_scores, key=regime_scores.get)
            confidence = regime_scores[primary_regime]
        else:
            primary_regime = 'Neutral'
            confidence = 0.5
        
        return {
            'primary_regime': primary_regime,
            'confidence': confidence,
            'regime_scores': regime_scores,
            'timestamp': datetime.now()
        }
    
    def get_regime_allocations(self, regime: str) -> Dict[str, float]:
        """Get suggested asset allocations for different regimes"""
        
        allocations = {
            'Bull Market': {
                'stocks': 0.70,
                'bonds': 0.20,
                'commodities': 0.05,
                'cash': 0.05
            },
            'Bear Market': {
                'stocks': 0.30,
                'bonds': 0.50,
                'commodities': 0.05,
                'cash': 0.15
            },
            'High Volatility': {
                'stocks': 0.40,
                'bonds': 0.40,
                'commodities': 0.10,
                'cash': 0.10
            },
            'Low Volatility': {
                'stocks': 0.65,
                'bonds': 0.25,
                'commodities': 0.05,
                'cash': 0.05
            },
            'Recession': {
                'stocks': 0.25,
                'bonds': 0.60,
                'commodities': 0.05,
                'cash': 0.10
            },
            'Recovery': {
                'stocks': 0.75,
                'bonds': 0.15,
                'commodities': 0.05,
                'cash': 0.05
            }
        }
        
        return allocations.get(regime, allocations['Bull Market'])

class AlternativeDataIntegrator:
    """
    Integrate alternative data sources into portfolio optimization
    """
    
    def __init__(self):
        self.economic_provider = EconomicDataProvider()
        self.sentiment_provider = SentimentDataProvider()
        self.regime_detector = MarketRegimeDetector()
        
    def create_alternative_features(self, 
                                  market_data: pd.DataFrame,
                                  start_date: str,
                                  end_date: str) -> pd.DataFrame:
        """Create features from alternative data sources"""
        
        # Get economic data
        economic_data = self.economic_provider.get_economic_indicators(start_date, end_date)
        
        # Get sector rotation data
        sector_data = self.economic_provider.get_sector_etf_data(start_date, end_date)
        
        # Combine all data sources
        feature_data = market_data.copy()
        
        # Add economic indicators (forward fill to match daily frequency)
        if not economic_data.empty:
            economic_daily = economic_data.resample('D').ffill()
            
            # Align with market data
            common_dates = feature_data.index.intersection(economic_daily.index)
            if len(common_dates) > 0:
                for col in economic_daily.columns:
                    feature_data[f'econ_{col}'] = economic_daily[col].reindex(feature_data.index, method='ffill')
        
        # Add sector momentum features
        if not sector_data.empty:
            sector_aligned = sector_data.reindex(feature_data.index, method='ffill')
            
            for col in sector_aligned.columns:
                # Sector momentum (20-day returns)
                feature_data[f'sector_momentum_{col}'] = sector_aligned[col].rolling(20).mean()
                
                # Sector volatility
                feature_data[f'sector_vol_{col}'] = sector_aligned[col].rolling(20).std()
        
        # Add regime indicators
        regime_info = self.regime_detector.detect_regime(market_data, economic_data)
        
        # Create regime dummy variables
        for regime in self.regime_detector.regimes:
            feature_data[f'regime_{regime}'] = (regime_info['primary_regime'] == regime).astype(int)
        
        # Add regime confidence
        feature_data['regime_confidence'] = regime_info['confidence']
        
        return feature_data
    
    def get_regime_adjusted_weights(self, 
                                  base_weights: pd.Series,
                                  market_data: pd.DataFrame,
                                  economic_data: pd.DataFrame) -> pd.Series:
        """Adjust portfolio weights based on market regime"""
        
        # Detect current regime
        regime_info = self.regime_detector.detect_regime(market_data, economic_data)
        current_regime = regime_info['primary_regime']
        confidence = regime_info['confidence']
        
        # Get regime-specific allocations
        regime_allocations = self.regime_detector.get_regime_allocations(current_regime)
        
        # Adjust weights based on regime (simple approach)
        adjusted_weights = base_weights.copy()
        
        # Apply regime tilt (weighted average between base weights and regime allocation)
        regime_weight = confidence * 0.3  # Max 30% adjustment based on regime
        
        # This is a simplified approach - in practice, you'd map assets to categories
        if confidence > 0.7:
            # High confidence regime signal - apply stronger adjustment
            adjustment_factor = regime_allocations.get('stocks', 0.6)
            adjusted_weights = adjusted_weights * (1 + regime_weight * (adjustment_factor - 0.6))
            adjusted_weights = adjusted_weights / adjusted_weights.sum()  # Renormalize
        
        return adjusted_weights