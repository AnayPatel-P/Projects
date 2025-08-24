import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import requests
import time
from alpha_vantage.timeseries import TimeSeries
import os
from dotenv import load_dotenv

load_dotenv()

class DataFetcher:
    def __init__(self, alpha_vantage_key: Optional[str] = None):
        self.alpha_vantage_key = alpha_vantage_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        if self.alpha_vantage_key:
            self.ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
        
    def fetch_yahoo_data(self, 
                        symbols: List[str], 
                        start_date: str, 
                        end_date: str,
                        interval: str = '1d') -> Dict[str, pd.DataFrame]:
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval=interval)
                if not df.empty:
                    data[symbol] = df
                    print(f"Fetched {len(df)} records for {symbol}")
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                print(f"Error fetching {symbol}: {str(e)}")
        return data
    
    def fetch_alpha_vantage_data(self, symbol: str, outputsize: str = 'full') -> pd.DataFrame:
        if not self.alpha_vantage_key:
            raise ValueError("Alpha Vantage API key not provided")
        
        try:
            data, meta_data = self.ts.get_daily_adjusted(symbol=symbol, outputsize=outputsize)
            data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividend', 'Split']
            return data.sort_index()
        except Exception as e:
            print(f"Error fetching {symbol} from Alpha Vantage: {str(e)}")
            return pd.DataFrame()
    
    def get_sp500_symbols(self) -> List[str]:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        try:
            tables = pd.read_html(url)
            df = tables[0]
            return df['Symbol'].str.replace('.', '-').tolist()
        except Exception as e:
            print(f"Error fetching S&P 500 symbols: {str(e)}")
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
    
    def fetch_market_data(self, 
                         symbols: List[str], 
                         start_date: str,
                         end_date: str,
                         source: str = 'yahoo') -> Dict[str, pd.DataFrame]:
        if source == 'yahoo':
            return self.fetch_yahoo_data(symbols, start_date, end_date)
        elif source == 'alpha_vantage' and len(symbols) == 1:
            return {symbols[0]: self.fetch_alpha_vantage_data(symbols[0])}
        else:
            raise ValueError("Unsupported data source or multiple symbols for Alpha Vantage")
    
    def save_data(self, data: Dict[str, pd.DataFrame], file_path: str):
        for symbol, df in data.items():
            df.to_csv(f"{file_path}/{symbol}.csv")
    
    def load_data(self, file_path: str, symbol: str) -> pd.DataFrame:
        try:
            return pd.read_csv(f"{file_path}/{symbol}.csv", index_col=0, parse_dates=True)
        except FileNotFoundError:
            print(f"File not found: {file_path}/{symbol}.csv")
            return pd.DataFrame()