import yfinance as yf
import pandas as pd
import logging
from cachetools import TTLCache
from datetime import datetime, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)

# Cache for storing price data (TTL = 1 hour)
price_cache = TTLCache(maxsize=100, ttl=3600)

def fetch_price_data(tickers: List[str], start_date: str = "2020-01-01", end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch historical price data for given tickers with caching and validation.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date for historical data
        end_date: End date for historical data (defaults to today)
    
    Returns:
        DataFrame with historical price data
    
    Raises:
        ValueError: If no valid data is found
    """
    logger.info(f"Fetching price data for tickers: {tickers}")
    
    # Create cache key
    cache_key = f"{'-'.join(sorted(tickers))}_{start_date}_{end_date}"
    
    # Check cache first
    if cache_key in price_cache:
        logger.info("Returning cached price data")
        return price_cache[cache_key]
    
    try:
        # Download data with timeout and error handling
        data = yf.download(
            tickers, 
            start=start_date, 
            end=end_date, 
            auto_adjust=True,
            progress=False
        )
        
        if data.empty:
            raise ValueError("No data downloaded for any ticker")
        
        # Handle single vs multiple ticker data structure
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.levels[0]:
                data = data['Close']
            else:
                raise ValueError("Close price data not available")
        else:
            # Single ticker case
            if 'Close' in data.columns:
                data = data[['Close']].rename(columns={'Close': tickers[0]})
            else:
                raise ValueError("Close price data not available")
        
        # Remove tickers with insufficient data (less than 30 days)
        min_data_points = 30
        valid_tickers = []
        
        for col in data.columns:
            valid_data = data[col].dropna()
            if len(valid_data) >= min_data_points:
                valid_tickers.append(col)
            else:
                logger.warning(f"Ticker {col} has insufficient data ({len(valid_data)} days)")
        
        if not valid_tickers:
            raise ValueError("No tickers have sufficient historical data")
        
        # Keep only valid tickers
        data = data[valid_tickers]
        
        # Remove rows where all values are NaN
        data = data.dropna(how='all')
        
        # Forward fill missing values (up to 5 consecutive days)
        data = data.ffill(limit=5)
        
        # Drop any remaining NaN values
        data = data.dropna()
        
        if data.empty:
            raise ValueError("No valid price data after cleaning")
        
        logger.info(f"Successfully downloaded data for {len(data.columns)} tickers: {list(data.columns)}")
        logger.info(f"Data range: {data.index[0].date()} to {data.index[-1].date()} ({len(data)} days)")
        
        # Cache the result
        price_cache[cache_key] = data
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching price data: {str(e)}")
        raise ValueError(f"Failed to fetch price data: {str(e)}")

def validate_tickers(tickers: List[str]) -> List[str]:
    """
    Validate ticker symbols by checking if they exist and have recent data.
    
    Args:
        tickers: List of ticker symbols to validate
    
    Returns:
        List of valid ticker symbols
    """
    logger.info(f"Validating tickers: {tickers}")
    valid_tickers = []
    
    for ticker in tickers:
        try:
            # Try to fetch recent data (last 30 days)
            recent_data = yf.download(
                ticker, 
                period="1mo", 
                progress=False
            )
            
            if not recent_data.empty and len(recent_data) > 5:
                valid_tickers.append(ticker)
                logger.info(f"Ticker {ticker} is valid")
            else:
                logger.warning(f"Ticker {ticker} has no recent data")
                
        except Exception as e:
            logger.warning(f"Ticker {ticker} validation failed: {str(e)}")
    
    return valid_tickers

def calculate_return_stats(price_df, frequency="biweekly"):
    if frequency == "biweekly":
        # Resample to biweekly using last Friday of each 2-week period
        resampled = price_df.resample('2W-FRI').last()
        factor = 26  # 26 biweekly periods in a year
    else:
        resampled = price_df
        factor = 252  # Trading days in a year

    returns = resampled.pct_change().dropna()

    if returns.empty:
        raise ValueError("No valid return data after resampling.")

    annualized_returns = returns.mean() * factor
    annualized_volatility = returns.std() * (factor ** 0.5)
    cov_matrix = returns.cov() * factor

    return {
        f"{frequency}_returns": returns,
        "annualized_returns": annualized_returns,
        "annualized_volatility": annualized_volatility,
        "cov_matrix": cov_matrix
    }

def export_returns_to_csv(returns_df, filename="biweekly_returns.csv"):
    returns_df.index = returns_df.index.strftime('%Y-%m-%d')  # Format index as string dates
    returns_df.to_csv(filename)
    print(f"[INFO] Exported biweekly returns to '{filename}'")
