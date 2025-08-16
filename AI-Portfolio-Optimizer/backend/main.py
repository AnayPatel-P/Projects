from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Optional
import os
import logging
from dotenv import load_dotenv
import re

from data_loader import fetch_price_data
from optimizer import optimize_portfolio

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Portfolio Optimizer API", version="1.0.0")

# Environment-based CORS configuration
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173,http://localhost:5174").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class OptimizeRequest(BaseModel):
    risk_level: str
    tickers: List[str]
    
    @validator('risk_level')
    def validate_risk_level(cls, v):
        if v not in ['low', 'medium', 'high']:
            raise ValueError('risk_level must be one of: low, medium, high')
        return v
    
    @validator('tickers')
    def validate_tickers(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one ticker must be provided')
        if len(v) > 20:
            raise ValueError('Maximum 20 tickers allowed')
        
        # Validate ticker format (letters only, 1-5 characters)
        ticker_pattern = re.compile(r'^[A-Z]{1,5}$')
        cleaned_tickers = []
        
        for ticker in v:
            cleaned = ticker.strip().upper()
            if not ticker_pattern.match(cleaned):
                raise ValueError(f'Invalid ticker format: {ticker}. Use 1-5 letter symbols only.')
            cleaned_tickers.append(cleaned)
        
        return cleaned_tickers

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "AI Portfolio Optimizer API is running"}

@app.post("/optimize")
def optimize(req: OptimizeRequest):
    try:
        logger.info(f"Optimization request received for tickers: {req.tickers}, risk level: {req.risk_level}")
        
        # Fetch and validate price data
        prices = fetch_price_data(req.tickers)
        
        if prices.empty:
            raise HTTPException(status_code=400, detail="No valid price data found for the provided tickers")
        
        if len(prices.columns) < 2:
            raise HTTPException(status_code=400, detail="At least 2 valid tickers are required for optimization")
        
        # Check for sufficient historical data (at least 252 trading days)
        if len(prices) < 252:
            logger.warning(f"Limited historical data: {len(prices)} days. Results may be less reliable.")
        
        # Perform optimization
        result = optimize_portfolio(prices, req.risk_level)
        
        # Normalize price history for percentage-based visualization
        normalized = prices / prices.iloc[0] * 100  # Convert to percentage starting at 100
        
        # Reset index and format Date for JSON serialization
        normalized = normalized.reset_index()
        normalized["Date"] = normalized["Date"].dt.strftime("%Y-%m-%d")
        history_dict = normalized.to_dict(orient="records")
        
        logger.info(f"Optimization completed successfully for {len(req.tickers)} tickers")
        
        return {
            **result,
            "price_history": history_dict,
            "data_points": len(prices),
            "valid_tickers": list(prices.columns)
        }
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during optimization")


