# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from data_loader import fetch_price_data
from optimizer import optimize_portfolio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizeRequest(BaseModel):
    risk_level: str

@app.post("/optimize")
def optimize(req: OptimizeRequest):
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JNJ', 'XOM', 'JPM', 'NVDA', 'PG']
    prices = fetch_price_data(tickers)
    result = optimize_portfolio(prices, req.risk_level)
    return result
