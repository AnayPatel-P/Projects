import pandas as pd
import numpy as np
import logging
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

def optimize_portfolio(
    price_df: pd.DataFrame, 
    risk_level: str = "medium", 
    max_weight: float = 0.4,
    min_weight: float = 0.01,
    sector_constraints: Optional[Dict] = None
) -> Dict:
    """
    Optimize portfolio using Modern Portfolio Theory with enhanced constraints.
    
    Args:
        price_df: Historical price data
        risk_level: Risk preference (low, medium, high)
        max_weight: Maximum weight for any single asset
        min_weight: Minimum weight for any single asset
        sector_constraints: Optional sector-based constraints
    
    Returns:
        Dictionary containing optimization results
    """
    logger.info(f"Optimizing portfolio with {len(price_df.columns)} assets, risk level: {risk_level}")
    
    try:
        # Compute expected returns using multiple methods for robustness
        mu_historical = expected_returns.mean_historical_return(price_df, frequency=252)
        mu_capm = expected_returns.capm_return(price_df, frequency=252)
        
        # Use ensemble of return estimates (weighted average)
        mu = 0.7 * mu_historical + 0.3 * mu_capm
        
        # Compute covariance matrix with robust estimator
        S = risk_models.sample_cov(price_df, frequency=252)
        
        # Add slight regularization to covariance matrix for numerical stability
        S = risk_models.fix_nonpositive_semidefinite(S)
        
        # Initialize efficient frontier
        ef = EfficientFrontier(mu, S)
        
        # Add weight constraints
        ef.add_constraint(lambda w: w >= min_weight)
        ef.add_constraint(lambda w: w <= max_weight)
        
        # Add regularization for smoother allocations
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        
        # Risk-based optimization strategy
        if risk_level == "low":
            # Conservative: minimize volatility
            weights = ef.min_volatility()
            
        elif risk_level == "high":
            # Aggressive: maximize Sharpe ratio
            weights = ef.max_sharpe()
            
        else:  # medium risk
            # Balanced: target moderate volatility with good returns
            try:
                # Try to target 15% annual volatility
                weights = ef.efficient_risk(target_volatility=0.15)
            except ValueError:
                try:
                    # Fallback to 12% volatility
                    weights = ef.efficient_risk(target_volatility=0.12)
                except ValueError:
                    # Final fallback to max Sharpe
                    logger.warning("Could not achieve target volatility, using max Sharpe")
                    weights = ef.max_sharpe()
        
        # Clean weights (remove tiny allocations)
        cleaned_weights = ef.clean_weights(cutoff=0.01)
        
        # Calculate portfolio performance
        perf = ef.portfolio_performance(verbose=False)
        expected_return, volatility, sharpe_ratio = perf
        
        # Calculate additional risk metrics
        portfolio_returns = (price_df.pct_change().dropna() @ pd.Series(cleaned_weights)).dropna()
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(portfolio_returns, 5)
        
        # Maximum Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Diversification ratio
        individual_volatilities = price_df.pct_change().std() * np.sqrt(252)
        weighted_vol = sum(cleaned_weights[asset] * individual_volatilities[asset] 
                          for asset in cleaned_weights.keys())
        diversification_ratio = weighted_vol / volatility if volatility > 0 else 0
        
        logger.info(f"Portfolio optimization completed: Return={expected_return:.2%}, "
                   f"Volatility={volatility:.2%}, Sharpe={sharpe_ratio:.2f}")
        
        return {
            "weights": cleaned_weights,
            "expected_return": float(expected_return),
            "expected_volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "var_95": float(var_95),
            "max_drawdown": float(max_drawdown),
            "diversification_ratio": float(diversification_ratio),
            "num_assets": len([w for w in cleaned_weights.values() if w > 0.01])
        }
        
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {str(e)}")
        raise ValueError(f"Optimization failed: {str(e)}")

def calculate_discrete_allocation(
    weights: Dict[str, float], 
    latest_prices: Dict[str, float], 
    total_portfolio_value: float = 10000
) -> Dict:
    """
    Calculate discrete share allocation for a given portfolio value.
    
    Args:
        weights: Portfolio weights
        latest_prices: Latest prices for each asset
        total_portfolio_value: Total value to invest
    
    Returns:
        Dictionary with share allocations and leftover cash
    """
    try:
        da = DiscreteAllocation(weights, latest_prices, total_value=total_portfolio_value)
        allocation, leftover = da.greedy_portfolio()
        
        return {
            "allocation": allocation,
            "leftover_cash": leftover,
            "total_invested": total_portfolio_value - leftover
        }
    except Exception as e:
        logger.error(f"Discrete allocation failed: {str(e)}")
        return {"error": str(e)}



def export_weights_to_csv(weights_dict, filename="optimized_weights.csv"):
    df = pd.DataFrame(list(weights_dict.items()), columns=["Ticker", "Weight"])
    df["Weight"] = (df["Weight"] * 100).round(2)  # Convert to %
    df.to_csv(filename, index=False)
    print(f"[INFO] Exported optimized weights to '{filename}'")
