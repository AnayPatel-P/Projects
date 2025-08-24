import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class RiskLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    EXTREME = 4

@dataclass
class RiskMetrics:
    var_95: float  # Value at Risk at 95% confidence
    var_99: float  # Value at Risk at 99% confidence
    cvar_95: float  # Conditional VaR at 95%
    volatility: float
    beta: float
    sharpe_ratio: float
    max_drawdown: float
    correlation_risk: float

class RiskManager:
    def __init__(self, 
                 max_position_size: float = 0.1,
                 max_portfolio_risk: float = 0.02,
                 max_daily_loss: float = 0.05,
                 var_confidence: float = 0.95):
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.max_daily_loss = max_daily_loss
        self.var_confidence = var_confidence
        self.position_limits = {}
        self.risk_metrics = {}
        
    def calculate_position_size(self, 
                              symbol: str,
                              price: float,
                              portfolio_value: float,
                              volatility: float,
                              confidence: float = 1.0) -> float:
        # Kelly Criterion based position sizing
        win_prob = min(max(confidence, 0.5), 0.9)  # Bounded between 50-90%
        avg_win = 0.02  # Assume 2% average win
        avg_loss = 0.015  # Assume 1.5% average loss
        
        kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))
        
        # Adjust for volatility
        vol_adjustment = 1 / (1 + volatility * 10)  # Reduce size for high volatility
        
        # Calculate position size
        target_value = portfolio_value * kelly_fraction * vol_adjustment
        shares = int(target_value / price)
        
        return max(shares, 0)
    
    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> Tuple[float, float]:
        if len(returns) < 30:
            return 0.0, 0.0
        
        # Historical VaR
        var = np.percentile(returns, (1 - confidence) * 100)
        
        # Conditional VaR (Expected Shortfall)
        cvar = returns[returns <= var].mean()
        
        return abs(var), abs(cvar)
    
    def calculate_portfolio_risk(self, 
                               positions: Dict[str, float],
                               returns_data: Dict[str, pd.Series],
                               correlation_matrix: Optional[pd.DataFrame] = None) -> RiskMetrics:
        if not positions or not returns_data:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        # Align returns data
        common_symbols = set(positions.keys()) & set(returns_data.keys())
        if not common_symbols:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        # Create returns matrix
        returns_df = pd.DataFrame({symbol: returns_data[symbol] for symbol in common_symbols})
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 30:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        # Portfolio weights
        total_value = sum(abs(positions[symbol]) for symbol in common_symbols)
        if total_value == 0:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        weights = np.array([positions[symbol] / total_value for symbol in common_symbols])
        
        # Portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        # Calculate metrics
        var_95, cvar_95 = self.calculate_var(portfolio_returns, 0.95)
        var_99, _ = self.calculate_var(portfolio_returns, 0.99)
        
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe = (portfolio_returns.mean() * 252) / (volatility + 1e-8)
        
        # Calculate max drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        drawdowns = 1 - cumulative / cumulative.expanding().max()
        max_drawdown = drawdowns.max()
        
        # Beta calculation (vs equal-weighted portfolio)
        market_returns = returns_df.mean(axis=1)
        if market_returns.std() != 0:
            beta = np.cov(portfolio_returns, market_returns)[0, 1] / market_returns.var()
        else:
            beta = 1.0
        
        # Correlation risk (average correlation)
        if correlation_matrix is not None:
            correlation_risk = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
        else:
            correlation_risk = returns_df.corr().values[np.triu_indices_from(returns_df.corr().values, k=1)].mean()
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            volatility=volatility,
            beta=beta,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            correlation_risk=correlation_risk
        )
    
    def assess_risk_level(self, risk_metrics: RiskMetrics) -> RiskLevel:
        risk_score = 0
        
        # VaR risk
        if risk_metrics.var_95 > 0.05:
            risk_score += 2
        elif risk_metrics.var_95 > 0.03:
            risk_score += 1
        
        # Volatility risk
        if risk_metrics.volatility > 0.4:
            risk_score += 2
        elif risk_metrics.volatility > 0.25:
            risk_score += 1
        
        # Drawdown risk
        if risk_metrics.max_drawdown > 0.3:
            risk_score += 2
        elif risk_metrics.max_drawdown > 0.2:
            risk_score += 1
        
        # Correlation risk
        if risk_metrics.correlation_risk > 0.8:
            risk_score += 2
        elif risk_metrics.correlation_risk > 0.6:
            risk_score += 1
        
        if risk_score >= 6:
            return RiskLevel.EXTREME
        elif risk_score >= 4:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def check_risk_limits(self, 
                         positions: Dict[str, float],
                         portfolio_value: float,
                         daily_pnl: float) -> Dict[str, bool]:
        checks = {
            'position_size_ok': True,
            'portfolio_risk_ok': True,
            'daily_loss_ok': True,
            'concentration_ok': True
        }
        
        # Check position size limits
        for symbol, position_value in positions.items():
            position_pct = abs(position_value) / portfolio_value
            if position_pct > self.max_position_size:
                checks['position_size_ok'] = False
                break
        
        # Check daily loss limit
        daily_loss_pct = abs(daily_pnl) / portfolio_value if daily_pnl < 0 else 0
        if daily_loss_pct > self.max_daily_loss:
            checks['daily_loss_ok'] = False
        
        # Check concentration risk
        if len(positions) < 5:  # Too concentrated
            total_value = sum(abs(pos) for pos in positions.values())
            largest_position = max(abs(pos) for pos in positions.values()) if positions else 0
            if largest_position / total_value > 0.4:
                checks['concentration_ok'] = False
        
        return checks
    
    def generate_risk_recommendations(self, 
                                    risk_metrics: RiskMetrics,
                                    risk_level: RiskLevel,
                                    risk_checks: Dict[str, bool]) -> List[str]:
        recommendations = []
        
        if risk_level in [RiskLevel.HIGH, RiskLevel.EXTREME]:
            recommendations.append("Consider reducing overall portfolio exposure")
            
        if risk_metrics.volatility > 0.3:
            recommendations.append("High volatility detected - consider defensive positions")
            
        if risk_metrics.max_drawdown > 0.2:
            recommendations.append("Significant drawdown risk - implement stop losses")
            
        if risk_metrics.correlation_risk > 0.7:
            recommendations.append("High correlation between positions - diversify holdings")
            
        if risk_metrics.var_95 > 0.04:
            recommendations.append("High VaR - consider hedging strategies")
            
        if not risk_checks['position_size_ok']:
            recommendations.append("Position size exceeds limits - reduce exposure")
            
        if not risk_checks['daily_loss_ok']:
            recommendations.append("Daily loss limit breached - halt trading")
            
        if not risk_checks['concentration_ok']:
            recommendations.append("Portfolio too concentrated - add diversification")
            
        if risk_metrics.sharpe_ratio < 0.5:
            recommendations.append("Poor risk-adjusted returns - review strategy")
            
        return recommendations
    
    def calculate_optimal_hedge(self, 
                              portfolio_beta: float,
                              portfolio_value: float,
                              hedge_instrument_beta: float = -1.0) -> float:
        # Calculate hedge ratio to neutralize market risk
        hedge_ratio = -portfolio_beta / hedge_instrument_beta
        hedge_notional = portfolio_value * hedge_ratio
        
        return hedge_notional
    
    def stress_test(self, 
                   positions: Dict[str, float],
                   returns_data: Dict[str, pd.Series],
                   shock_scenarios: Dict[str, float]) -> Dict[str, float]:
        results = {}
        
        # Default stress scenarios if none provided
        if not shock_scenarios:
            shock_scenarios = {
                'market_crash_20pct': -0.20,
                'market_crash_10pct': -0.10,
                'volatility_spike_2x': 2.0,
                'interest_rate_spike': 0.02
            }
        
        for scenario, shock_value in shock_scenarios.items():
            portfolio_impact = 0
            
            for symbol, position_value in positions.items():
                if symbol in returns_data:
                    returns = returns_data[symbol]
                    if 'crash' in scenario:
                        # Apply direct price shock
                        impact = position_value * shock_value
                    elif 'volatility' in scenario:
                        # Estimate impact of increased volatility
                        current_vol = returns.std()
                        impact = position_value * current_vol * (shock_value - 1) * -0.5
                    else:
                        # Generic impact
                        impact = position_value * shock_value * 0.1
                    
                    portfolio_impact += impact
            
            results[scenario] = portfolio_impact
        
        return results