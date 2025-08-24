import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class RiskMetric(Enum):
    VAR = "value_at_risk"
    CVAR = "conditional_var"
    MAX_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    BETA = "beta"
    ALPHA = "alpha"
    TRACKING_ERROR = "tracking_error"
    INFORMATION_RATIO = "information_ratio"

@dataclass
class RiskAlert:
    timestamp: datetime
    metric: RiskMetric
    current_value: float
    threshold: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    symbol: Optional[str] = None
    strategy: Optional[str] = None

class AdvancedRiskModel:
    """
    Advanced risk modeling and measurement system
    """
    
    def __init__(self, confidence_level: float = 0.05, lookback_period: int = 252):
        self.confidence_level = confidence_level
        self.lookback_period = lookback_period
        self.risk_alerts = []
        self.risk_limits = self._setup_default_limits()
        
    def _setup_default_limits(self) -> Dict[RiskMetric, float]:
        """Setup default risk limits"""
        return {
            RiskMetric.VAR: -0.05,  # 5% daily VaR limit
            RiskMetric.MAX_DRAWDOWN: -0.20,  # 20% max drawdown
            RiskMetric.SHARPE_RATIO: 0.5,  # Minimum Sharpe ratio
            RiskMetric.BETA: 1.5,  # Maximum beta
            RiskMetric.TRACKING_ERROR: 0.10  # 10% tracking error limit
        }
    
    def calculate_var(self, returns: pd.Series, method: str = 'historical') -> float:
        """
        Calculate Value at Risk using different methods
        """
        if len(returns) < 30:
            return np.nan
        
        if method == 'historical':
            return np.percentile(returns.dropna(), self.confidence_level * 100)
        
        elif method == 'parametric':
            mu = returns.mean()
            sigma = returns.std()
            return stats.norm.ppf(self.confidence_level, mu, sigma)
        
        elif method == 'monte_carlo':
            mu = returns.mean()
            sigma = returns.std()
            simulated = np.random.normal(mu, sigma, 10000)
            return np.percentile(simulated, self.confidence_level * 100)
        
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    def calculate_cvar(self, returns: pd.Series) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        """
        if len(returns) < 30:
            return np.nan
        
        var = self.calculate_var(returns, method='historical')
        return returns[returns <= var].mean()
    
    def calculate_max_drawdown(self, returns: pd.Series) -> Tuple[float, datetime, datetime]:
        """
        Calculate maximum drawdown and its duration
        """
        if len(returns) < 2:
            return np.nan, None, None
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_end = drawdown.idxmin()
        
        # Find start of max drawdown period
        max_dd_start = running_max.loc[:max_dd_end].idxmax()
        
        return max_dd, max_dd_start, max_dd_end
    
    def calculate_risk_adjusted_metrics(self, returns: pd.Series, 
                                       benchmark_returns: Optional[pd.Series] = None,
                                       risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate comprehensive risk-adjusted performance metrics
        """
        if len(returns) < 30:
            return {}
        
        annual_factor = 252
        returns_clean = returns.dropna()
        
        # Basic statistics
        annual_return = returns_clean.mean() * annual_factor
        annual_vol = returns_clean.std() * np.sqrt(annual_factor)
        
        # Sharpe Ratio
        sharpe_ratio = (annual_return - risk_free_rate) / annual_vol if annual_vol != 0 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns_clean[returns_clean < 0]
        downside_vol = downside_returns.std() * np.sqrt(annual_factor) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_vol if downside_vol != 0 else 0
        
        # Maximum Drawdown
        max_dd, dd_start, dd_end = self.calculate_max_drawdown(returns_clean)
        
        # Calmar Ratio
        calmar_ratio = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        # VaR and CVaR
        var_95 = self.calculate_var(returns_clean)
        cvar_95 = self.calculate_cvar(returns_clean)
        
        # Skewness and Kurtosis
        skewness = stats.skew(returns_clean)
        kurtosis = stats.kurtosis(returns_clean)
        
        metrics = {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_dd,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'win_rate': len(returns_clean[returns_clean > 0]) / len(returns_clean),
            'profit_factor': returns_clean[returns_clean > 0].sum() / abs(returns_clean[returns_clean < 0].sum()) if len(returns_clean[returns_clean < 0]) > 0 else np.inf
        }
        
        # Benchmark-relative metrics
        if benchmark_returns is not None:
            benchmark_clean = benchmark_returns.dropna()
            aligned_returns = returns_clean.align(benchmark_clean, join='inner')[0]
            aligned_benchmark = returns_clean.align(benchmark_clean, join='inner')[1]
            
            if len(aligned_returns) > 30:
                # Beta and Alpha
                covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                benchmark_variance = np.var(aligned_benchmark)
                beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
                
                benchmark_annual_return = aligned_benchmark.mean() * annual_factor
                alpha = annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
                
                # Tracking Error
                active_returns = aligned_returns - aligned_benchmark
                tracking_error = active_returns.std() * np.sqrt(annual_factor)
                
                # Information Ratio
                information_ratio = active_returns.mean() * annual_factor / tracking_error if tracking_error != 0 else 0
                
                metrics.update({
                    'beta': beta,
                    'alpha': alpha,
                    'tracking_error': tracking_error,
                    'information_ratio': information_ratio
                })
        
        return metrics
    
    def calculate_portfolio_risk(self, returns_df: pd.DataFrame, weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate portfolio-level risk metrics
        """
        if len(returns_df) < 30:
            return {}
        
        # Portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        # Individual asset metrics
        portfolio_metrics = self.calculate_risk_adjusted_metrics(portfolio_returns)
        
        # Correlation analysis
        correlation_matrix = returns_df.corr()
        avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, 1)].mean()
        
        # Concentration risk (Herfindahl Index)
        concentration_risk = np.sum(weights**2)
        
        # Risk contribution analysis
        cov_matrix = returns_df.cov() * 252  # Annualized
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        marginal_risk = np.dot(cov_matrix, weights)
        risk_contributions = weights * marginal_risk / portfolio_var if portfolio_var != 0 else np.zeros_like(weights)
        
        portfolio_metrics.update({
            'avg_correlation': avg_correlation,
            'concentration_risk': concentration_risk,
            'portfolio_variance': portfolio_var,
            'risk_contributions': risk_contributions,
            'effective_num_assets': 1 / concentration_risk if concentration_risk > 0 else len(weights)
        })
        
        return portfolio_metrics
    
    def stress_test_portfolio(self, returns_df: pd.DataFrame, weights: np.ndarray, 
                            scenarios: Optional[Dict[str, Dict]] = None) -> Dict[str, float]:
        """
        Perform stress testing on portfolio
        """
        if scenarios is None:
            scenarios = self._get_default_stress_scenarios()
        
        stress_results = {}
        
        for scenario_name, scenario_params in scenarios.items():
            # Apply scenario shocks
            shocked_returns = returns_df.copy()
            
            for asset, shock in scenario_params.items():
                if asset in shocked_returns.columns:
                    if isinstance(shock, dict):
                        # Volatility and mean shock
                        vol_shock = shock.get('volatility_mult', 1.0)
                        mean_shock = shock.get('mean_add', 0.0)
                        
                        shocked_returns[asset] = (shocked_returns[asset] * vol_shock) + mean_shock
                    else:
                        # Simple additive shock
                        shocked_returns[asset] = shocked_returns[asset] + shock
            
            # Calculate portfolio performance under stress
            stressed_portfolio_returns = (shocked_returns * weights).sum(axis=1)
            stressed_metrics = self.calculate_risk_adjusted_metrics(stressed_portfolio_returns)
            
            stress_results[scenario_name] = {
                'portfolio_return': stressed_metrics.get('annual_return', 0),
                'portfolio_vol': stressed_metrics.get('annual_volatility', 0),
                'max_drawdown': stressed_metrics.get('max_drawdown', 0),
                'var_95': stressed_metrics.get('var_95', 0)
            }
        
        return stress_results
    
    def _get_default_stress_scenarios(self) -> Dict[str, Dict]:
        """
        Define default stress test scenarios
        """
        return {
            'market_crash': {
                'AAPL': -0.20, 'GOOGL': -0.25, 'MSFT': -0.18, 'TSLA': -0.35, 'NVDA': -0.30
            },
            'tech_selloff': {
                'AAPL': -0.15, 'GOOGL': -0.20, 'MSFT': -0.12, 'TSLA': -0.25, 'NVDA': -0.22
            },
            'volatility_spike': {
                'AAPL': {'volatility_mult': 2.0}, 'GOOGL': {'volatility_mult': 2.5},
                'MSFT': {'volatility_mult': 1.8}, 'TSLA': {'volatility_mult': 3.0},
                'NVDA': {'volatility_mult': 2.8}
            },
            'interest_rate_shock': {
                'AAPL': -0.05, 'GOOGL': -0.08, 'MSFT': -0.04, 'TSLA': -0.12, 'NVDA': -0.10
            }
        }
    
    def monte_carlo_risk_simulation(self, returns_df: pd.DataFrame, weights: np.ndarray,
                                   num_simulations: int = 10000, horizon_days: int = 252) -> Dict[str, np.ndarray]:
        """
        Monte Carlo simulation for risk assessment
        """
        # Calculate parameters
        mean_returns = returns_df.mean().values
        cov_matrix = returns_df.cov().values
        
        # Generate random scenarios
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.multivariate_normal(
            mean_returns, cov_matrix, (num_simulations, horizon_days)
        )
        
        # Calculate portfolio paths
        portfolio_paths = np.zeros((num_simulations, horizon_days))
        for i in range(num_simulations):
            daily_returns = np.dot(simulated_returns[i], weights)
            portfolio_paths[i] = np.cumprod(1 + daily_returns)
        
        # Calculate risk metrics from simulations
        final_values = portfolio_paths[:, -1]
        final_returns = final_values - 1
        
        # Risk metrics
        var_95 = np.percentile(final_returns, 5)
        var_99 = np.percentile(final_returns, 1)
        expected_return = np.mean(final_returns)
        
        # Probability of loss
        prob_loss = np.mean(final_returns < 0)
        prob_large_loss = np.mean(final_returns < -0.20)  # 20% loss
        
        # Maximum drawdown simulation
        max_drawdowns = []
        for path in portfolio_paths:
            running_max = np.maximum.accumulate(path)
            drawdown = (path - running_max) / running_max
            max_drawdowns.append(np.min(drawdown))
        
        return {
            'simulated_paths': portfolio_paths,
            'final_returns': final_returns,
            'var_95': var_95,
            'var_99': var_99,
            'expected_return': expected_return,
            'probability_of_loss': prob_loss,
            'probability_large_loss': prob_large_loss,
            'simulated_max_drawdowns': np.array(max_drawdowns),
            'expected_max_drawdown': np.mean(max_drawdowns)
        }
    
    def calculate_risk_parity_weights(self, returns_df: pd.DataFrame) -> np.ndarray:
        """
        Calculate risk parity portfolio weights
        """
        cov_matrix = returns_df.cov().values * 252  # Annualized
        
        def risk_budget_objective(weights, cov_matrix):
            portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
            marginal_risk = np.dot(cov_matrix, weights)
            risk_contributions = weights * marginal_risk
            
            # Equal risk contribution target
            target_risk = portfolio_var / len(weights)
            return np.sum((risk_contributions - target_risk)**2)
        
        # Optimization constraints
        num_assets = len(returns_df.columns)
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0.01, 0.5) for _ in range(num_assets))
        initial_guess = np.array([1/num_assets] * num_assets)
        
        result = minimize(
            risk_budget_objective,
            initial_guess,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else initial_guess
    
    def check_risk_limits(self, returns: pd.Series, current_metrics: Dict[str, float],
                         portfolio_weights: Optional[np.ndarray] = None) -> List[RiskAlert]:
        """
        Check current risk metrics against defined limits
        """
        alerts = []
        current_time = datetime.now()
        
        for metric, limit in self.risk_limits.items():
            current_value = current_metrics.get(metric.value)
            
            if current_value is not None:
                breach = False
                severity = 'low'
                
                if metric in [RiskMetric.VAR, RiskMetric.MAX_DRAWDOWN]:
                    # For negative metrics (losses), check if more negative than limit
                    breach = current_value < limit
                    if breach:
                        severity = 'critical' if current_value < limit * 1.5 else 'high'
                elif metric in [RiskMetric.SHARPE_RATIO]:
                    # For positive metrics, check if below limit
                    breach = current_value < limit
                    severity = 'medium' if current_value < limit * 0.8 else 'low'
                elif metric == RiskMetric.BETA:
                    # For beta, check if above limit
                    breach = current_value > limit
                    severity = 'medium' if current_value > limit * 1.2 else 'low'
                
                if breach:
                    alert = RiskAlert(
                        timestamp=current_time,
                        metric=metric,
                        current_value=current_value,
                        threshold=limit,
                        severity=severity,
                        message=f"{metric.value} breach: {current_value:.4f} vs limit {limit:.4f}"
                    )
                    alerts.append(alert)
        
        # Add concentration risk check
        if portfolio_weights is not None:
            max_weight = np.max(portfolio_weights)
            if max_weight > 0.40:  # 40% concentration limit
                alerts.append(RiskAlert(
                    timestamp=current_time,
                    metric=RiskMetric.VAR,  # Using VAR as proxy for concentration
                    current_value=max_weight,
                    threshold=0.40,
                    severity='high' if max_weight > 0.50 else 'medium',
                    message=f"High concentration risk: {max_weight:.1%} in single position"
                ))
        
        self.risk_alerts.extend(alerts)
        return alerts
    
    def generate_risk_report(self, returns_df: pd.DataFrame, weights: np.ndarray,
                           benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """
        Generate comprehensive risk report
        """
        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        # Calculate all risk metrics
        risk_metrics = self.calculate_risk_adjusted_metrics(portfolio_returns, benchmark_returns)
        portfolio_risk = self.calculate_portfolio_risk(returns_df, weights)
        
        # Stress test results
        stress_results = self.stress_test_portfolio(returns_df, weights)
        
        # Monte Carlo simulation
        mc_results = self.monte_carlo_risk_simulation(returns_df, weights)
        
        # Risk limit checks
        alerts = self.check_risk_limits(portfolio_returns, risk_metrics, weights)
        
        # Risk parity comparison
        risk_parity_weights = self.calculate_risk_parity_weights(returns_df)
        rp_returns = (returns_df * risk_parity_weights).sum(axis=1)
        rp_metrics = self.calculate_risk_adjusted_metrics(rp_returns)
        
        report = {
            'timestamp': datetime.now(),
            'portfolio_metrics': risk_metrics,
            'portfolio_risk_analysis': portfolio_risk,
            'stress_test_results': stress_results,
            'monte_carlo_results': {
                'var_95': mc_results['var_95'],
                'var_99': mc_results['var_99'],
                'expected_return': mc_results['expected_return'],
                'probability_of_loss': mc_results['probability_of_loss'],
                'expected_max_drawdown': mc_results['expected_max_drawdown']
            },
            'risk_alerts': [alert.__dict__ for alert in alerts],
            'current_weights': weights.tolist(),
            'risk_parity_weights': risk_parity_weights.tolist(),
            'risk_parity_metrics': rp_metrics,
            'recommendations': self._generate_recommendations(risk_metrics, alerts, weights)
        }
        
        return report
    
    def _generate_recommendations(self, metrics: Dict, alerts: List[RiskAlert], 
                                weights: np.ndarray) -> List[str]:
        """
        Generate risk management recommendations
        """
        recommendations = []
        
        # High volatility recommendation
        if metrics.get('annual_volatility', 0) > 0.25:
            recommendations.append("Consider reducing position sizes due to high portfolio volatility")
        
        # Low Sharpe ratio recommendation
        if metrics.get('sharpe_ratio', 0) < 0.5:
            recommendations.append("Review strategy performance - Sharpe ratio below acceptable threshold")
        
        # High drawdown recommendation
        if metrics.get('max_drawdown', 0) < -0.15:
            recommendations.append("Implement stronger stop-loss mechanisms to reduce drawdown risk")
        
        # Concentration risk
        if np.max(weights) > 0.30:
            recommendations.append("Reduce concentration risk by diversifying position sizes")
        
        # Critical alerts
        critical_alerts = [a for a in alerts if a.severity == 'critical']
        if critical_alerts:
            recommendations.append("URGENT: Critical risk limits breached - consider reducing positions")
        
        # Beta risk
        if metrics.get('beta', 1.0) > 1.5:
            recommendations.append("High market beta detected - consider hedging strategies")
        
        return recommendations
    
    def plot_risk_analysis(self, returns_df: pd.DataFrame, weights: np.ndarray, 
                          save_path: Optional[str] = None) -> None:
        """
        Create comprehensive risk analysis plots
        """
        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Advanced Risk Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Portfolio returns histogram
        axes[0, 0].hist(portfolio_returns.dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(portfolio_returns.mean(), color='red', linestyle='--', 
                          label=f'Mean: {portfolio_returns.mean():.4f}')
        var_95 = self.calculate_var(portfolio_returns)
        axes[0, 0].axvline(var_95, color='orange', linestyle='--', 
                          label=f'VaR (95%): {var_95:.4f}')
        axes[0, 0].set_title('Portfolio Returns Distribution')
        axes[0, 0].legend()
        
        # 2. Cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        axes[0, 1].plot(cumulative_returns.index, cumulative_returns.values)
        axes[0, 1].set_title('Cumulative Portfolio Returns')
        axes[0, 1].grid(True)
        
        # 3. Drawdown analysis
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        axes[0, 2].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        axes[0, 2].set_title('Portfolio Drawdown')
        axes[0, 2].grid(True)
        
        # 4. Rolling volatility
        rolling_vol = portfolio_returns.rolling(window=30).std() * np.sqrt(252)
        axes[1, 0].plot(rolling_vol.index, rolling_vol.values)
        axes[1, 0].set_title('Rolling Volatility (30-day)')
        axes[1, 0].grid(True)
        
        # 5. Asset weights
        asset_names = returns_df.columns
        axes[1, 1].pie(weights, labels=asset_names, autopct='%1.1f%%')
        axes[1, 1].set_title('Portfolio Allocation')
        
        # 6. Correlation heatmap
        correlation_matrix = returns_df.corr()
        im = axes[1, 2].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 2].set_xticks(range(len(asset_names)))
        axes[1, 2].set_yticks(range(len(asset_names)))
        axes[1, 2].set_xticklabels(asset_names, rotation=45)
        axes[1, 2].set_yticklabels(asset_names)
        axes[1, 2].set_title('Asset Correlation Matrix')
        
        # Add colorbar for correlation
        plt.colorbar(im, ax=axes[1, 2], shrink=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class RealTimeRiskMonitor:
    """
    Real-time risk monitoring system
    """
    
    def __init__(self, risk_model: AdvancedRiskModel):
        self.risk_model = risk_model
        self.monitoring_active = False
        self.current_positions = {}
        self.price_history = {}
        self.risk_callbacks = []
        
    def add_risk_callback(self, callback):
        """Add callback function for risk alerts"""
        self.risk_callbacks.append(callback)
        
    def update_positions(self, positions: Dict[str, Dict]):
        """Update current portfolio positions"""
        self.current_positions = positions
        
    def update_prices(self, symbol: str, price: float, timestamp: datetime):
        """Update price history for risk calculations"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            
        self.price_history[symbol].append({
            'price': price,
            'timestamp': timestamp
        })
        
        # Keep only recent history
        if len(self.price_history[symbol]) > 1000:
            self.price_history[symbol] = self.price_history[symbol][-1000:]
    
    def calculate_real_time_risk(self) -> Dict:
        """Calculate real-time risk metrics"""
        if not self.current_positions or not self.price_history:
            return {}
        
        # Convert price history to returns
        returns_data = {}
        for symbol, history in self.price_history.items():
            if len(history) > 30:  # Need sufficient data
                prices = pd.Series([h['price'] for h in history])
                returns_data[symbol] = prices.pct_change().dropna()
        
        if not returns_data:
            return {}
        
        # Align returns and calculate portfolio metrics
        returns_df = pd.DataFrame(returns_data).dropna()
        
        if returns_df.empty:
            return {}
        
        # Calculate position weights
        total_value = sum(pos.get('market_value', 0) for pos in self.current_positions.values())
        weights = np.array([
            self.current_positions.get(symbol, {}).get('market_value', 0) / total_value 
            if total_value > 0 else 0
            for symbol in returns_df.columns
        ])
        
        # Calculate risk metrics
        portfolio_returns = (returns_df * weights).sum(axis=1)
        risk_metrics = self.risk_model.calculate_risk_adjusted_metrics(portfolio_returns)
        
        # Check for risk alerts
        alerts = self.risk_model.check_risk_limits(portfolio_returns, risk_metrics, weights)
        
        # Trigger callbacks for any alerts
        for alert in alerts:
            for callback in self.risk_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    print(f"Error in risk callback: {e}")
        
        return {
            'timestamp': datetime.now(),
            'risk_metrics': risk_metrics,
            'portfolio_weights': weights.tolist(),
            'alerts': [alert.__dict__ for alert in alerts],
            'total_portfolio_value': total_value
        }
    
    def start_monitoring(self, update_interval: int = 60):
        """Start real-time risk monitoring"""
        import threading
        import time
        
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    risk_data = self.calculate_real_time_risk()
                    if risk_data:
                        # Log or process risk data
                        print(f"Risk Update: {risk_data['timestamp']} - "
                              f"Portfolio Value: ${risk_data['total_portfolio_value']:,.2f}")
                        
                        # Check for high-severity alerts
                        high_severity_alerts = [a for a in risk_data['alerts'] 
                                              if a['severity'] in ['high', 'critical']]
                        if high_severity_alerts:
                            print(f"⚠️  RISK ALERT: {len(high_severity_alerts)} high-severity alerts")
                    
                    time.sleep(update_interval)
                    
                except Exception as e:
                    print(f"Error in risk monitoring: {e}")
                    time.sleep(update_interval)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
        print(f"Real-time risk monitoring started (update interval: {update_interval}s)")
    
    def stop_monitoring(self):
        """Stop real-time risk monitoring"""
        self.monitoring_active = False
        print("Real-time risk monitoring stopped")