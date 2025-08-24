import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union
import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import minimize, differential_evolution
import cvxpy as cp
from sklearn.metrics import mean_squared_error

class MultiObjectiveOptimizer:
    """
    Multi-objective portfolio optimization using various methods
    """
    
    def __init__(self):
        self.pareto_front = None
        self.optimal_portfolios = []
        
    def pareto_optimization(self, returns: pd.DataFrame,
                          objectives: List[str] = ['return', 'risk', 'drawdown'],
                          constraints: Optional[Dict] = None,
                          num_portfolios: int = 100) -> Dict:
        """
        Generate Pareto-optimal portfolios for multiple objectives
        
        Objectives can include:
        - 'return': Expected return (maximize)
        - 'risk': Volatility (minimize) 
        - 'drawdown': Maximum drawdown (minimize)
        - 'var': Value at Risk (minimize)
        - 'cvar': Conditional VaR (minimize)
        - 'tracking_error': Tracking error vs benchmark (minimize)
        - 'turnover': Portfolio turnover (minimize)
        """
        
        n_assets = len(returns.columns)
        
        # Calculate metrics
        mu = returns.mean() * 252
        cov = returns.cov() * 252
        
        pareto_portfolios = []
        
        # Generate random weight combinations
        for i in range(num_portfolios):
            # Random weights (Dirichlet distribution for simplex constraint)
            weights = np.random.dirichlet(np.ones(n_assets))
            weights = pd.Series(weights, index=returns.columns)
            
            # Apply constraints
            if self._satisfies_constraints(weights, constraints):
                # Calculate objective values
                objectives_values = {}
                
                for obj in objectives:
                    if obj == 'return':
                        objectives_values[obj] = (weights * mu).sum()
                    elif obj == 'risk':
                        objectives_values[obj] = np.sqrt(np.dot(weights, np.dot(cov, weights)))
                    elif obj == 'drawdown':
                        objectives_values[obj] = self._calculate_max_drawdown(weights, returns)
                    elif obj == 'var':
                        objectives_values[obj] = self._calculate_var(weights, returns)
                    elif obj == 'cvar':
                        objectives_values[obj] = self._calculate_cvar(weights, returns)
                    elif obj == 'sharpe':
                        ret = (weights * mu).sum()
                        vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
                        objectives_values[obj] = ret / vol if vol > 0 else 0
                
                pareto_portfolios.append({
                    'weights': weights,
                    'objectives': objectives_values
                })
        
        # Filter for Pareto-optimal portfolios
        pareto_optimal = self._find_pareto_optimal(pareto_portfolios, objectives)
        
        self.pareto_front = pareto_optimal
        return {
            'pareto_portfolios': pareto_optimal,
            'num_portfolios': len(pareto_optimal),
            'objectives': objectives
        }
    
    def scalarization_optimization(self, returns: pd.DataFrame,
                                 objectives: Dict[str, float],
                                 method: str = 'weighted_sum') -> Dict:
        """
        Multi-objective optimization using scalarization methods
        
        Parameters:
        - returns: Historical returns
        - objectives: Dictionary of {objective: weight} for combining objectives
        - method: 'weighted_sum', 'goal_programming', or 'compromise_programming'
        """
        
        mu = returns.mean() * 252
        cov = returns.cov() * 252
        n_assets = len(returns.columns)
        
        if method == 'weighted_sum':
            return self._weighted_sum_optimization(returns, objectives, mu, cov)
        elif method == 'goal_programming':
            return self._goal_programming_optimization(returns, objectives, mu, cov)
        elif method == 'compromise_programming':
            return self._compromise_programming_optimization(returns, objectives, mu, cov)
        else:
            raise ValueError("Method must be 'weighted_sum', 'goal_programming', or 'compromise_programming'")
    
    def _weighted_sum_optimization(self, returns: pd.DataFrame, 
                                 objectives: Dict[str, float],
                                 mu: pd.Series, cov: pd.DataFrame) -> Dict:
        """Weighted sum scalarization"""
        
        n_assets = len(returns.columns)
        w = cp.Variable(n_assets)
        
        # Calculate individual objectives
        portfolio_return = mu.values @ w
        portfolio_risk = cp.sqrt(cp.quad_form(w, cov.values))
        
        # Weighted combination
        combined_objective = 0
        
        for obj_name, weight in objectives.items():
            if obj_name == 'return':
                combined_objective += weight * portfolio_return
            elif obj_name == 'risk':
                combined_objective -= weight * portfolio_risk  # Minimize risk
            elif obj_name == 'sharpe':
                # Approximate Sharpe ratio maximization
                combined_objective += weight * (portfolio_return - 0.5 * cp.quad_form(w, cov.values))
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= 0.4
        ]
        
        problem = cp.Problem(cp.Maximize(combined_objective), constraints)
        problem.solve()
        
        if problem.status == 'optimal':
            weights = pd.Series(w.value, index=returns.columns)
            weights = weights / weights.sum()
            
            return self._calculate_portfolio_metrics(weights, returns, mu, cov)
        else:
            return self._fallback_solution(returns, mu, cov)
    
    def _goal_programming_optimization(self, returns: pd.DataFrame,
                                     objectives: Dict[str, float],
                                     mu: pd.Series, cov: pd.DataFrame) -> Dict:
        """Goal programming approach"""
        
        n_assets = len(returns.columns)
        w = cp.Variable(n_assets)
        
        # Deviation variables (positive and negative)
        dev_pos = {}
        dev_neg = {}
        
        # Set goals and calculate deviations
        penalty = 0
        
        for obj_name, goal_value in objectives.items():
            dev_pos[obj_name] = cp.Variable(nonneg=True)
            dev_neg[obj_name] = cp.Variable(nonneg=True)
            
            if obj_name == 'return':
                portfolio_return = mu.values @ w
                penalty += dev_pos[obj_name] + dev_neg[obj_name]
                # Add constraint: actual - goal = dev_pos - dev_neg
                
            elif obj_name == 'risk':
                portfolio_risk = cp.sqrt(cp.quad_form(w, cov.values))
                penalty += dev_pos[obj_name] + dev_neg[obj_name]
        
        # Minimize total penalty
        objective = cp.Minimize(penalty)
        
        # Standard constraints
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= 0.4
        ]
        
        # Add goal constraints
        if 'return' in objectives:
            portfolio_return = mu.values @ w
            constraints.append(
                portfolio_return - objectives['return'] == dev_pos['return'] - dev_neg['return']
            )
        
        if 'risk' in objectives:
            portfolio_risk = cp.sqrt(cp.quad_form(w, cov.values))
            constraints.append(
                portfolio_risk - objectives['risk'] == dev_pos['risk'] - dev_neg['risk']
            )
        
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve()
            
            if problem.status == 'optimal':
                weights = pd.Series(w.value, index=returns.columns)
                weights = weights / weights.sum()
                
                result = self._calculate_portfolio_metrics(weights, returns, mu, cov)
                
                # Add goal achievement info
                result['goal_deviations'] = {}
                for obj_name in objectives.keys():
                    if obj_name in dev_pos:
                        result['goal_deviations'][f'{obj_name}_pos'] = dev_pos[obj_name].value
                        result['goal_deviations'][f'{obj_name}_neg'] = dev_neg[obj_name].value
                
                return result
        
        except Exception as e:
            print(f"Goal programming failed: {e}")
        
        return self._fallback_solution(returns, mu, cov)
    
    def _compromise_programming_optimization(self, returns: pd.DataFrame,
                                          objectives: Dict[str, float],
                                          mu: pd.Series, cov: pd.DataFrame) -> Dict:
        """Compromise programming (minimize distance to ideal point)"""
        
        # First, find ideal and nadir points for each objective
        ideal_point = {}
        nadir_point = {}
        
        n_assets = len(returns.columns)
        
        # Find individual optima
        for obj_name in objectives.keys():
            w = cp.Variable(n_assets)
            constraints = [cp.sum(w) == 1, w >= 0, w <= 0.4]
            
            if obj_name == 'return':
                objective = cp.Maximize(mu.values @ w)
            elif obj_name == 'risk':
                objective = cp.Minimize(cp.sqrt(cp.quad_form(w, cov.values)))
            
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status == 'optimal':
                if obj_name == 'return':
                    ideal_point[obj_name] = (mu.values @ w).value
                elif obj_name == 'risk':
                    ideal_point[obj_name] = cp.sqrt(cp.quad_form(w, cov.values)).value
        
        # Now optimize compromise solution
        w = cp.Variable(n_assets)
        
        # Calculate normalized distances
        distance_terms = []
        
        for obj_name, weight in objectives.items():
            if obj_name == 'return':
                portfolio_return = mu.values @ w
                if obj_name in ideal_point:
                    normalized_distance = cp.abs(portfolio_return - ideal_point[obj_name])
                    distance_terms.append(weight * normalized_distance)
            
            elif obj_name == 'risk':
                portfolio_risk = cp.sqrt(cp.quad_form(w, cov.values))
                if obj_name in ideal_point:
                    normalized_distance = cp.abs(portfolio_risk - ideal_point[obj_name])
                    distance_terms.append(weight * normalized_distance)
        
        # Minimize weighted distance to ideal point
        if distance_terms:
            objective = cp.Minimize(cp.sum(distance_terms))
        else:
            # Fallback to return maximization
            objective = cp.Maximize(mu.values @ w)
        
        constraints = [cp.sum(w) == 1, w >= 0, w <= 0.4]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status == 'optimal':
            weights = pd.Series(w.value, index=returns.columns)
            weights = weights / weights.sum()
            
            result = self._calculate_portfolio_metrics(weights, returns, mu, cov)
            result['ideal_point'] = ideal_point
            
            return result
        
        return self._fallback_solution(returns, mu, cov)
    
    def robust_optimization(self, returns: pd.DataFrame,
                          uncertainty_sets: Dict[str, Dict],
                          confidence_level: float = 0.95) -> Dict:
        """
        Robust portfolio optimization under parameter uncertainty
        
        Parameters:
        - returns: Historical returns
        - uncertainty_sets: Dictionary defining uncertainty for parameters
        - confidence_level: Confidence level for robust solution
        """
        
        mu = returns.mean() * 252
        cov = returns.cov() * 252
        n_assets = len(returns.columns)
        
        # Uncertainty in expected returns
        mu_uncertainty = uncertainty_sets.get('returns', 0.1)  # 10% uncertainty by default
        
        # Uncertainty in covariance matrix
        cov_uncertainty = uncertainty_sets.get('covariance', 0.05)  # 5% uncertainty by default
        
        w = cp.Variable(n_assets)
        
        # Worst-case expected return
        # Assume returns can be mu +/- mu_uncertainty * mu
        worst_case_return = mu.values @ w - mu_uncertainty * cp.norm(w, 1) * cp.max(mu.values)
        
        # Robust risk measure (add uncertainty to covariance)
        robust_cov = cov.values * (1 + cov_uncertainty)
        portfolio_risk = cp.sqrt(cp.quad_form(w, robust_cov))
        
        # Robust Sharpe ratio maximization
        objective = cp.Maximize(worst_case_return - 0.5 * cp.quad_form(w, robust_cov))
        
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= 0.4
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status == 'optimal':
            weights = pd.Series(w.value, index=returns.columns)
            weights = weights / weights.sum()
            
            result = self._calculate_portfolio_metrics(weights, returns, mu, cov)
            result['robust_return'] = worst_case_return.value
            result['uncertainty_adjusted'] = True
            
            return result
        
        return self._fallback_solution(returns, mu, cov)
    
    def _satisfies_constraints(self, weights: pd.Series, 
                              constraints: Optional[Dict]) -> bool:
        """Check if portfolio satisfies constraints"""
        
        if constraints is None:
            return True
        
        # Long-only constraint
        if constraints.get('long_only', False) and (weights < 0).any():
            return False
        
        # Maximum weight constraint
        max_weight = constraints.get('max_weight')
        if max_weight and (weights > max_weight).any():
            return False
        
        # Minimum weight constraint
        min_weight = constraints.get('min_weight')
        if min_weight and (weights < min_weight).any():
            return False
        
        return True
    
    def _calculate_max_drawdown(self, weights: pd.Series, returns: pd.DataFrame) -> float:
        """Calculate maximum drawdown for given weights"""
        
        portfolio_returns = (returns * weights).sum(axis=1)
        cumulative = (1 + portfolio_returns).cumprod()
        drawdowns = 1 - cumulative / cumulative.expanding().max()
        
        return drawdowns.max()
    
    def _calculate_var(self, weights: pd.Series, returns: pd.DataFrame,
                      confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        
        portfolio_returns = (returns * weights).sum(axis=1)
        return -np.percentile(portfolio_returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, weights: pd.Series, returns: pd.DataFrame,
                       confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk"""
        
        portfolio_returns = (returns * weights).sum(axis=1)
        var = self._calculate_var(weights, returns, confidence)
        
        # CVaR is the mean of returns below VaR
        tail_returns = portfolio_returns[portfolio_returns <= -var]
        
        return -tail_returns.mean() if len(tail_returns) > 0 else var
    
    def _find_pareto_optimal(self, portfolios: List[Dict], 
                           objectives: List[str]) -> List[Dict]:
        """Find Pareto-optimal portfolios"""
        
        pareto_optimal = []
        
        for i, portfolio_i in enumerate(portfolios):
            is_dominated = False
            
            for j, portfolio_j in enumerate(portfolios):
                if i != j:
                    # Check if portfolio_j dominates portfolio_i
                    dominates = True
                    
                    for obj in objectives:
                        # Determine if we want to maximize or minimize
                        if obj in ['return', 'sharpe']:
                            # Maximize: j dominates i if j >= i for all objectives and j > i for at least one
                            if portfolio_j['objectives'][obj] < portfolio_i['objectives'][obj]:
                                dominates = False
                                break
                        else:
                            # Minimize: j dominates i if j <= i for all objectives and j < i for at least one
                            if portfolio_j['objectives'][obj] > portfolio_i['objectives'][obj]:
                                dominates = False
                                break
                    
                    if dominates:
                        # Check if at least one objective is strictly better
                        strictly_better = False
                        for obj in objectives:
                            if obj in ['return', 'sharpe']:
                                if portfolio_j['objectives'][obj] > portfolio_i['objectives'][obj]:
                                    strictly_better = True
                                    break
                            else:
                                if portfolio_j['objectives'][obj] < portfolio_i['objectives'][obj]:
                                    strictly_better = True
                                    break
                        
                        if strictly_better:
                            is_dominated = True
                            break
            
            if not is_dominated:
                pareto_optimal.append(portfolio_i)
        
        return pareto_optimal
    
    def _calculate_portfolio_metrics(self, weights: pd.Series, returns: pd.DataFrame,
                                   mu: pd.Series, cov: pd.DataFrame) -> Dict:
        """Calculate standard portfolio metrics"""
        
        portfolio_return = (weights * mu).sum()
        portfolio_variance = np.dot(weights, np.dot(cov, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'status': 'optimal'
        }
    
    def _fallback_solution(self, returns: pd.DataFrame, mu: pd.Series, cov: pd.DataFrame) -> Dict:
        """Fallback to equal weights"""
        
        n_assets = len(returns.columns)
        equal_weights = pd.Series(1/n_assets, index=returns.columns)
        
        return self._calculate_portfolio_metrics(equal_weights, returns, mu, cov)

class DynamicPortfolioOptimizer:
    """
    Dynamic portfolio optimization with rebalancing
    """
    
    def __init__(self, rebalance_frequency: str = 'quarterly',
                 lookback_window: int = 252,
                 min_history: int = 60):
        
        self.rebalance_frequency = rebalance_frequency
        self.lookback_window = lookback_window
        self.min_history = min_history
        self.optimization_history = []
        
    def dynamic_optimization(self, returns: pd.DataFrame,
                           optimizer_type: str = 'mean_variance',
                           **optimizer_kwargs) -> Dict:
        """
        Run dynamic portfolio optimization with periodic rebalancing
        
        Parameters:
        - returns: Historical returns
        - optimizer_type: Type of optimizer to use
        - optimizer_kwargs: Additional arguments for optimizer
        """
        
        # Determine rebalancing dates
        rebalance_dates = self._get_rebalance_dates(returns.index)
        
        portfolio_weights = []
        portfolio_returns = []
        optimization_results = []
        
        for i, rebal_date in enumerate(rebalance_dates):
            if rebal_date not in returns.index:
                continue
            
            # Get historical data for optimization
            end_idx = returns.index.get_loc(rebal_date)
            start_idx = max(0, end_idx - self.lookback_window)
            
            if end_idx - start_idx < self.min_history:
                continue
            
            hist_returns = returns.iloc[start_idx:end_idx]
            
            # Run optimization
            if optimizer_type == 'mean_variance':
                from .portfolio_optimizer import MeanVarianceOptimizer
                optimizer = MeanVarianceOptimizer()
                result = optimizer.optimize_portfolio(hist_returns, **optimizer_kwargs)
                
            elif optimizer_type == 'risk_parity':
                from .portfolio_optimizer import RiskParityOptimizer
                optimizer = RiskParityOptimizer()
                result = optimizer.optimize_risk_parity(hist_returns, **optimizer_kwargs)
                
            elif optimizer_type == 'black_litterman':
                from .portfolio_optimizer import BlackLittermanOptimizer
                optimizer = BlackLittermanOptimizer()
                # Need market weights for BL
                market_weights = pd.Series(1/len(returns.columns), index=returns.columns)
                result = optimizer.optimize_with_views(
                    hist_returns, market_weights, 
                    optimizer_kwargs.get('views', {}),
                    optimizer_kwargs.get('risk_aversion', 3)
                )
            
            else:
                raise ValueError(f"Unknown optimizer type: {optimizer_type}")
            
            if result['status'] in ['optimal', 'optimal_inaccurate']:
                weights = result['weights']
                portfolio_weights.append({
                    'date': rebal_date,
                    'weights': weights
                })
                
                optimization_results.append({
                    'date': rebal_date,
                    'result': result
                })
            
        # Calculate portfolio performance
        if portfolio_weights:
            performance = self._calculate_dynamic_performance(returns, portfolio_weights)
            
            return {
                'portfolio_weights': portfolio_weights,
                'performance': performance,
                'optimization_results': optimization_results,
                'rebalance_dates': rebalance_dates
            }
        
        return {'error': 'No successful optimizations'}
    
    def _get_rebalance_dates(self, date_index: pd.DatetimeIndex) -> List:
        """Generate rebalancing dates based on frequency"""
        
        if self.rebalance_frequency == 'monthly':
            freq = 'M'
        elif self.rebalance_frequency == 'quarterly':
            freq = 'Q'
        elif self.rebalance_frequency == 'yearly':
            freq = 'Y'
        else:
            raise ValueError("Frequency must be 'monthly', 'quarterly', or 'yearly'")
        
        # Generate rebalancing schedule
        start_date = date_index[self.min_history] if len(date_index) > self.min_history else date_index[0]
        end_date = date_index[-1]
        
        rebal_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Find closest available dates in the index
        available_rebal_dates = []
        for date in rebal_dates:
            closest_date = date_index[date_index <= date]
            if len(closest_date) > 0:
                available_rebal_dates.append(closest_date[-1])
        
        return available_rebal_dates
    
    def _calculate_dynamic_performance(self, returns: pd.DataFrame, 
                                     portfolio_weights: List[Dict]) -> Dict:
        """Calculate performance of dynamic portfolio"""
        
        if not portfolio_weights:
            return {}
        
        # Create weight matrix
        weight_df = pd.DataFrame()
        for pw in portfolio_weights:
            weight_df = pd.concat([weight_df, pd.DataFrame([pw['weights']], index=[pw['date']])])
        
        # Forward-fill weights between rebalancing dates
        full_dates = returns.index
        weight_df = weight_df.reindex(full_dates, method='ffill')
        
        # Calculate portfolio returns
        portfolio_rets = (returns * weight_df).sum(axis=1, skipna=True)
        
        # Performance metrics
        total_return = (1 + portfolio_rets).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_rets)) - 1
        volatility = portfolio_rets.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + portfolio_rets).cumprod()
        drawdowns = 1 - cumulative / cumulative.expanding().max()
        max_drawdown = drawdowns.max()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_returns': portfolio_rets
        }