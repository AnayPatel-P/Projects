import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import minimize
from scipy import linalg
import cvxpy as cp

class MeanVarianceOptimizer:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.weights = None
        self.expected_return = None
        self.portfolio_variance = None
        
    def calculate_expected_returns(self, returns: pd.DataFrame, 
                                 method: str = 'historical') -> pd.Series:
        """Calculate expected returns using various methods"""
        
        if method == 'historical':
            return returns.mean() * 252  # Annualized
        
        elif method == 'exponential':
            # Exponentially weighted moving average
            return returns.ewm(halflife=63).mean().iloc[-1] * 252
        
        elif method == 'shrinkage':
            # James-Stein shrinkage estimator
            sample_mean = returns.mean() * 252
            grand_mean = sample_mean.mean()
            
            # Calculate shrinkage intensity
            n_assets = len(sample_mean)
            numerator = ((sample_mean - grand_mean) ** 2).sum()
            denominator = n_assets * returns.var().mean() * 252
            
            if denominator > 0:
                shrinkage = min(1, numerator / denominator)
            else:
                shrinkage = 0
            
            return shrinkage * grand_mean + (1 - shrinkage) * sample_mean
        
        else:
            raise ValueError("Method must be 'historical', 'exponential', or 'shrinkage'")
    
    def calculate_covariance_matrix(self, returns: pd.DataFrame,
                                  method: str = 'sample') -> pd.DataFrame:
        """Calculate covariance matrix using various methods"""
        
        if method == 'sample':
            return returns.cov() * 252  # Annualized
        
        elif method == 'exponential':
            return returns.ewm(halflife=63).cov() * 252
        
        elif method == 'ledoit_wolf':
            # Ledoit-Wolf shrinkage estimator
            from sklearn.covariance import LedoitWolf
            
            lw = LedoitWolf()
            cov_lw = lw.fit(returns.fillna(0)).covariance_
            return pd.DataFrame(cov_lw, index=returns.columns, 
                              columns=returns.columns) * 252
        
        elif method == 'robust':
            # Robust covariance estimation
            from sklearn.covariance import EmpiricalCovariance
            
            robust_cov = EmpiricalCovariance()
            cov_robust = robust_cov.fit(returns.fillna(0)).covariance_
            return pd.DataFrame(cov_robust, index=returns.columns, 
                              columns=returns.columns) * 252
        
        else:
            raise ValueError("Method must be 'sample', 'exponential', 'ledoit_wolf', or 'robust'")
    
    def optimize_portfolio(self, returns: pd.DataFrame, 
                          target_return: Optional[float] = None,
                          risk_tolerance: Optional[float] = None,
                          constraints: Optional[Dict] = None) -> Dict:
        """
        Optimize portfolio using mean-variance optimization
        
        Parameters:
        - returns: Historical returns DataFrame
        - target_return: Target portfolio return (if None, maximize Sharpe ratio)
        - risk_tolerance: Risk aversion parameter (higher = more risk averse)
        - constraints: Additional constraints dictionary
        """
        
        # Calculate expected returns and covariance
        mu = self.calculate_expected_returns(returns, method='shrinkage')
        cov = self.calculate_covariance_matrix(returns, method='ledoit_wolf')
        
        n_assets = len(mu)
        
        # Set up optimization variables
        w = cp.Variable(n_assets)
        
        # Objective function
        portfolio_return = mu.values @ w
        portfolio_variance = cp.quad_form(w, cov.values)
        
        # Constraints
        constraints_list = [cp.sum(w) == 1]  # Weights sum to 1
        
        # Default constraints
        if constraints is None:
            constraints = {
                'long_only': True,
                'max_weight': 0.4,
                'min_weight': 0.01
            }
        
        # Long-only constraint
        if constraints.get('long_only', True):
            constraints_list.append(w >= 0)
        
        # Weight bounds
        if 'max_weight' in constraints:
            constraints_list.append(w <= constraints['max_weight'])
        
        if 'min_weight' in constraints:
            constraints_list.append(w >= constraints['min_weight'])
        
        # Sector constraints
        if 'sector_constraints' in constraints:
            for sector, (assets, max_weight) in constraints['sector_constraints'].items():
                sector_indices = [i for i, asset in enumerate(returns.columns) if asset in assets]
                if sector_indices:
                    constraints_list.append(cp.sum([w[i] for i in sector_indices]) <= max_weight)
        
        # Target return constraint
        if target_return is not None:
            constraints_list.append(portfolio_return >= target_return)
        
        # Solve optimization problem
        if target_return is not None:
            # Minimize risk for target return
            objective = cp.Minimize(portfolio_variance)
        elif risk_tolerance is not None:
            # Utility maximization: return - (risk_tolerance/2) * variance
            objective = cp.Maximize(portfolio_return - (risk_tolerance / 2) * portfolio_variance)
        else:
            # Maximize Sharpe ratio (approximate)
            objective = cp.Maximize(portfolio_return - self.risk_free_rate - 0.5 * portfolio_variance)
        
        problem = cp.Problem(objective, constraints_list)
        
        try:
            problem.solve()
            
            if problem.status not in ['optimal', 'optimal_inaccurate']:
                raise ValueError(f"Optimization failed with status: {problem.status}")
            
            weights = pd.Series(w.value, index=returns.columns)
            weights = weights.clip(lower=0)  # Remove tiny negative values
            weights = weights / weights.sum()  # Renormalize
            
            # Calculate portfolio metrics
            portfolio_return_val = (weights * mu).sum()
            portfolio_variance_val = np.dot(weights, np.dot(cov, weights))
            portfolio_volatility = np.sqrt(portfolio_variance_val)
            sharpe_ratio = (portfolio_return_val - self.risk_free_rate) / portfolio_volatility
            
            self.weights = weights
            self.expected_return = portfolio_return_val
            self.portfolio_variance = portfolio_variance_val
            
            return {
                'weights': weights,
                'expected_return': portfolio_return_val,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'status': problem.status
            }
            
        except Exception as e:
            print(f"Optimization error: {e}")
            # Fallback to equal weights
            equal_weights = pd.Series(1/n_assets, index=returns.columns)
            return {
                'weights': equal_weights,
                'expected_return': (equal_weights * mu).sum(),
                'volatility': np.sqrt(np.dot(equal_weights, np.dot(cov, equal_weights))),
                'sharpe_ratio': 0,
                'status': 'fallback'
            }
    
    def efficient_frontier(self, returns: pd.DataFrame, 
                          num_points: int = 50) -> pd.DataFrame:
        """Generate efficient frontier points"""
        
        mu = self.calculate_expected_returns(returns)
        cov = self.calculate_covariance_matrix(returns)
        
        # Range of target returns
        min_return = mu.min()
        max_return = mu.max()
        target_returns = np.linspace(min_return, max_return, num_points)
        
        frontier_results = []
        
        for target in target_returns:
            try:
                result = self.optimize_portfolio(returns, target_return=target)
                
                frontier_results.append({
                    'target_return': target,
                    'expected_return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe_ratio': result['sharpe_ratio']
                })
                
            except:
                continue
        
        return pd.DataFrame(frontier_results)

class BlackLittermanOptimizer:
    def __init__(self, risk_free_rate: float = 0.02, tau: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self.tau = tau  # Uncertainty parameter
        
    def calculate_implied_returns(self, cov: pd.DataFrame, 
                                market_weights: pd.Series,
                                risk_aversion: float = 3) -> pd.Series:
        """Calculate implied equilibrium returns"""
        return risk_aversion * np.dot(cov, market_weights)
    
    def optimize_with_views(self, returns: pd.DataFrame,
                           market_weights: pd.Series,
                           views: Dict[str, Dict],
                           risk_aversion: float = 3) -> Dict:
        """
        Black-Litterman optimization with investor views
        
        Parameters:
        - returns: Historical returns
        - market_weights: Market capitalization weights
        - views: Dictionary of views {view_name: {'assets': [assets], 'return': expected_return, 'confidence': confidence}}
        - risk_aversion: Risk aversion parameter
        """
        
        # Calculate covariance matrix
        cov = returns.cov() * 252
        
        # Step 1: Calculate implied equilibrium returns
        pi = self.calculate_implied_returns(cov, market_weights, risk_aversion)
        
        # Step 2: Set up views
        if not views:
            # No views, return market weights
            return {
                'weights': market_weights,
                'expected_return': (market_weights * pi).sum(),
                'volatility': np.sqrt(np.dot(market_weights, np.dot(cov, market_weights))),
                'sharpe_ratio': ((market_weights * pi).sum() - self.risk_free_rate) / 
                               np.sqrt(np.dot(market_weights, np.dot(cov, market_weights)))
            }
        
        # Create P matrix (picking matrix)
        P = []
        Q = []  # Expected returns from views
        Omega = []  # Uncertainty matrix
        
        assets = list(returns.columns)
        
        for view_name, view_data in views.items():
            # Create picking vector for this view
            p_vector = np.zeros(len(assets))
            
            for asset in view_data['assets']:
                if asset in assets:
                    idx = assets.index(asset)
                    p_vector[idx] = 1 / len(view_data['assets'])  # Equal weight in view
            
            P.append(p_vector)
            Q.append(view_data['return'])
            
            # Uncertainty (lower confidence = higher uncertainty)
            confidence = view_data.get('confidence', 0.5)
            uncertainty = (1 - confidence) * np.dot(p_vector, np.dot(cov, p_vector))
            Omega.append(uncertainty)
        
        P = np.array(P)
        Q = np.array(Q)
        Omega = np.diag(Omega)
        
        # Step 3: Black-Litterman formula
        tau_cov = self.tau * cov
        
        try:
            # Calculate new expected returns
            term1 = linalg.inv(tau_cov)
            term2 = np.dot(P.T, np.dot(linalg.inv(Omega), P))
            term3 = np.dot(linalg.inv(tau_cov), pi) + np.dot(P.T, np.dot(linalg.inv(Omega), Q))
            
            mu_bl = np.dot(linalg.inv(term1 + term2), term3)
            
            # Calculate new covariance matrix
            cov_bl = linalg.inv(term1 + term2)
            
            # Step 4: Optimize portfolio with Black-Litterman inputs
            n_assets = len(mu_bl)
            
            # Use mean-variance optimization with BL estimates
            w = cp.Variable(n_assets)
            
            objective = cp.Maximize(mu_bl @ w - (risk_aversion / 2) * cp.quad_form(w, cov_bl))
            constraints = [
                cp.sum(w) == 1,
                w >= 0,  # Long-only
                w <= 0.4  # Max weight
            ]
            
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status == 'optimal':
                weights = pd.Series(w.value, index=assets)
                weights = weights / weights.sum()  # Renormalize
                
                portfolio_return = (weights * mu_bl).sum()
                portfolio_variance = np.dot(weights, np.dot(cov_bl, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
                
                return {
                    'weights': weights,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'bl_returns': pd.Series(mu_bl, index=assets),
                    'status': 'optimal'
                }
            
        except Exception as e:
            print(f"Black-Litterman optimization error: {e}")
        
        # Fallback to market weights
        return {
            'weights': market_weights,
            'expected_return': (market_weights * pi).sum(),
            'volatility': np.sqrt(np.dot(market_weights, np.dot(cov, market_weights))),
            'sharpe_ratio': 0,
            'status': 'fallback'
        }

class RiskParityOptimizer:
    def __init__(self):
        self.weights = None
        
    def calculate_risk_contribution(self, weights: np.ndarray, 
                                  cov_matrix: np.ndarray) -> np.ndarray:
        """Calculate risk contribution of each asset"""
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
        risk_contrib = weights * marginal_contrib / portfolio_vol
        return risk_contrib
    
    def optimize_risk_parity(self, returns: pd.DataFrame,
                           target_risk_contrib: Optional[np.ndarray] = None) -> Dict:
        """
        Optimize portfolio for risk parity
        
        Parameters:
        - returns: Historical returns DataFrame
        - target_risk_contrib: Target risk contributions (if None, equal risk contribution)
        """
        
        cov = returns.cov().values * 252
        n_assets = len(returns.columns)
        
        # Target risk contributions (equal by default)
        if target_risk_contrib is None:
            target_risk_contrib = np.ones(n_assets) / n_assets
        
        # Objective function: minimize sum of squared differences from target risk contributions
        def objective(weights):
            weights = np.array(weights)
            risk_contrib = self.calculate_risk_contribution(weights, cov)
            return np.sum((risk_contrib - target_risk_contrib) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        ]
        
        # Bounds (long-only, reasonable weight bounds)
        bounds = [(0.01, 0.5) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            weights = pd.Series(result.x, index=returns.columns)
            weights = weights / weights.sum()  # Renormalize
            
            # Calculate portfolio metrics
            portfolio_variance = np.dot(weights, np.dot(cov, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            expected_returns = returns.mean() * 252
            portfolio_return = (weights * expected_returns).sum()
            
            risk_contributions = self.calculate_risk_contribution(weights.values, cov)
            
            self.weights = weights
            
            return {
                'weights': weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'risk_contributions': pd.Series(risk_contributions, index=returns.columns),
                'diversification_ratio': self._calculate_diversification_ratio(weights, cov),
                'status': 'optimal'
            }
        
        else:
            # Fallback to equal weights
            equal_weights = pd.Series(1/n_assets, index=returns.columns)
            expected_returns = returns.mean() * 252
            
            return {
                'weights': equal_weights,
                'expected_return': (equal_weights * expected_returns).sum(),
                'volatility': np.sqrt(np.dot(equal_weights, np.dot(cov, equal_weights))),
                'risk_contributions': pd.Series([1/n_assets] * n_assets, index=returns.columns),
                'diversification_ratio': 1.0,
                'status': 'fallback'
            }
    
    def _calculate_diversification_ratio(self, weights: pd.Series, cov: np.ndarray) -> float:
        """Calculate diversification ratio"""
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
        asset_vols = np.sqrt(np.diag(cov))
        weighted_avg_vol = np.dot(weights, asset_vols)
        
        return weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0

class FactorOptimizer:
    def __init__(self, factors: List[str] = None):
        self.factors = factors or ['market', 'size', 'value', 'momentum', 'quality']
        self.factor_loadings = None
        self.factor_returns = None
        
    def estimate_factor_model(self, returns: pd.DataFrame, 
                            factor_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Estimate factor model using PCA or predefined factors
        
        Parameters:
        - returns: Asset returns
        - factor_data: External factor data (if None, use PCA)
        """
        
        if factor_data is not None:
            # Use provided factor data
            common_dates = returns.index.intersection(factor_data.index)
            if len(common_dates) < 50:
                raise ValueError("Insufficient overlapping data for factor analysis")
            
            Y = returns.loc[common_dates]
            X = factor_data.loc[common_dates]
            
            # Run regression for each asset
            from sklearn.linear_model import LinearRegression
            
            factor_loadings = pd.DataFrame(index=Y.columns, columns=X.columns)
            residual_vars = pd.Series(index=Y.columns)
            
            for asset in Y.columns:
                reg = LinearRegression().fit(X, Y[asset])
                factor_loadings.loc[asset] = reg.coef_
                
                # Calculate residual variance
                predictions = reg.predict(X)
                residuals = Y[asset] - predictions
                residual_vars[asset] = residuals.var()
            
            self.factor_loadings = factor_loadings
            self.factor_returns = X.mean() * 252
            
        else:
            # Use PCA to extract factors
            from sklearn.decomposition import PCA
            
            # Fill missing values
            returns_filled = returns.fillna(returns.mean())
            
            # Determine number of factors (explained variance > 80%)
            pca_full = PCA()
            pca_full.fit(returns_filled)
            
            cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
            n_factors = np.argmax(cumsum_var >= 0.8) + 1
            n_factors = min(n_factors, len(self.factors))
            
            # Extract factors
            pca = PCA(n_components=n_factors)
            factor_scores = pca.fit_transform(returns_filled)
            
            # Create factor loadings
            factor_loadings = pd.DataFrame(
                pca.components_.T,
                index=returns.columns,
                columns=[f'Factor_{i+1}' for i in range(n_factors)]
            )
            
            # Factor returns (mean of factor scores)
            factor_returns = pd.DataFrame(factor_scores, index=returns.index).mean() * 252
            
            self.factor_loadings = factor_loadings
            self.factor_returns = factor_returns
            
            # Calculate residual variances
            reconstructed = np.dot(factor_scores, pca.components_)
            residuals = returns_filled.values - reconstructed
            residual_vars = pd.Series(np.var(residuals, axis=0), index=returns.columns)
        
        return {
            'factor_loadings': self.factor_loadings,
            'factor_returns': self.factor_returns,
            'residual_variances': residual_vars,
            'explained_variance': cumsum_var[-1] if 'cumsum_var' in locals() else None
        }
    
    def optimize_factor_portfolio(self, returns: pd.DataFrame,
                                factor_views: Optional[Dict] = None,
                                target_exposures: Optional[Dict] = None) -> Dict:
        """
        Optimize portfolio using factor model
        
        Parameters:
        - returns: Asset returns
        - factor_views: Views on factor returns {'factor': expected_return}
        - target_exposures: Target factor exposures {'factor': target_exposure}
        """
        
        # Estimate factor model if not already done
        if self.factor_loadings is None:
            self.estimate_factor_model(returns)
        
        n_assets = len(returns.columns)
        
        # Set up optimization
        w = cp.Variable(n_assets)
        
        # Expected returns based on factor model
        if factor_views:
            # Update factor returns with views
            factor_returns = self.factor_returns.copy()
            for factor, view_return in factor_views.items():
                if factor in factor_returns.index:
                    factor_returns[factor] = view_return
        else:
            factor_returns = self.factor_returns
        
        # Calculate expected asset returns from factor model
        expected_returns = np.dot(self.factor_loadings, factor_returns)
        
        portfolio_return = expected_returns @ w
        
        # Risk model: factor covariance + residual risk
        factor_cov = np.cov(self.factor_loadings.T) * 252
        portfolio_factor_risk = cp.quad_form(self.factor_loadings.T @ w, factor_cov)
        
        # Add residual risk (diagonal)
        residual_risk = cp.sum(cp.multiply(cp.square(w), returns.var() * 252))
        total_risk = portfolio_factor_risk + residual_risk
        
        # Constraints
        constraints = [cp.sum(w) == 1, w >= 0]
        
        # Factor exposure constraints
        if target_exposures:
            for factor, target in target_exposures.items():
                if factor in self.factor_loadings.columns:
                    factor_exposure = self.factor_loadings[factor] @ w
                    constraints.append(cp.abs(factor_exposure - target) <= 0.1)
        
        # Optimization objective (maximize Sharpe ratio approximation)
        objective = cp.Maximize(portfolio_return - 0.5 * total_risk)
        
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve()
            
            if problem.status == 'optimal':
                weights = pd.Series(w.value, index=returns.columns)
                weights = weights / weights.sum()
                
                # Calculate portfolio metrics
                portfolio_return_val = (weights * expected_returns).sum()
                portfolio_risk_val = np.sqrt(np.dot(weights, np.dot(returns.cov() * 252, weights)))
                
                # Factor exposures
                factor_exposures = np.dot(self.factor_loadings.T, weights)
                
                return {
                    'weights': weights,
                    'expected_return': portfolio_return_val,
                    'volatility': portfolio_risk_val,
                    'sharpe_ratio': portfolio_return_val / portfolio_risk_val if portfolio_risk_val > 0 else 0,
                    'factor_exposures': pd.Series(factor_exposures, index=self.factor_loadings.columns),
                    'status': 'optimal'
                }
            
        except Exception as e:
            print(f"Factor optimization error: {e}")
        
        # Fallback
        equal_weights = pd.Series(1/n_assets, index=returns.columns)
        return {
            'weights': equal_weights,
            'expected_return': (equal_weights * returns.mean() * 252).sum(),
            'volatility': np.sqrt(np.dot(equal_weights, np.dot(returns.cov() * 252, equal_weights))),
            'sharpe_ratio': 0,
            'status': 'fallback'
        }