import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize_scalar, brentq
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

class ExerciseStyle(Enum):
    EUROPEAN = "european"
    AMERICAN = "american"

@dataclass
class Option:
    symbol: str
    option_type: OptionType
    strike_price: float
    expiration_date: datetime
    premium: float
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN
    underlying_symbol: str = ""
    
    def days_to_expiration(self) -> int:
        return (self.expiration_date - datetime.now()).days
    
    def time_to_expiration(self) -> float:
        return self.days_to_expiration() / 365.25

@dataclass
class OptionPosition:
    option: Option
    quantity: int  # Positive for long, negative for short
    entry_price: float
    entry_date: datetime
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0

class BlackScholesModel:
    """
    Black-Scholes option pricing model with Greeks
    """
    
    @staticmethod
    def calculate_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """Calculate d1 and d2 parameters"""
        if T <= 0:
            return 0, 0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    @classmethod
    def option_price(cls, S: float, K: float, T: float, r: float, sigma: float, 
                    option_type: OptionType) -> float:
        """
        Calculate Black-Scholes option price
        
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility
        """
        if T <= 0:
            if option_type == OptionType.CALL:
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1, d2 = cls.calculate_d1_d2(S, K, T, r, sigma)
        
        if option_type == OptionType.CALL:
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return max(price, 0)
    
    @classmethod
    def calculate_greeks(cls, S: float, K: float, T: float, r: float, sigma: float,
                        option_type: OptionType) -> Dict[str, float]:
        """Calculate option Greeks"""
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        d1, d2 = cls.calculate_d1_d2(S, K, T, r, sigma)
        
        # Delta
        if option_type == OptionType.CALL:
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        # Gamma
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        if option_type == OptionType.CALL:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                    r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                    r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        # Vega
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Rho
        if option_type == OptionType.CALL:
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    @classmethod
    def implied_volatility(cls, market_price: float, S: float, K: float, T: float, r: float,
                          option_type: OptionType, max_iterations: int = 100) -> float:
        """Calculate implied volatility using Newton-Raphson method"""
        if T <= 0:
            return 0
        
        def objective(sigma):
            return cls.option_price(S, K, T, r, sigma, option_type) - market_price
        
        try:
            # Use Brent's method for root finding
            iv = brentq(objective, 0.001, 5.0, maxiter=max_iterations)
            return iv
        except:
            # Fallback to approximation
            return np.sqrt(2 * np.pi / T) * market_price / S

class BinomialTreeModel:
    """
    Binomial tree model for American and European option pricing
    """
    
    @staticmethod
    def option_price(S: float, K: float, T: float, r: float, sigma: float,
                    option_type: OptionType, exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN,
                    num_steps: int = 100) -> float:
        """
        Calculate option price using binomial tree
        """
        dt = T / num_steps
        u = np.exp(sigma * np.sqrt(dt))  # Up movement
        d = 1 / u  # Down movement
        p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability
        
        # Initialize asset prices at expiration
        asset_prices = np.zeros(num_steps + 1)
        for i in range(num_steps + 1):
            asset_prices[i] = S * (u ** (num_steps - i)) * (d ** i)
        
        # Initialize option values at expiration
        option_values = np.zeros(num_steps + 1)
        for i in range(num_steps + 1):
            if option_type == OptionType.CALL:
                option_values[i] = max(asset_prices[i] - K, 0)
            else:
                option_values[i] = max(K - asset_prices[i], 0)
        
        # Backward induction
        for j in range(num_steps - 1, -1, -1):
            for i in range(j + 1):
                # Option value from continuation
                continuation_value = (p * option_values[i] + (1 - p) * option_values[i + 1]) * np.exp(-r * dt)
                
                if exercise_style == ExerciseStyle.AMERICAN:
                    # Current asset price
                    current_asset_price = S * (u ** (j - i)) * (d ** i)
                    
                    # Exercise value
                    if option_type == OptionType.CALL:
                        exercise_value = max(current_asset_price - K, 0)
                    else:
                        exercise_value = max(K - current_asset_price, 0)
                    
                    # American option: max of continuation and exercise
                    option_values[i] = max(continuation_value, exercise_value)
                else:
                    # European option: only continuation
                    option_values[i] = continuation_value
        
        return option_values[0]

class OptionStrategy:
    """
    Base class for option strategies
    """
    
    def __init__(self, name: str):
        self.name = name
        self.positions = []
        self.entry_date = datetime.now()
    
    def add_position(self, position: OptionPosition):
        """Add option position to strategy"""
        self.positions.append(position)
    
    def calculate_payoff(self, underlying_prices: np.ndarray) -> np.ndarray:
        """Calculate strategy payoff at different underlying prices"""
        total_payoff = np.zeros_like(underlying_prices)
        
        for position in self.positions:
            option = position.option
            quantity = position.quantity
            
            # Calculate individual option payoffs
            if option.option_type == OptionType.CALL:
                option_payoff = np.maximum(underlying_prices - option.strike_price, 0)
            else:
                option_payoff = np.maximum(option.strike_price - underlying_prices, 0)
            
            # Account for position size (long vs short)
            total_payoff += quantity * option_payoff
        
        return total_payoff
    
    def calculate_profit_loss(self, underlying_prices: np.ndarray) -> np.ndarray:
        """Calculate profit/loss including premiums paid/received"""
        payoff = self.calculate_payoff(underlying_prices)
        
        # Subtract net premium paid
        net_premium = sum(pos.quantity * pos.entry_price for pos in self.positions)
        
        return payoff - net_premium
    
    def get_breakeven_points(self) -> List[float]:
        """Find breakeven points for the strategy"""
        # Create price range around strikes
        strikes = [pos.option.strike_price for pos in self.positions]
        min_strike = min(strikes) * 0.8
        max_strike = max(strikes) * 1.2
        
        prices = np.linspace(min_strike, max_strike, 1000)
        pnl = self.calculate_profit_loss(prices)
        
        # Find where P&L crosses zero
        breakevens = []
        for i in range(len(pnl) - 1):
            if pnl[i] * pnl[i + 1] < 0:  # Sign change
                # Linear interpolation to find exact breakeven
                breakeven = prices[i] - pnl[i] * (prices[i + 1] - prices[i]) / (pnl[i + 1] - pnl[i])
                breakevens.append(breakeven)
        
        return breakevens
    
    def get_max_profit(self) -> float:
        """Calculate maximum profit potential"""
        strikes = [pos.option.strike_price for pos in self.positions]
        min_strike = min(strikes) * 0.5
        max_strike = max(strikes) * 1.5
        
        prices = np.linspace(min_strike, max_strike, 1000)
        pnl = self.calculate_profit_loss(prices)
        
        return np.max(pnl)
    
    def get_max_loss(self) -> float:
        """Calculate maximum loss potential"""
        strikes = [pos.option.strike_price for pos in self.positions]
        min_strike = min(strikes) * 0.5
        max_strike = max(strikes) * 1.5
        
        prices = np.linspace(min_strike, max_strike, 1000)
        pnl = self.calculate_profit_loss(prices)
        
        return np.min(pnl)
    
    def plot_payoff_diagram(self, current_price: Optional[float] = None, save_path: Optional[str] = None):
        """Plot strategy payoff diagram"""
        strikes = [pos.option.strike_price for pos in self.positions]
        min_price = min(strikes) * 0.8
        max_price = max(strikes) * 1.2
        
        prices = np.linspace(min_price, max_price, 200)
        payoffs = self.calculate_payoff(prices)
        pnl = self.calculate_profit_loss(prices)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Payoff diagram
        ax1.plot(prices, payoffs, 'b-', linewidth=2, label='Payoff')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.fill_between(prices, payoffs, 0, alpha=0.3)
        ax1.set_title(f'{self.name} - Payoff at Expiration')
        ax1.set_ylabel('Payoff ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # P&L diagram
        ax2.plot(prices, pnl, 'r-', linewidth=2, label='Profit/Loss')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.fill_between(prices, pnl, 0, where=(pnl >= 0), alpha=0.3, color='green', label='Profit')
        ax2.fill_between(prices, pnl, 0, where=(pnl < 0), alpha=0.3, color='red', label='Loss')
        
        # Mark breakeven points
        breakevens = self.get_breakeven_points()
        for be in breakevens:
            ax2.axvline(x=be, color='orange', linestyle=':', alpha=0.7)
            ax2.annotate(f'BE: ${be:.2f}', xy=(be, 0), xytext=(10, 20),
                        textcoords='offset points', fontsize=8)
        
        # Mark current price if provided
        if current_price:
            for ax in [ax1, ax2]:
                ax.axvline(x=current_price, color='purple', linestyle='-', alpha=0.7, 
                          label=f'Current: ${current_price:.2f}')
        
        ax2.set_title(f'{self.name} - Profit/Loss (including premiums)')
        ax2.set_xlabel('Underlying Price ($)')
        ax2.set_ylabel('Profit/Loss ($)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class CommonOptionStrategies:
    """
    Factory class for common option strategies
    """
    
    @staticmethod
    def covered_call(stock_price: float, call_strike: float, call_premium: float,
                    expiration_date: datetime, underlying_symbol: str = "STOCK") -> OptionStrategy:
        """
        Create covered call strategy (long stock + short call)
        """
        strategy = OptionStrategy("Covered Call")
        
        # Short call option
        call_option = Option(
            symbol=f"{underlying_symbol}_C_{call_strike}_{expiration_date.strftime('%Y%m%d')}",
            option_type=OptionType.CALL,
            strike_price=call_strike,
            expiration_date=expiration_date,
            premium=call_premium,
            underlying_symbol=underlying_symbol
        )
        
        call_position = OptionPosition(
            option=call_option,
            quantity=-1,  # Short position
            entry_price=call_premium,
            entry_date=datetime.now()
        )
        
        strategy.add_position(call_position)
        return strategy
    
    @staticmethod
    def protective_put(stock_price: float, put_strike: float, put_premium: float,
                      expiration_date: datetime, underlying_symbol: str = "STOCK") -> OptionStrategy:
        """
        Create protective put strategy (long stock + long put)
        """
        strategy = OptionStrategy("Protective Put")
        
        # Long put option
        put_option = Option(
            symbol=f"{underlying_symbol}_P_{put_strike}_{expiration_date.strftime('%Y%m%d')}",
            option_type=OptionType.PUT,
            strike_price=put_strike,
            expiration_date=expiration_date,
            premium=put_premium,
            underlying_symbol=underlying_symbol
        )
        
        put_position = OptionPosition(
            option=put_option,
            quantity=1,  # Long position
            entry_price=put_premium,
            entry_date=datetime.now()
        )
        
        strategy.add_position(put_position)
        return strategy
    
    @staticmethod
    def bull_call_spread(lower_strike: float, higher_strike: float,
                        lower_call_premium: float, higher_call_premium: float,
                        expiration_date: datetime, underlying_symbol: str = "STOCK") -> OptionStrategy:
        """
        Create bull call spread (long lower strike call + short higher strike call)
        """
        strategy = OptionStrategy("Bull Call Spread")
        
        # Long call (lower strike)
        long_call = Option(
            symbol=f"{underlying_symbol}_C_{lower_strike}_{expiration_date.strftime('%Y%m%d')}",
            option_type=OptionType.CALL,
            strike_price=lower_strike,
            expiration_date=expiration_date,
            premium=lower_call_premium,
            underlying_symbol=underlying_symbol
        )
        
        long_call_position = OptionPosition(
            option=long_call,
            quantity=1,
            entry_price=lower_call_premium,
            entry_date=datetime.now()
        )
        
        # Short call (higher strike)
        short_call = Option(
            symbol=f"{underlying_symbol}_C_{higher_strike}_{expiration_date.strftime('%Y%m%d')}",
            option_type=OptionType.CALL,
            strike_price=higher_strike,
            expiration_date=expiration_date,
            premium=higher_call_premium,
            underlying_symbol=underlying_symbol
        )
        
        short_call_position = OptionPosition(
            option=short_call,
            quantity=-1,
            entry_price=higher_call_premium,
            entry_date=datetime.now()
        )
        
        strategy.add_position(long_call_position)
        strategy.add_position(short_call_position)
        return strategy
    
    @staticmethod
    def bear_put_spread(lower_strike: float, higher_strike: float,
                       lower_put_premium: float, higher_put_premium: float,
                       expiration_date: datetime, underlying_symbol: str = "STOCK") -> OptionStrategy:
        """
        Create bear put spread (short lower strike put + long higher strike put)
        """
        strategy = OptionStrategy("Bear Put Spread")
        
        # Short put (lower strike)
        short_put = Option(
            symbol=f"{underlying_symbol}_P_{lower_strike}_{expiration_date.strftime('%Y%m%d')}",
            option_type=OptionType.PUT,
            strike_price=lower_strike,
            expiration_date=expiration_date,
            premium=lower_put_premium,
            underlying_symbol=underlying_symbol
        )
        
        short_put_position = OptionPosition(
            option=short_put,
            quantity=-1,
            entry_price=lower_put_premium,
            entry_date=datetime.now()
        )
        
        # Long put (higher strike)
        long_put = Option(
            symbol=f"{underlying_symbol}_P_{higher_strike}_{expiration_date.strftime('%Y%m%d')}",
            option_type=OptionType.PUT,
            strike_price=higher_strike,
            expiration_date=expiration_date,
            premium=higher_put_premium,
            underlying_symbol=underlying_symbol
        )
        
        long_put_position = OptionPosition(
            option=long_put,
            quantity=1,
            entry_price=higher_put_premium,
            entry_date=datetime.now()
        )
        
        strategy.add_position(short_put_position)
        strategy.add_position(long_put_position)
        return strategy
    
    @staticmethod
    def straddle(strike_price: float, call_premium: float, put_premium: float,
                expiration_date: datetime, underlying_symbol: str = "STOCK",
                is_long: bool = True) -> OptionStrategy:
        """
        Create straddle strategy (long/short call and put at same strike)
        """
        strategy_name = "Long Straddle" if is_long else "Short Straddle"
        strategy = OptionStrategy(strategy_name)
        
        multiplier = 1 if is_long else -1
        
        # Call option
        call_option = Option(
            symbol=f"{underlying_symbol}_C_{strike_price}_{expiration_date.strftime('%Y%m%d')}",
            option_type=OptionType.CALL,
            strike_price=strike_price,
            expiration_date=expiration_date,
            premium=call_premium,
            underlying_symbol=underlying_symbol
        )
        
        call_position = OptionPosition(
            option=call_option,
            quantity=multiplier,
            entry_price=call_premium,
            entry_date=datetime.now()
        )
        
        # Put option
        put_option = Option(
            symbol=f"{underlying_symbol}_P_{strike_price}_{expiration_date.strftime('%Y%m%d')}",
            option_type=OptionType.PUT,
            strike_price=strike_price,
            expiration_date=expiration_date,
            premium=put_premium,
            underlying_symbol=underlying_symbol
        )
        
        put_position = OptionPosition(
            option=put_option,
            quantity=multiplier,
            entry_price=put_premium,
            entry_date=datetime.now()
        )
        
        strategy.add_position(call_position)
        strategy.add_position(put_position)
        return strategy
    
    @staticmethod
    def strangle(call_strike: float, put_strike: float, call_premium: float, put_premium: float,
                expiration_date: datetime, underlying_symbol: str = "STOCK",
                is_long: bool = True) -> OptionStrategy:
        """
        Create strangle strategy (long/short call and put at different strikes)
        """
        strategy_name = "Long Strangle" if is_long else "Short Strangle"
        strategy = OptionStrategy(strategy_name)
        
        multiplier = 1 if is_long else -1
        
        # Call option (higher strike)
        call_option = Option(
            symbol=f"{underlying_symbol}_C_{call_strike}_{expiration_date.strftime('%Y%m%d')}",
            option_type=OptionType.CALL,
            strike_price=call_strike,
            expiration_date=expiration_date,
            premium=call_premium,
            underlying_symbol=underlying_symbol
        )
        
        call_position = OptionPosition(
            option=call_option,
            quantity=multiplier,
            entry_price=call_premium,
            entry_date=datetime.now()
        )
        
        # Put option (lower strike)
        put_option = Option(
            symbol=f"{underlying_symbol}_P_{put_strike}_{expiration_date.strftime('%Y%m%d')}",
            option_type=OptionType.PUT,
            strike_price=put_strike,
            expiration_date=expiration_date,
            premium=put_premium,
            underlying_symbol=underlying_symbol
        )
        
        put_position = OptionPosition(
            option=put_option,
            quantity=multiplier,
            entry_price=put_premium,
            entry_date=datetime.now()
        )
        
        strategy.add_position(call_position)
        strategy.add_position(put_position)
        return strategy
    
    @staticmethod
    def iron_condor(call_spread_strikes: Tuple[float, float], 
                   put_spread_strikes: Tuple[float, float],
                   call_premiums: Tuple[float, float], 
                   put_premiums: Tuple[float, float],
                   expiration_date: datetime, underlying_symbol: str = "STOCK") -> OptionStrategy:
        """
        Create iron condor strategy (short strangle + long strangle protection)
        """
        strategy = OptionStrategy("Iron Condor")
        
        put_low_strike, put_high_strike = put_spread_strikes
        call_low_strike, call_high_strike = call_spread_strikes
        put_low_premium, put_high_premium = put_premiums
        call_low_premium, call_high_premium = call_premiums
        
        # Long put (lowest strike)
        long_put_low = Option(
            symbol=f"{underlying_symbol}_P_{put_low_strike}_{expiration_date.strftime('%Y%m%d')}",
            option_type=OptionType.PUT,
            strike_price=put_low_strike,
            expiration_date=expiration_date,
            premium=put_low_premium,
            underlying_symbol=underlying_symbol
        )
        
        strategy.add_position(OptionPosition(long_put_low, 1, put_low_premium, datetime.now()))
        
        # Short put (higher strike)
        short_put_high = Option(
            symbol=f"{underlying_symbol}_P_{put_high_strike}_{expiration_date.strftime('%Y%m%d')}",
            option_type=OptionType.PUT,
            strike_price=put_high_strike,
            expiration_date=expiration_date,
            premium=put_high_premium,
            underlying_symbol=underlying_symbol
        )
        
        strategy.add_position(OptionPosition(short_put_high, -1, put_high_premium, datetime.now()))
        
        # Short call (lower strike)
        short_call_low = Option(
            symbol=f"{underlying_symbol}_C_{call_low_strike}_{expiration_date.strftime('%Y%m%d')}",
            option_type=OptionType.CALL,
            strike_price=call_low_strike,
            expiration_date=expiration_date,
            premium=call_low_premium,
            underlying_symbol=underlying_symbol
        )
        
        strategy.add_position(OptionPosition(short_call_low, -1, call_low_premium, datetime.now()))
        
        # Long call (highest strike)
        long_call_high = Option(
            symbol=f"{underlying_symbol}_C_{call_high_strike}_{expiration_date.strftime('%Y%m%d')}",
            option_type=OptionType.CALL,
            strike_price=call_high_strike,
            expiration_date=expiration_date,
            premium=call_high_premium,
            underlying_symbol=underlying_symbol
        )
        
        strategy.add_position(OptionPosition(long_call_high, 1, call_high_premium, datetime.now()))
        
        return strategy

class OptionsPortfolioManager:
    """
    Manage a portfolio of option strategies
    """
    
    def __init__(self):
        self.strategies = {}
        self.current_positions = {}
        self.historical_performance = []
        
    def add_strategy(self, strategy_id: str, strategy: OptionStrategy):
        """Add strategy to portfolio"""
        self.strategies[strategy_id] = strategy
        
    def remove_strategy(self, strategy_id: str):
        """Remove strategy from portfolio"""
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
    
    def calculate_portfolio_greeks(self, current_prices: Dict[str, float],
                                 risk_free_rate: float = 0.05,
                                 volatilities: Dict[str, float] = None) -> Dict[str, float]:
        """
        Calculate portfolio-level Greeks
        """
        if volatilities is None:
            volatilities = {}
        
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        total_rho = 0
        
        for strategy_id, strategy in self.strategies.items():
            for position in strategy.positions:
                option = position.option
                underlying_symbol = option.underlying_symbol or "STOCK"
                
                if underlying_symbol not in current_prices:
                    continue
                
                S = current_prices[underlying_symbol]
                K = option.strike_price
                T = option.time_to_expiration()
                r = risk_free_rate
                sigma = volatilities.get(underlying_symbol, 0.20)  # Default 20% vol
                
                if T <= 0:
                    continue
                
                greeks = BlackScholesModel.calculate_greeks(S, K, T, r, sigma, option.option_type)
                quantity = position.quantity
                
                total_delta += quantity * greeks['delta']
                total_gamma += quantity * greeks['gamma']
                total_theta += quantity * greeks['theta']
                total_vega += quantity * greeks['vega']
                total_rho += quantity * greeks['rho']
        
        return {
            'portfolio_delta': total_delta,
            'portfolio_gamma': total_gamma,
            'portfolio_theta': total_theta,
            'portfolio_vega': total_vega,
            'portfolio_rho': total_rho
        }
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float],
                                risk_free_rate: float = 0.05,
                                volatilities: Dict[str, float] = None) -> float:
        """
        Calculate current portfolio market value
        """
        if volatilities is None:
            volatilities = {}
        
        total_value = 0
        
        for strategy_id, strategy in self.strategies.items():
            for position in strategy.positions:
                option = position.option
                underlying_symbol = option.underlying_symbol or "STOCK"
                
                if underlying_symbol not in current_prices:
                    continue
                
                S = current_prices[underlying_symbol]
                K = option.strike_price
                T = option.time_to_expiration()
                r = risk_free_rate
                sigma = volatilities.get(underlying_symbol, 0.20)
                
                # Calculate option value
                if T <= 0:
                    # At expiration
                    if option.option_type == OptionType.CALL:
                        option_value = max(S - K, 0)
                    else:
                        option_value = max(K - S, 0)
                else:
                    option_value = BlackScholesModel.option_price(S, K, T, r, sigma, option.option_type)
                
                position_value = position.quantity * option_value
                total_value += position_value
        
        return total_value
    
    def generate_portfolio_report(self, current_prices: Dict[str, float],
                                risk_free_rate: float = 0.05,
                                volatilities: Dict[str, float] = None) -> Dict:
        """
        Generate comprehensive portfolio report
        """
        portfolio_greeks = self.calculate_portfolio_greeks(current_prices, risk_free_rate, volatilities)
        portfolio_value = self.calculate_portfolio_value(current_prices, risk_free_rate, volatilities)
        
        # Strategy-level analysis
        strategy_analysis = {}
        for strategy_id, strategy in self.strategies.items():
            breakevens = strategy.get_breakeven_points()
            max_profit = strategy.get_max_profit()
            max_loss = strategy.get_max_loss()
            
            strategy_analysis[strategy_id] = {
                'strategy_name': strategy.name,
                'num_positions': len(strategy.positions),
                'breakeven_points': breakevens,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'entry_date': strategy.entry_date.isoformat()
            }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': portfolio_value,
            'portfolio_greeks': portfolio_greeks,
            'num_strategies': len(self.strategies),
            'strategy_analysis': strategy_analysis,
            'risk_metrics': {
                'net_delta': portfolio_greeks['portfolio_delta'],
                'gamma_exposure': portfolio_greeks['portfolio_gamma'],
                'theta_decay': portfolio_greeks['portfolio_theta'],
                'vega_exposure': portfolio_greeks['portfolio_vega']
            }
        }
    
    def plot_portfolio_payoff(self, current_prices: Dict[str, float], save_path: Optional[str] = None):
        """
        Plot combined portfolio payoff diagram
        """
        if not self.strategies:
            print("No strategies in portfolio")
            return
        
        # Determine price range
        all_strikes = []
        for strategy in self.strategies.values():
            all_strikes.extend([pos.option.strike_price for pos in strategy.positions])
        
        min_price = min(all_strikes) * 0.8
        max_price = max(all_strikes) * 1.2
        prices = np.linspace(min_price, max_price, 200)
        
        # Calculate combined payoff
        total_payoff = np.zeros_like(prices)
        total_pnl = np.zeros_like(prices)
        
        plt.figure(figsize=(15, 10))
        
        # Plot individual strategies
        for i, (strategy_id, strategy) in enumerate(self.strategies.items()):
            strategy_payoff = strategy.calculate_payoff(prices)
            strategy_pnl = strategy.calculate_profit_loss(prices)
            
            total_payoff += strategy_payoff
            total_pnl += strategy_pnl
            
            plt.subplot(2, 2, 1)
            plt.plot(prices, strategy_pnl, label=f'{strategy_id}: {strategy.name}', alpha=0.7)
        
        # Individual strategies P&L
        plt.subplot(2, 2, 1)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.title('Individual Strategy P&L')
        plt.xlabel('Underlying Price ($)')
        plt.ylabel('Profit/Loss ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Combined portfolio P&L
        plt.subplot(2, 2, 2)
        plt.plot(prices, total_pnl, 'r-', linewidth=3, label='Combined Portfolio')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.fill_between(prices, total_pnl, 0, where=(total_pnl >= 0), alpha=0.3, color='green')
        plt.fill_between(prices, total_pnl, 0, where=(total_pnl < 0), alpha=0.3, color='red')
        
        # Mark current prices
        for symbol, price in current_prices.items():
            plt.axvline(x=price, color='purple', linestyle=':', alpha=0.7, 
                       label=f'{symbol}: ${price:.2f}')
        
        plt.title('Combined Portfolio P&L')
        plt.xlabel('Underlying Price ($)')
        plt.ylabel('Profit/Loss ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Greeks over time (if we had historical data)
        plt.subplot(2, 2, 3)
        greeks = self.calculate_portfolio_greeks(current_prices)
        greek_names = list(greeks.keys())
        greek_values = list(greeks.values())
        
        plt.bar([name.replace('portfolio_', '') for name in greek_names], greek_values)
        plt.title('Portfolio Greeks')
        plt.ylabel('Greek Value')
        plt.xticks(rotation=45)
        
        # Risk metrics
        plt.subplot(2, 2, 4)
        risk_data = {
            'Max Profit': np.max(total_pnl),
            'Max Loss': np.min(total_pnl),
            'Current Value': self.calculate_portfolio_value(current_prices)
        }
        
        plt.bar(risk_data.keys(), risk_data.values())
        plt.title('Portfolio Risk Metrics')
        plt.ylabel('Value ($)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()