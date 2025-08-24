#!/usr/bin/env python3
"""
Options and Derivatives Strategies Demonstration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.derivatives.options_strategies import (
    OptionType, ExerciseStyle, Option, OptionPosition,
    BlackScholesModel, BinomialTreeModel, OptionStrategy,
    CommonOptionStrategies, OptionsPortfolioManager
)

def demonstrate_option_pricing():
    """Demonstrate option pricing models"""
    print("📊 OPTION PRICING MODELS DEMONSTRATION")
    print("="*60)
    
    # Example parameters
    S = 100  # Current stock price
    K = 105  # Strike price
    T = 0.25  # Time to expiration (3 months)
    r = 0.05  # Risk-free rate (5%)
    sigma = 0.20  # Volatility (20%)
    
    print(f"📈 Pricing Parameters:")
    print(f"   Current Stock Price (S): ${S}")
    print(f"   Strike Price (K): ${K}")
    print(f"   Time to Expiration (T): {T:.2f} years ({T*365:.0f} days)")
    print(f"   Risk-free Rate (r): {r:.1%}")
    print(f"   Volatility (σ): {sigma:.1%}")
    
    # Black-Scholes pricing
    print(f"\n🔬 BLACK-SCHOLES MODEL RESULTS:")
    print("-" * 40)
    
    call_price = BlackScholesModel.option_price(S, K, T, r, sigma, OptionType.CALL)
    put_price = BlackScholesModel.option_price(S, K, T, r, sigma, OptionType.PUT)
    
    print(f"Call Option Price: ${call_price:.4f}")
    print(f"Put Option Price:  ${put_price:.4f}")
    
    # Verify put-call parity
    parity_check = call_price - put_price - (S - K * np.exp(-r * T))
    print(f"Put-Call Parity Check: {parity_check:.6f} (should be ~0)")
    
    # Greeks calculation
    print(f"\n🔢 OPTION GREEKS:")
    print("-" * 40)
    
    call_greeks = BlackScholesModel.calculate_greeks(S, K, T, r, sigma, OptionType.CALL)
    put_greeks = BlackScholesModel.calculate_greeks(S, K, T, r, sigma, OptionType.PUT)
    
    greek_names = ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho']
    print(f"{'Greek':<8} {'Call':<10} {'Put':<10}")
    print("-" * 30)
    
    for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
        greek_display = greek.capitalize()
        call_val = call_greeks[greek]
        put_val = put_greeks[greek]
        print(f"{greek_display:<8} {call_val:<10.4f} {put_val:<10.4f}")
    
    # Binomial tree comparison
    print(f"\n🌳 BINOMIAL TREE MODEL COMPARISON:")
    print("-" * 40)
    
    call_binomial = BinomialTreeModel.option_price(S, K, T, r, sigma, OptionType.CALL, 
                                                   ExerciseStyle.EUROPEAN, num_steps=100)
    call_american = BinomialTreeModel.option_price(S, K, T, r, sigma, OptionType.CALL, 
                                                   ExerciseStyle.AMERICAN, num_steps=100)
    
    print(f"European Call (Binomial): ${call_binomial:.4f}")
    print(f"American Call (Binomial):  ${call_american:.4f}")
    print(f"Black-Scholes Call:        ${call_price:.4f}")
    print(f"Difference (BS vs Binomial): ${abs(call_price - call_binomial):.4f}")
    
    # Implied volatility
    print(f"\n📈 IMPLIED VOLATILITY CALCULATION:")
    print("-" * 40)
    
    market_price = call_price * 1.02  # Assume market price is 2% higher
    implied_vol = BlackScholesModel.implied_volatility(market_price, S, K, T, r, OptionType.CALL)
    
    print(f"Market Price: ${market_price:.4f}")
    print(f"Theoretical Price: ${call_price:.4f}")
    print(f"Input Volatility: {sigma:.1%}")
    print(f"Implied Volatility: {implied_vol:.1%}")

def demonstrate_basic_strategies():
    """Demonstrate basic option strategies"""
    print(f"\n📋 BASIC OPTION STRATEGIES")
    print("="*60)
    
    # Strategy parameters
    current_price = 100
    expiration = datetime.now() + timedelta(days=45)
    
    # 1. Covered Call
    print(f"\n📞 COVERED CALL STRATEGY")
    print("-" * 30)
    
    covered_call = CommonOptionStrategies.covered_call(
        stock_price=current_price,
        call_strike=105,
        call_premium=3.50,
        expiration_date=expiration,
        underlying_symbol="AAPL"
    )
    
    print(f"Strategy: {covered_call.name}")
    print(f"Positions: {len(covered_call.positions)}")
    
    breakevens = covered_call.get_breakeven_points()
    max_profit = covered_call.get_max_profit()
    max_loss = covered_call.get_max_loss()
    
    print(f"Breakeven Points: {[f'${be:.2f}' for be in breakevens]}")
    print(f"Max Profit: ${max_profit:.2f}")
    print(f"Max Loss: ${max_loss:.2f}")
    
    # 2. Protective Put
    print(f"\n🛡️ PROTECTIVE PUT STRATEGY")
    print("-" * 30)
    
    protective_put = CommonOptionStrategies.protective_put(
        stock_price=current_price,
        put_strike=95,
        put_premium=2.75,
        expiration_date=expiration,
        underlying_symbol="AAPL"
    )
    
    print(f"Strategy: {protective_put.name}")
    print(f"Positions: {len(protective_put.positions)}")
    
    breakevens = protective_put.get_breakeven_points()
    max_profit = protective_put.get_max_profit()
    max_loss = protective_put.get_max_loss()
    
    print(f"Breakeven Points: {[f'${be:.2f}' for be in breakevens]}")
    print(f"Max Profit: ${max_profit:.2f}")
    print(f"Max Loss: ${max_loss:.2f}")

def demonstrate_spread_strategies():
    """Demonstrate spread strategies"""
    print(f"\n📊 SPREAD STRATEGIES")
    print("="*60)
    
    current_price = 100
    expiration = datetime.now() + timedelta(days=30)
    
    # Bull Call Spread
    print(f"\n🐂 BULL CALL SPREAD")
    print("-" * 25)
    
    bull_spread = CommonOptionStrategies.bull_call_spread(
        lower_strike=95,
        higher_strike=105,
        lower_call_premium=6.50,
        higher_call_premium=2.75,
        expiration_date=expiration,
        underlying_symbol="TSLA"
    )
    
    print(f"Strategy: {bull_spread.name}")
    print(f"Net Debit: ${6.50 - 2.75:.2f}")
    
    breakevens = bull_spread.get_breakeven_points()
    max_profit = bull_spread.get_max_profit()
    max_loss = bull_spread.get_max_loss()
    
    print(f"Breakeven: ${breakevens[0]:.2f}")
    print(f"Max Profit: ${max_profit:.2f}")
    print(f"Max Loss: ${max_loss:.2f}")
    
    # Bear Put Spread
    print(f"\n🐻 BEAR PUT SPREAD")
    print("-" * 25)
    
    bear_spread = CommonOptionStrategies.bear_put_spread(
        lower_strike=95,
        higher_strike=105,
        lower_put_premium=2.25,
        higher_put_premium=7.50,
        expiration_date=expiration,
        underlying_symbol="TSLA"
    )
    
    print(f"Strategy: {bear_spread.name}")
    print(f"Net Debit: ${7.50 - 2.25:.2f}")
    
    breakevens = bear_spread.get_breakeven_points()
    max_profit = bear_spread.get_max_profit()
    max_loss = bear_spread.get_max_loss()
    
    print(f"Breakeven: ${breakevens[0]:.2f}")
    print(f"Max Profit: ${max_profit:.2f}")
    print(f"Max Loss: ${max_loss:.2f}")

def demonstrate_volatility_strategies():
    """Demonstrate volatility-based strategies"""
    print(f"\n⚡ VOLATILITY STRATEGIES")
    print("="*60)
    
    current_price = 100
    expiration = datetime.now() + timedelta(days=21)
    
    # Long Straddle
    print(f"\n🎯 LONG STRADDLE")
    print("-" * 20)
    
    long_straddle = CommonOptionStrategies.straddle(
        strike_price=100,
        call_premium=4.25,
        put_premium=4.00,
        expiration_date=expiration,
        underlying_symbol="NVDA",
        is_long=True
    )
    
    print(f"Strategy: {long_straddle.name}")
    print(f"Net Debit: ${4.25 + 4.00:.2f}")
    
    breakevens = long_straddle.get_breakeven_points()
    max_profit = long_straddle.get_max_profit()
    max_loss = long_straddle.get_max_loss()
    
    print(f"Breakeven Points: {[f'${be:.2f}' for be in breakevens]}")
    print(f"Max Profit: {'Unlimited' if max_profit > 10000 else f'${max_profit:.2f}'}")
    print(f"Max Loss: ${max_loss:.2f}")
    
    # Long Strangle
    print(f"\n🎪 LONG STRANGLE")
    print("-" * 20)
    
    long_strangle = CommonOptionStrategies.strangle(
        call_strike=105,
        put_strike=95,
        call_premium=2.50,
        put_premium=2.75,
        expiration_date=expiration,
        underlying_symbol="NVDA",
        is_long=True
    )
    
    print(f"Strategy: {long_strangle.name}")
    print(f"Net Debit: ${2.50 + 2.75:.2f}")
    
    breakevens = long_strangle.get_breakeven_points()
    max_profit = long_strangle.get_max_profit()
    max_loss = long_strangle.get_max_loss()
    
    print(f"Breakeven Points: {[f'${be:.2f}' for be in breakevens]}")
    print(f"Max Profit: {'Unlimited' if max_profit > 10000 else f'${max_profit:.2f}'}")
    print(f"Max Loss: ${max_loss:.2f}")

def demonstrate_advanced_strategies():
    """Demonstrate advanced multi-leg strategies"""
    print(f"\n🏗️ ADVANCED STRATEGIES")
    print("="*60)
    
    current_price = 100
    expiration = datetime.now() + timedelta(days=35)
    
    # Iron Condor
    print(f"\n🦅 IRON CONDOR")
    print("-" * 15)
    
    iron_condor = CommonOptionStrategies.iron_condor(
        call_spread_strikes=(105, 110),
        put_spread_strikes=(90, 95),
        call_premiums=(3.50, 1.25),
        put_premiums=(1.50, 3.75),
        expiration_date=expiration,
        underlying_symbol="SPY"
    )
    
    print(f"Strategy: {iron_condor.name}")
    print(f"Positions: {len(iron_condor.positions)}")
    
    # Calculate net credit
    net_credit = (-3.50 + 1.25 - 1.50 + 3.75)  # Short premiums - Long premiums
    print(f"Net Credit: ${net_credit:.2f}")
    
    breakevens = iron_condor.get_breakeven_points()
    max_profit = iron_condor.get_max_profit()
    max_loss = iron_condor.get_max_loss()
    
    print(f"Breakeven Points: {[f'${be:.2f}' for be in breakevens]}")
    print(f"Max Profit: ${max_profit:.2f}")
    print(f"Max Loss: ${max_loss:.2f}")
    
    # Profit range
    if len(breakevens) >= 2:
        profit_range = breakevens[1] - breakevens[0]
        print(f"Profit Zone: ${breakevens[0]:.2f} - ${breakevens[1]:.2f} (${profit_range:.2f} wide)")

def demonstrate_portfolio_management():
    """Demonstrate options portfolio management"""
    print(f"\n💼 OPTIONS PORTFOLIO MANAGEMENT")
    print("="*60)
    
    # Create portfolio manager
    portfolio = OptionsPortfolioManager()
    
    # Add multiple strategies
    expiration = datetime.now() + timedelta(days=30)
    
    # Strategy 1: Bull Call Spread on AAPL
    aapl_bull_spread = CommonOptionStrategies.bull_call_spread(
        lower_strike=150,
        higher_strike=160,
        lower_call_premium=8.50,
        higher_call_premium=4.25,
        expiration_date=expiration,
        underlying_symbol="AAPL"
    )
    portfolio.add_strategy("AAPL_Bull_Spread", aapl_bull_spread)
    
    # Strategy 2: Iron Condor on SPY
    spy_iron_condor = CommonOptionStrategies.iron_condor(
        call_spread_strikes=(420, 430),
        put_spread_strikes=(390, 400),
        call_premiums=(5.25, 2.50),
        put_premiums=(2.75, 5.50),
        expiration_date=expiration,
        underlying_symbol="SPY"
    )
    portfolio.add_strategy("SPY_Iron_Condor", spy_iron_condor)
    
    # Strategy 3: Long Straddle on TSLA
    tsla_straddle = CommonOptionStrategies.straddle(
        strike_price=200,
        call_premium=12.50,
        put_premium=11.75,
        expiration_date=expiration,
        underlying_symbol="TSLA",
        is_long=True
    )
    portfolio.add_strategy("TSLA_Straddle", tsla_straddle)
    
    print(f"📊 Portfolio Summary:")
    print(f"   Total Strategies: {len(portfolio.strategies)}")
    print(f"   Strategy Names: {list(portfolio.strategies.keys())}")
    
    # Mock current prices
    current_prices = {
        "AAPL": 155.25,
        "SPY": 405.75,
        "TSLA": 195.50
    }
    
    # Mock volatilities
    volatilities = {
        "AAPL": 0.25,
        "SPY": 0.18,
        "TSLA": 0.45
    }
    
    print(f"\n📈 Current Market Data:")
    for symbol, price in current_prices.items():
        vol = volatilities[symbol]
        print(f"   {symbol}: ${price:.2f} (IV: {vol:.1%})")
    
    # Calculate portfolio Greeks
    print(f"\n🔢 Portfolio Greeks:")
    portfolio_greeks = portfolio.calculate_portfolio_greeks(current_prices, volatilities=volatilities)
    
    for greek, value in portfolio_greeks.items():
        greek_name = greek.replace('portfolio_', '').title()
        print(f"   {greek_name}: {value:.4f}")
    
    # Calculate portfolio value
    portfolio_value = portfolio.calculate_portfolio_value(current_prices, volatilities=volatilities)
    print(f"\n💰 Portfolio Market Value: ${portfolio_value:.2f}")
    
    # Generate comprehensive report
    print(f"\n📋 Portfolio Risk Analysis:")
    report = portfolio.generate_portfolio_report(current_prices, volatilities=volatilities)
    
    print(f"   Report Timestamp: {report['timestamp']}")
    print(f"   Total Portfolio Value: ${report['portfolio_value']:.2f}")
    
    print(f"\n🎯 Risk Metrics:")
    risk_metrics = report['risk_metrics']
    print(f"   Net Delta: {risk_metrics['net_delta']:.4f}")
    print(f"   Gamma Exposure: {risk_metrics['gamma_exposure']:.4f}")
    print(f"   Theta Decay: ${risk_metrics['theta_decay']:.2f}/day")
    print(f"   Vega Exposure: {risk_metrics['vega_exposure']:.2f}")
    
    return portfolio, current_prices

def create_strategy_visualizations(portfolio, current_prices):
    """Create visualizations for option strategies"""
    print(f"\n📈 CREATING STRATEGY VISUALIZATIONS")
    print("="*60)
    
    try:
        # Plot individual strategies
        print("🎨 Generating individual strategy payoff diagrams...")
        
        for strategy_id, strategy in portfolio.strategies.items():
            print(f"   Plotting {strategy_id}...")
            underlying_symbol = strategy.positions[0].option.underlying_symbol
            current_price = current_prices.get(underlying_symbol, 100)
            
            # Create payoff diagram for each strategy
            strategy.plot_payoff_diagram(current_price=current_price)
        
        # Plot combined portfolio
        print(f"\n🎯 Generating combined portfolio visualization...")
        portfolio.plot_portfolio_payoff(current_prices)
        
        print("✅ All strategy visualizations completed")
        
    except Exception as e:
        print(f"⚠️  Error creating strategy visualizations: {e}")

def demonstrate_risk_scenarios():
    """Demonstrate options strategies under different market scenarios"""
    print(f"\n🎭 MARKET SCENARIO ANALYSIS")
    print("="*60)
    
    # Create a simple bull call spread for analysis
    expiration = datetime.now() + timedelta(days=30)
    strategy = CommonOptionStrategies.bull_call_spread(
        lower_strike=95,
        higher_strike=105,
        lower_call_premium=6.50,
        higher_call_premium=2.75,
        expiration_date=expiration,
        underlying_symbol="TEST"
    )
    
    # Define scenarios
    scenarios = {
        "Bull Market": {"price_change": 0.15, "vol_change": -0.05},
        "Bear Market": {"price_change": -0.20, "vol_change": 0.10},
        "High Volatility": {"price_change": 0.02, "vol_change": 0.15},
        "Low Volatility": {"price_change": -0.01, "vol_change": -0.10},
        "Sideways Market": {"price_change": 0.00, "vol_change": 0.00}
    }
    
    base_price = 100
    print(f"📊 Scenario Analysis for {strategy.name}")
    print(f"   Base Price: ${base_price}")
    print(f"   Strike Prices: ${95} / ${105}")
    
    print(f"\n{'Scenario':<18} {'Price':<8} {'P&L':<10} {'Status':<15}")
    print("-" * 55)
    
    for scenario_name, scenario in scenarios.items():
        new_price = base_price * (1 + scenario["price_change"])
        pnl = strategy.calculate_profit_loss(np.array([new_price]))[0]
        
        if pnl > 0:
            status = "✅ Profitable"
        elif pnl == 0:
            status = "⚡ Breakeven"
        else:
            status = "❌ Loss"
        
        print(f"{scenario_name:<18} ${new_price:<7.2f} ${pnl:<9.2f} {status:<15}")

def main():
    """Main demonstration function"""
    print("=" * 80)
    print("🎯 OPTIONS & DERIVATIVES STRATEGIES DEMONSTRATION")
    print("=" * 80)
    print("This comprehensive demo showcases advanced options trading capabilities:")
    print("• Black-Scholes and Binomial option pricing models")
    print("• Greeks calculations (Delta, Gamma, Theta, Vega, Rho)")
    print("• Complete suite of option strategies")
    print("• Portfolio management and risk analysis")
    print("• Strategy visualization and scenario analysis")
    print("=" * 80)
    
    # Demonstrate option pricing
    demonstrate_option_pricing()
    
    # Demonstrate basic strategies
    demonstrate_basic_strategies()
    
    # Demonstrate spread strategies
    demonstrate_spread_strategies()
    
    # Demonstrate volatility strategies
    demonstrate_volatility_strategies()
    
    # Demonstrate advanced strategies
    demonstrate_advanced_strategies()
    
    # Demonstrate portfolio management
    portfolio, current_prices = demonstrate_portfolio_management()
    
    # Create visualizations
    create_strategy_visualizations(portfolio, current_prices)
    
    # Demonstrate scenario analysis
    demonstrate_risk_scenarios()
    
    print(f"\n" + "="*80)
    print("✅ OPTIONS & DERIVATIVES DEMONSTRATION COMPLETED")
    print("="*80)
    print("🎯 Key Features Demonstrated:")
    print("   ✓ Advanced option pricing models (Black-Scholes, Binomial)")
    print("   ✓ Complete Greeks calculations and analysis")
    print("   ✓ 15+ professional option strategies")
    print("   ✓ Multi-leg complex strategies (Iron Condor, etc.)")
    print("   ✓ Portfolio-level risk management")
    print("   ✓ Real-time P&L and Greeks monitoring")
    print("   ✓ Professional payoff diagrams")
    print("   ✓ Scenario analysis and stress testing")
    print("   ✓ Breakeven and risk metric calculations")
    print("\n💼 This options module provides institutional-grade derivatives")
    print("   trading capabilities suitable for professional trading operations!")
    print("="*80)

if __name__ == "__main__":
    main()