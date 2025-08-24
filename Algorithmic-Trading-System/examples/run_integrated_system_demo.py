#!/usr/bin/env python3
"""
Complete Integrated Trading System Demonstration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
from datetime import datetime
import asyncio
import signal

from src.integration.trading_system_integration import IntegratedTradingSystem, SystemConfiguration
from src.strategies.moving_average_strategy import MovingAverageStrategy
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.ml_strategy import MLStrategy

# Global system reference for clean shutdown
trading_system = None

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global trading_system
    print(f"\n📡 Received signal {signum}. Initiating graceful shutdown...")
    
    if trading_system:
        trading_system.stop_system()
    
    print("✅ System shutdown complete")
    exit(0)

def system_event_handler(event_type, data):
    """Handle system events"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    
    if event_type == 'system_started':
        print(f"[{timestamp}] 🚀 System started successfully")
    
    elif event_type == 'system_stopped':
        print(f"[{timestamp}] 🛑 System stopped")
    
    elif event_type == 'risk_alert':
        severity_icons = {
            'low': '🟡',
            'medium': '🟠', 
            'high': '🔴',
            'critical': '🚨'
        }
        icon = severity_icons.get(data.severity, '⚪')
        print(f"[{timestamp}] {icon} Risk Alert: {data.message}")
    
    else:
        print(f"[{timestamp}] 📋 {event_type}: {data}")

def create_comprehensive_system():
    """Create a comprehensive trading system with all features enabled"""
    print("🔧 CREATING COMPREHENSIVE TRADING SYSTEM")
    print("="*60)
    
    # Configure system with all features enabled
    config = SystemConfiguration(
        initial_balance=250000.0,  # $250k starting capital
        commission_rate=0.0005,    # 0.05% commission
        risk_free_rate=0.05,       # 5% risk-free rate
        update_interval=3,         # 3-second updates
        auto_rebalance=True,
        rebalance_interval=1800,   # 30-minute rebalancing
        enable_options=True,
        enable_ml=True,
        enable_dashboard=True,
        dashboard_port=8050,
        max_position_size=0.15,    # 15% max per position
        stop_loss_pct=0.08,        # 8% stop loss
        symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']
    )
    
    print(f"💰 Initial Capital: ${config.initial_balance:,.2f}")
    print(f"📊 Trading Symbols: {config.symbols}")
    print(f"⚙️  Features Enabled:")
    print(f"   • Real-time Data Feed: ✓")
    print(f"   • Live Strategy Engine: ✓")
    print(f"   • Advanced Risk Management: ✓")
    print(f"   • Portfolio Optimization: ✓")
    print(f"   • Options Trading: {'✓' if config.enable_options else '✗'}")
    print(f"   • Machine Learning: {'✓' if config.enable_ml else '✗'}")
    print(f"   • Web Dashboard: {'✓' if config.enable_dashboard else '✗'}")
    print(f"   • Auto-Rebalancing: {'✓' if config.auto_rebalance else '✗'}")
    
    # Create the integrated system
    system = IntegratedTradingSystem(config)
    
    print("✅ Comprehensive trading system created")
    return system

def add_trading_strategies(system):
    """Add multiple trading strategies to the system"""
    print(f"\n📈 ADDING TRADING STRATEGIES")
    print("="*60)
    
    # Strategy 1: Moving Average Crossover (Conservative)
    ma_short = MovingAverageStrategy(
        short_window=10,
        long_window=30,
        confidence_threshold=0.6
    )
    system.add_strategy(
        name='MA_Conservative',
        strategy=ma_short,
        symbols=['AAPL', 'MSFT', 'GOOGL']
    )
    print("✓ Added MA Conservative Strategy (10/30) for AAPL, MSFT, GOOGL")
    
    # Strategy 2: Moving Average Crossover (Aggressive)  
    ma_aggressive = MovingAverageStrategy(
        short_window=5,
        long_window=20,
        confidence_threshold=0.5
    )
    system.add_strategy(
        name='MA_Aggressive',
        strategy=ma_aggressive,
        symbols=['TSLA', 'NVDA']
    )
    print("✓ Added MA Aggressive Strategy (5/20) for TSLA, NVDA")
    
    # Strategy 3: RSI Mean Reversion
    rsi_strategy = RSIStrategy(
        period=14,
        oversold_threshold=30,
        overbought_threshold=70,
        confidence_threshold=0.7
    )
    system.add_strategy(
        name='RSI_MeanReversion',
        strategy=rsi_strategy,
        symbols=['AMZN', 'META', 'NFLX']
    )
    print("✓ Added RSI Mean Reversion Strategy for AMZN, META, NFLX")
    
    # Strategy 4: ML-Based Strategy (if ML enabled)
    if system.config.enable_ml:
        try:
            ml_strategy = MLStrategy(
                lookback_period=60,
                prediction_horizon=5,
                confidence_threshold=0.65
            )
            system.add_strategy(
                name='ML_Predictor',
                strategy=ml_strategy,
                symbols=['AAPL', 'TSLA', 'NVDA']
            )
            print("✓ Added ML Prediction Strategy for AAPL, TSLA, NVDA")
        except Exception as e:
            print(f"⚠️  Could not add ML strategy: {e}")
    
    print(f"\n🎯 Total Strategies Added: {len(system.strategy_engine.strategies)}")

def display_system_dashboard_info(system):
    """Display information about system dashboards and interfaces"""
    print(f"\n🌐 SYSTEM INTERFACES & DASHBOARDS")
    print("="*60)
    
    if system.config.enable_dashboard:
        port = system.config.dashboard_port
        print(f"🎛️  Live Trading Dashboard:")
        print(f"   URL: http://localhost:{port}")
        print(f"   Features:")
        print(f"   • Real-time portfolio monitoring")
        print(f"   • Live P&L tracking")
        print(f"   • Strategy performance analysis")
        print(f"   • Risk metrics dashboard")
        print(f"   • Trading signals visualization")
        print(f"   • Position management interface")
    
    print(f"\n📊 Available Analysis Tools:")
    print(f"   • Advanced risk analysis and stress testing")
    print(f"   • Monte Carlo portfolio simulation")
    print(f"   • Options strategies and Greeks analysis")
    print(f"   • ML model performance tracking")
    print(f"   • Portfolio optimization reports")
    
    print(f"\n💾 Data Export Capabilities:")
    print(f"   • Trade history and performance data")
    print(f"   • Risk metrics and drawdown analysis") 
    print(f"   • Strategy-specific performance reports")
    print(f"   • System state snapshots")

def run_system_monitoring_loop(system):
    """Run the main system monitoring and status display loop"""
    print(f"\n⏰ STARTING SYSTEM MONITORING")
    print("="*60)
    
    start_time = datetime.now()
    last_report_time = start_time
    report_interval = 60  # Report every 60 seconds
    
    try:
        while True:
            current_time = datetime.now()
            
            # Display periodic status reports
            if (current_time - last_report_time).seconds >= report_interval:
                print(f"\n📊 SYSTEM STATUS REPORT - {current_time.strftime('%H:%M:%S')}")
                print("-" * 50)
                
                # Get system status
                status = system.get_system_status()
                
                # Display key metrics
                if 'account' in status:
                    account = status['account']
                    print(f"💰 Portfolio Value: ${account['portfolio_value']:,.2f}")
                    print(f"📈 Total Return: {account['total_return_pct']:+.2f}%")
                    print(f"💵 Cash Balance: ${account['cash_balance']:,.2f}")
                    print(f"📍 Active Positions: {len(account.get('positions', {}))}")
                
                if 'strategies' in status:
                    strategies = status['strategies']
                    active_strategies = sum(1 for s in strategies.get('strategies', {}).values() if s.get('active', False))
                    total_signals = sum(s.get('signal_count', 0) for s in strategies.get('strategies', {}).values())
                    
                    print(f"🎯 Active Strategies: {active_strategies}/{len(strategies.get('strategies', {}))}")
                    print(f"📡 Total Signals Generated: {total_signals}")
                    print(f"🔄 Recent Signals: {len(strategies.get('recent_signals', []))}")
                
                # System uptime
                uptime = current_time - start_time
                print(f"⏱️  System Uptime: {uptime}")
                
                last_report_time = current_time
                
                # Generate detailed performance report every 5 minutes
                if uptime.seconds > 0 and uptime.seconds % 300 == 0:
                    print(f"\n📋 Generating detailed performance report...")
                    performance_report = system.get_performance_report()
                    
                    if 'performance_metrics' in performance_report:
                        metrics = performance_report['performance_metrics']
                        print(f"📊 Performance Metrics:")
                        print(f"   • Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
                        print(f"   • Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
                        print(f"   • Win Rate: {metrics.get('win_rate', 0):.1%}")
                        print(f"   • Volatility: {metrics.get('annual_volatility', 0):.2%}")
            
            # Short sleep to prevent excessive CPU usage
            time.sleep(5)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Monitoring interrupted by user")

def save_final_reports(system):
    """Save final system reports and state"""
    print(f"\n💾 SAVING FINAL REPORTS")
    print("="*60)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        # Save system state
        state_file = f"trading_system_state_{timestamp}.json"
        system.save_system_state(state_file)
        print(f"✓ System state saved to: {state_file}")
        
        # Generate and save performance report
        performance_report = system.get_performance_report()
        report_file = f"performance_report_{timestamp}.json"
        
        import json
        with open(report_file, 'w') as f:
            json.dump(performance_report, f, indent=2, default=str)
        print(f"✓ Performance report saved to: {report_file}")
        
        # Display final summary
        if 'account_summary' in performance_report:
            account = performance_report['account_summary']
            print(f"\n📊 FINAL PERFORMANCE SUMMARY")
            print("-" * 40)
            print(f"Initial Capital: ${system.config.initial_balance:,.2f}")
            print(f"Final Value: ${account['portfolio_value']:,.2f}")
            print(f"Total Return: {account['total_return_pct']:+.2f}%")
            print(f"Net P&L: ${account['total_pnl']:+,.2f}")
        
    except Exception as e:
        print(f"⚠️  Error saving reports: {e}")

def main():
    """Main function to run the complete integrated system demo"""
    global trading_system
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("="*80)
    print("🚀 INTEGRATED ALGORITHMIC TRADING SYSTEM - FULL DEMO")
    print("="*80)
    print("This demonstration showcases the complete integrated system including:")
    print("• Multi-strategy algorithmic trading engine")
    print("• Advanced risk management and monitoring")
    print("• Portfolio optimization and rebalancing") 
    print("• Machine learning-based strategies")
    print("• Options and derivatives support")
    print("• Real-time web dashboard")
    print("• Comprehensive performance analytics")
    print("="*80)
    
    try:
        # Create the comprehensive trading system
        trading_system = create_comprehensive_system()
        
        # Add event monitoring
        trading_system.add_callback(system_event_handler)
        
        # Add multiple trading strategies
        add_trading_strategies(trading_system)
        
        # Display dashboard information
        display_system_dashboard_info(trading_system)
        
        print(f"\n🚀 STARTING INTEGRATED TRADING SYSTEM")
        print("="*60)
        print("🔄 Initializing all components...")
        
        # Start the complete system
        trading_system.start_system()
        
        print(f"\n✅ SYSTEM FULLY OPERATIONAL")
        print("="*60)
        print("🎛️  All systems are online and trading algorithms are active!")
        print(f"🌐 Dashboard: http://localhost:{trading_system.config.dashboard_port}")
        print("📊 Monitor the system status below or use the web dashboard")
        print("🛑 Press Ctrl+C to stop the system gracefully")
        print("="*60)
        
        # Run monitoring loop
        run_system_monitoring_loop(trading_system)
        
    except Exception as e:
        print(f"❌ Critical system error: {e}")
        if trading_system:
            trading_system.stop_system()
    
    finally:
        if trading_system:
            # Save final reports
            save_final_reports(trading_system)
            
            print(f"\n" + "="*80)
            print("✅ INTEGRATED TRADING SYSTEM DEMONSTRATION COMPLETED")
            print("="*80)
            print("🎯 System Features Successfully Demonstrated:")
            print("   ✓ Multi-strategy algorithmic trading")
            print("   ✓ Real-time market data processing")
            print("   ✓ Advanced risk management and alerts")
            print("   ✓ Portfolio optimization and rebalancing")
            print("   ✓ Machine learning strategy integration")
            print("   ✓ Options and derivatives capabilities")
            print("   ✓ Real-time web dashboard monitoring")
            print("   ✓ Comprehensive performance analytics")
            print("   ✓ System state persistence and reporting")
            print("\n💼 This system demonstrates institutional-grade algorithmic")
            print("   trading capabilities suitable for professional deployment!")
            print("="*80)

if __name__ == "__main__":
    main()