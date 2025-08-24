import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Optional

# Configure page
st.set_page_config(
    page_title="Live Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitTradingDashboard:
    """
    Streamlit-based live trading dashboard
    """
    
    def __init__(self):
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'strategy_engine' not in st.session_state:
            st.session_state.strategy_engine = None
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
    
    def setup_sidebar(self):
        """Setup dashboard sidebar"""
        st.sidebar.title("🎛️ Dashboard Controls")
        
        # Engine Status
        st.sidebar.subheader("Engine Status")
        if st.session_state.strategy_engine:
            if st.session_state.strategy_engine.is_running:
                st.sidebar.success("🟢 Engine Running")
            else:
                st.sidebar.error("🔴 Engine Stopped")
        else:
            st.sidebar.warning("⚪ Engine Not Connected")
        
        # Controls
        st.sidebar.subheader("Controls")
        auto_refresh = st.sidebar.checkbox("Auto Refresh (5s)", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_refresh
        
        if st.sidebar.button("🔄 Manual Refresh"):
            st.rerun()
        
        # Settings
        st.sidebar.subheader("Display Settings")
        show_signals = st.sidebar.checkbox("Show Recent Signals", value=True)
        show_trades = st.sidebar.checkbox("Show Trade History", value=True)
        show_risk_metrics = st.sidebar.checkbox("Show Risk Metrics", value=True)
        
        return {
            'show_signals': show_signals,
            'show_trades': show_trades,
            'show_risk_metrics': show_risk_metrics
        }
    
    def display_kpi_metrics(self, account_summary: Dict, performance_metrics: Dict):
        """Display key performance indicators"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Portfolio Value",
                value=f"${account_summary['portfolio_value']:,.2f}",
                delta=f"${account_summary['total_pnl']:+,.2f}"
            )
        
        with col2:
            st.metric(
                label="Total Return",
                value=f"{account_summary['total_return_pct']:+.2f}%",
                delta=f"{account_summary['total_return_pct']:+.2f}%"
            )
        
        with col3:
            st.metric(
                label="Cash Balance",
                value=f"${account_summary['cash_balance']:,.2f}",
                delta=None
            )
        
        with col4:
            positions_count = performance_metrics.get('current_positions', 0)
            st.metric(
                label="Active Positions",
                value=str(positions_count),
                delta=None
            )
    
    def display_portfolio_chart(self, portfolio_history: List[Dict]):
        """Display portfolio performance chart"""
        if not portfolio_history:
            st.warning("No portfolio history data available")
            return
        
        df = pd.DataFrame(portfolio_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = go.Figure()
        
        # Portfolio value line
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        
        # Add unrealized P&L as area
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['unrealized_pnl'],
            mode='lines',
            name='Unrealized P&L',
            fill='tonexty',
            line=dict(color='green', width=1),
            opacity=0.3
        ))
        
        fig.update_layout(
            title="Portfolio Performance Over Time",
            xaxis_title="Time",
            yaxis_title="Value ($)",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_positions_table(self, positions: Dict):
        """Display current positions table"""
        if not positions:
            st.info("No active positions")
            return
        
        positions_data = []
        for symbol, pos in positions.items():
            positions_data.append({
                'Symbol': symbol,
                'Quantity': pos['quantity'],
                'Avg Cost': f"${pos['avg_cost']:.2f}",
                'Market Value': f"${pos['market_value']:.2f}",
                'Unrealized P&L': f"${pos['unrealized_pnl']:+.2f}",
                'Return %': f"{(pos['unrealized_pnl'] / (pos['avg_cost'] * pos['quantity']) * 100):+.2f}%" if pos['avg_cost'] * pos['quantity'] != 0 else "0.00%"
            })
        
        df = pd.DataFrame(positions_data)
        
        # Style the dataframe
        def color_pnl(val):
            if '+' in val:
                return 'color: green'
            elif '-' in val:
                return 'color: red'
            return ''
        
        styled_df = df.style.applymap(color_pnl, subset=['Unrealized P&L', 'Return %'])
        st.dataframe(styled_df, use_container_width=True)
    
    def display_signals_chart(self, recent_signals: List[Dict]):
        """Display recent trading signals"""
        if not recent_signals:
            st.info("No recent signals")
            return
        
        df = pd.DataFrame(recent_signals)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = go.Figure()
        
        # Group signals by type and color
        signal_colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'blue'}
        
        for signal_type in df['signal'].unique():
            signal_data = df[df['signal'] == signal_type]
            
            fig.add_trace(go.Scatter(
                x=signal_data['timestamp'],
                y=signal_data['price'],
                mode='markers',
                name=f'{signal_type} Signals',
                marker=dict(
                    color=signal_colors.get(signal_type, 'blue'),
                    size=10,
                    symbol='triangle-up' if signal_type == 'BUY' else 'triangle-down'
                ),
                text=[f"{row['strategy']}<br>{row['symbol']}<br>Confidence: {row['confidence']:.2f}" 
                      for _, row in signal_data.iterrows()],
                hovertemplate='%{text}<br>Price: $%{y:.2f}<br>%{x}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Recent Trading Signals",
            xaxis_title="Time",
            yaxis_title="Price ($)",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_trades_table(self, trades: List[Dict]):
        """Display recent trades table"""
        if not trades:
            st.info("No recent trades")
            return
        
        trades_data = []
        for trade in trades[:20]:  # Show last 20 trades
            trades_data.append({
                'Time': pd.to_datetime(trade['timestamp']).strftime('%H:%M:%S'),
                'Symbol': trade['symbol'],
                'Side': trade['side'].upper(),
                'Quantity': trade['quantity'],
                'Price': f"${trade['price']:.2f}",
                'Value': f"${trade['quantity'] * trade['price']:.2f}",
                'Commission': f"${trade.get('commission', 0):.2f}"
            })
        
        df = pd.DataFrame(trades_data)
        
        # Style the dataframe
        def color_side(val):
            if val == 'BUY':
                return 'color: green'
            elif val == 'SELL':
                return 'color: red'
            return ''
        
        styled_df = df.style.applymap(color_side, subset=['Side'])
        st.dataframe(styled_df, use_container_width=True)
    
    def display_risk_metrics(self, performance_metrics: Dict, account_summary: Dict):
        """Display risk metrics"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Metrics")
            
            metrics_to_display = [
                ("Sharpe Ratio", f"{performance_metrics.get('sharpe_ratio', 0):.3f}"),
                ("Max Drawdown", f"{performance_metrics.get('max_drawdown_pct', 0):.2f}%"),
                ("Volatility", f"{performance_metrics.get('volatility', 0):.3f}"),
                ("Total Trades", str(performance_metrics.get('total_trades', 0))),
                ("Winning Trades", str(performance_metrics.get('winning_trades', 0)))
            ]
            
            for metric_name, metric_value in metrics_to_display:
                st.metric(metric_name, metric_value)
        
        with col2:
            st.subheader("Account Details")
            
            account_metrics = [
                ("Initial Balance", f"${account_summary['initial_balance']:,.2f}"),
                ("Cash Balance", f"${account_summary['cash_balance']:,.2f}"),
                ("Market Value", f"${account_summary['market_value']:,.2f}"),
                ("Unrealized P&L", f"${account_summary['unrealized_pnl']:+,.2f}"),
                ("Realized P&L", f"${account_summary['realized_pnl']:+,.2f}")
            ]
            
            for metric_name, metric_value in account_metrics:
                st.metric(metric_name, metric_value)
    
    def display_strategy_performance(self, strategies: Dict):
        """Display strategy performance table"""
        if not strategies:
            st.info("No active strategies")
            return
        
        strategy_data = []
        for name, info in strategies.items():
            strategy_data.append({
                'Strategy': name,
                'Status': '🟢 Active' if info['active'] else '🔴 Inactive',
                'Symbols': ', '.join(info['symbols']),
                'Signals Generated': info['signal_count'],
                'Trades Executed': info['trades'],
                'Active Positions': len([p for p in info['positions'].values() if p != 0])
            })
        
        df = pd.DataFrame(strategy_data)
        st.dataframe(df, use_container_width=True)
    
    def display_performance_analytics(self, portfolio_history: List[Dict]):
        """Display advanced performance analytics"""
        if len(portfolio_history) < 10:
            st.info("Insufficient data for advanced analytics")
            return
        
        df = pd.DataFrame(portfolio_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['returns'] = df['portfolio_value'].pct_change()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Returns histogram
            fig_hist = px.histogram(
                df.dropna(), 
                x='returns', 
                nbins=30, 
                title="Returns Distribution"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Rolling volatility
            rolling_vol = df['returns'].rolling(window=20).std() * np.sqrt(252 * 24 * 60)
            
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=df['timestamp'][20:],
                y=rolling_vol[20:],
                mode='lines',
                name='Rolling Volatility',
                line=dict(color='orange')
            ))
            fig_vol.update_layout(
                title="Rolling Volatility (20-period)",
                xaxis_title="Time",
                yaxis_title="Volatility"
            )
            st.plotly_chart(fig_vol, use_container_width=True)
    
    def run_dashboard(self, strategy_engine):
        """Run the Streamlit dashboard"""
        st.session_state.strategy_engine = strategy_engine
        
        # Header
        st.title("📈 Live Algorithmic Trading Dashboard")
        st.markdown("---")
        
        # Setup sidebar
        settings = self.setup_sidebar()
        
        # Auto-refresh logic
        if st.session_state.auto_refresh:
            placeholder = st.empty()
            time.sleep(5)
            st.rerun()
        
        try:
            # Get current data
            if strategy_engine and strategy_engine.is_running:
                status = strategy_engine.get_engine_status()
                performance_data = strategy_engine.paper_account.export_performance_data()
                
                account_summary = status['account_summary']
                performance_metrics = performance_data['performance_metrics']
                positions = performance_data['positions']
                trades = performance_data['trades']
                portfolio_history = performance_data['portfolio_history']
                recent_signals = status['recent_signals']
                strategies = status['strategies']
                
                # Display KPI metrics
                self.display_kpi_metrics(account_summary, performance_metrics)
                st.markdown("---")
                
                # Main content tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Portfolio", "📍 Positions", "🎯 Signals", "📈 Analytics", "⚙️ Strategies"
                ])
                
                with tab1:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("Portfolio Performance")
                        self.display_portfolio_chart(portfolio_history)
                    
                    with col2:
                        if settings['show_risk_metrics']:
                            self.display_risk_metrics(performance_metrics, account_summary)
                
                with tab2:
                    st.subheader("Current Positions")
                    self.display_positions_table(positions)
                    
                    if settings['show_trades']:
                        st.subheader("Recent Trades")
                        self.display_trades_table(trades)
                
                with tab3:
                    if settings['show_signals']:
                        st.subheader("Recent Trading Signals")
                        self.display_signals_chart(recent_signals)
                        
                        # Signals table
                        if recent_signals:
                            st.subheader("Signal Details")
                            signals_df = pd.DataFrame(recent_signals)
                            signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp']).dt.strftime('%H:%M:%S')
                            st.dataframe(signals_df, use_container_width=True)
                
                with tab4:
                    st.subheader("Advanced Analytics")
                    self.display_performance_analytics(portfolio_history)
                
                with tab5:
                    st.subheader("Strategy Performance")
                    self.display_strategy_performance(strategies)
                
                # Status footer
                st.markdown("---")
                last_update = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                st.caption(f"Last updated: {last_update}")
                
            else:
                st.error("Strategy engine is not running or not connected")
                st.info("Please ensure the live trading system is active before using the dashboard")
        
        except Exception as e:
            st.error(f"Error loading dashboard data: {str(e)}")
            st.info("Please check that the trading system is running properly")

def main():
    """Main function to run Streamlit dashboard"""
    dashboard = StreamlitTradingDashboard()
    
    # This is a placeholder - in real usage, you would pass your actual strategy engine
    st.warning("This is a standalone dashboard view. Connect your LiveStrategyEngine to see live data.")
    
    # Demo data for display purposes
    demo_account_summary = {
        'portfolio_value': 105000.00,
        'total_pnl': 5000.00,
        'total_return_pct': 5.00,
        'cash_balance': 45000.00
    }
    
    demo_performance_metrics = {
        'current_positions': 3,
        'sharpe_ratio': 1.2,
        'max_drawdown_pct': 2.5,
        'volatility': 0.15,
        'total_trades': 25,
        'winning_trades': 15
    }
    
    dashboard.display_kpi_metrics(demo_account_summary, demo_performance_metrics)

if __name__ == "__main__":
    main()