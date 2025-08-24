import dash
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import json
from typing import Dict, List, Optional
import queue

from ..real_time.live_strategy_engine import LiveStrategyEngine
from ..real_time.paper_trading import PaperTradingAccount
from ..real_time.live_data_feed import LiveDataFeed

class LiveTradingDashboard:
    """
    Real-time trading dashboard with live updates
    """
    
    def __init__(self, strategy_engine: LiveStrategyEngine):
        self.strategy_engine = strategy_engine
        self.app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
        self.data_queue = queue.Queue()
        
        # Dashboard state
        self.last_update = datetime.now()
        self.performance_history = []
        self.signal_history = []
        self.position_history = []
        
        # Setup real-time data collection
        self.setup_data_collection()
        
        # Build layout and callbacks
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_data_collection(self):
        """Setup real-time data collection from strategy engine"""
        def data_collector():
            while True:
                try:
                    if self.strategy_engine.is_running:
                        # Get current engine status
                        status = self.strategy_engine.get_engine_status()
                        performance_data = self.strategy_engine.paper_account.export_performance_data()
                        
                        # Queue the data for dashboard update
                        self.data_queue.put({
                            'timestamp': datetime.now(),
                            'engine_status': status,
                            'performance_data': performance_data
                        })
                    
                    time.sleep(1)  # Update every second
                    
                except Exception as e:
                    print(f"Error in data collector: {e}")
                    time.sleep(5)
        
        # Start data collection thread
        self.data_thread = threading.Thread(target=data_collector, daemon=True)
        self.data_thread.start()
    
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Live Algorithmic Trading Dashboard", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
                
                # Status indicators
                html.Div([
                    html.Div([
                        html.H4("Engine Status", style={'textAlign': 'center'}),
                        html.Div(id='engine-status', style={'textAlign': 'center', 'fontSize': 18})
                    ], className='three columns'),
                    
                    html.Div([
                        html.H4("Portfolio Value", style={'textAlign': 'center'}),
                        html.Div(id='portfolio-value', style={'textAlign': 'center', 'fontSize': 18})
                    ], className='three columns'),
                    
                    html.Div([
                        html.H4("Total Return", style={'textAlign': 'center'}),
                        html.Div(id='total-return', style={'textAlign': 'center', 'fontSize': 18})
                    ], className='three columns'),
                    
                    html.Div([
                        html.H4("Active Positions", style={'textAlign': 'center'}),
                        html.Div(id='active-positions', style={'textAlign': 'center', 'fontSize': 18})
                    ], className='three columns'),
                ], className='row', style={'marginBottom': 30}),
            ], style={'backgroundColor': '#ecf0f1', 'padding': 20, 'margin': 10}),
            
            # Main content
            html.Div([
                # Performance Charts
                html.Div([
                    html.H3("Portfolio Performance"),
                    dcc.Graph(id='portfolio-chart'),
                ], className='six columns'),
                
                html.Div([
                    html.H3("Current Positions"),
                    dash_table.DataTable(
                        id='positions-table',
                        columns=[
                            {'name': 'Symbol', 'id': 'symbol'},
                            {'name': 'Quantity', 'id': 'quantity'},
                            {'name': 'Avg Cost', 'id': 'avg_cost', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                            {'name': 'Market Value', 'id': 'market_value', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                            {'name': 'P&L', 'id': 'unrealized_pnl', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        ],
                        style_cell={'textAlign': 'left'},
                        style_data_conditional=[
                            {
                                'if': {'filter_query': '{unrealized_pnl} > 0'},
                                'backgroundColor': '#d5f4e6',
                                'color': 'black',
                            },
                            {
                                'if': {'filter_query': '{unrealized_pnl} < 0'},
                                'backgroundColor': '#ffeaa7',
                                'color': 'black',
                            }
                        ]
                    )
                ], className='six columns'),
            ], className='row'),
            
            # Strategy Performance
            html.Div([
                html.Div([
                    html.H3("Strategy Signals"),
                    dcc.Graph(id='signals-chart'),
                ], className='six columns'),
                
                html.Div([
                    html.H3("Recent Trades"),
                    dash_table.DataTable(
                        id='trades-table',
                        columns=[
                            {'name': 'Time', 'id': 'timestamp'},
                            {'name': 'Symbol', 'id': 'symbol'},
                            {'name': 'Side', 'id': 'side'},
                            {'name': 'Quantity', 'id': 'quantity'},
                            {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        ],
                        style_cell={'textAlign': 'left'},
                        page_size=10
                    )
                ], className='six columns'),
            ], className='row', style={'marginTop': 30}),
            
            # Risk Metrics
            html.Div([
                html.Div([
                    html.H3("Risk Metrics"),
                    html.Div(id='risk-metrics'),
                ], className='six columns'),
                
                html.Div([
                    html.H3("Strategy Performance"),
                    dash_table.DataTable(
                        id='strategy-table',
                        columns=[
                            {'name': 'Strategy', 'id': 'strategy'},
                            {'name': 'Status', 'id': 'status'},
                            {'name': 'Signals', 'id': 'signals'},
                            {'name': 'Trades', 'id': 'trades'},
                            {'name': 'Positions', 'id': 'positions'},
                        ],
                        style_cell={'textAlign': 'left'}
                    )
                ], className='six columns'),
            ], className='row', style={'marginTop': 30}),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=2000,  # Update every 2 seconds
                n_intervals=0
            )
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('engine-status', 'children'),
             Output('engine-status', 'style'),
             Output('portfolio-value', 'children'),
             Output('total-return', 'children'),
             Output('total-return', 'style'),
             Output('active-positions', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_status_indicators(n):
            try:
                # Get latest data
                status = self.strategy_engine.get_engine_status()
                account_summary = status['account_summary']
                
                # Engine status
                if status['is_running']:
                    engine_status_text = "🟢 RUNNING"
                    engine_status_style = {'textAlign': 'center', 'fontSize': 18, 'color': 'green'}
                else:
                    engine_status_text = "🔴 STOPPED"
                    engine_status_style = {'textAlign': 'center', 'fontSize': 18, 'color': 'red'}
                
                # Portfolio value
                portfolio_value = f"${account_summary['portfolio_value']:,.2f}"
                
                # Total return
                return_pct = account_summary['total_return_pct']
                return_text = f"{return_pct:+.2f}%"
                return_style = {
                    'textAlign': 'center', 
                    'fontSize': 18,
                    'color': 'green' if return_pct >= 0 else 'red'
                }
                
                # Active positions
                positions_count = len([s for s in status['strategies'].values() 
                                     for p in s['positions'].values() if p != 0])
                
                return (engine_status_text, engine_status_style, portfolio_value, 
                       return_text, return_style, str(positions_count))
                
            except Exception as e:
                return ("Error", {}, "N/A", "N/A", {}, "N/A")
        
        @self.app.callback(
            Output('portfolio-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_portfolio_chart(n):
            try:
                performance_data = self.strategy_engine.paper_account.export_performance_data()
                portfolio_history = performance_data['portfolio_history']
                
                if not portfolio_history:
                    return {'data': [], 'layout': {'title': 'No Data Available'}}
                
                df = pd.DataFrame(portfolio_history)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['portfolio_value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                ))
                
                fig.update_layout(
                    title='Portfolio Value Over Time',
                    xaxis_title='Time',
                    yaxis_title='Portfolio Value ($)',
                    height=400
                )
                
                return fig
                
            except Exception as e:
                return {'data': [], 'layout': {'title': f'Error: {str(e)}'}}
        
        @self.app.callback(
            Output('positions-table', 'data'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_positions_table(n):
            try:
                positions = self.strategy_engine.paper_account.get_positions()
                
                if not positions:
                    return []
                
                positions_data = []
                for symbol, pos in positions.items():
                    positions_data.append({
                        'symbol': symbol,
                        'quantity': pos['quantity'],
                        'avg_cost': pos['avg_cost'],
                        'market_value': pos['market_value'],
                        'unrealized_pnl': pos['unrealized_pnl']
                    })
                
                return positions_data
                
            except Exception as e:
                return []
        
        @self.app.callback(
            Output('signals-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_signals_chart(n):
            try:
                status = self.strategy_engine.get_engine_status()
                recent_signals = status['recent_signals']
                
                if not recent_signals:
                    return {'data': [], 'layout': {'title': 'No Recent Signals'}}
                
                df = pd.DataFrame(recent_signals)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Create scatter plot for signals
                fig = go.Figure()
                
                # Group by signal type
                for signal_type in df['signal'].unique():
                    signal_data = df[df['signal'] == signal_type]
                    color = 'green' if signal_type == 'BUY' else 'red' if signal_type == 'SELL' else 'blue'
                    
                    fig.add_trace(go.Scatter(
                        x=signal_data['timestamp'],
                        y=signal_data['price'],
                        mode='markers',
                        name=signal_type,
                        marker=dict(color=color, size=8),
                        text=signal_data['strategy'] + ' - ' + signal_data['symbol'],
                        hovertemplate='%{text}<br>Price: $%{y:.2f}<br>Time: %{x}<extra></extra>'
                    ))
                
                fig.update_layout(
                    title='Recent Trading Signals',
                    xaxis_title='Time',
                    yaxis_title='Price ($)',
                    height=400
                )
                
                return fig
                
            except Exception as e:
                return {'data': [], 'layout': {'title': f'Error: {str(e)}'}}
        
        @self.app.callback(
            Output('trades-table', 'data'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_trades_table(n):
            try:
                trades = self.strategy_engine.paper_account.get_trades(limit=20)
                
                if not trades:
                    return []
                
                trades_data = []
                for trade in trades:
                    trades_data.append({
                        'timestamp': pd.to_datetime(trade['timestamp']).strftime('%H:%M:%S'),
                        'symbol': trade['symbol'],
                        'side': trade['side'].upper(),
                        'quantity': trade['quantity'],
                        'price': trade['price']
                    })
                
                return trades_data
                
            except Exception as e:
                return []
        
        @self.app.callback(
            Output('risk-metrics', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_risk_metrics(n):
            try:
                performance_data = self.strategy_engine.paper_account.export_performance_data()
                metrics = performance_data['performance_metrics']
                account_summary = performance_data['account_summary']
                
                return html.Div([
                    html.P(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}"),
                    html.P(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%"),
                    html.P(f"Volatility: {metrics.get('volatility', 0):.3f}"),
                    html.P(f"Cash Balance: ${account_summary['cash_balance']:,.2f}"),
                    html.P(f"Buying Power: ${account_summary['buying_power']:,.2f}"),
                ])
                
            except Exception as e:
                return html.Div([html.P(f"Error loading metrics: {str(e)}")])
        
        @self.app.callback(
            Output('strategy-table', 'data'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_strategy_table(n):
            try:
                status = self.strategy_engine.get_engine_status()
                strategies = status['strategies']
                
                strategy_data = []
                for name, info in strategies.items():
                    strategy_data.append({
                        'strategy': name,
                        'status': 'Active' if info['active'] else 'Inactive',
                        'signals': info['signal_count'],
                        'trades': info['trades'],
                        'positions': len([p for p in info['positions'].values() if p != 0])
                    })
                
                return strategy_data
                
            except Exception as e:
                return []
    
    def run_server(self, debug=False, port=8050, host='127.0.0.1'):
        """Run the dashboard server"""
        print(f"Starting Live Trading Dashboard at http://{host}:{port}")
        self.app.run(debug=debug, port=port, host=host)

class PortfolioAnalyticsDashboard:
    """
    Advanced portfolio analytics dashboard
    """
    
    def __init__(self, strategy_engine: LiveStrategyEngine):
        self.strategy_engine = strategy_engine
        self.app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
        
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup analytics dashboard layout"""
        self.app.layout = html.Div([
            html.H1("Portfolio Analytics Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': 30}),
            
            # Controls
            html.Div([
                html.Div([
                    html.Label('Analysis Period:'),
                    dcc.Dropdown(
                        id='period-dropdown',
                        options=[
                            {'label': 'Last Hour', 'value': '1H'},
                            {'label': 'Last Day', 'value': '1D'},
                            {'label': 'Last Week', 'value': '1W'},
                            {'label': 'All Time', 'value': 'ALL'}
                        ],
                        value='1D'
                    )
                ], className='six columns'),
                
                html.Div([
                    html.Label('Strategy Filter:'),
                    dcc.Dropdown(
                        id='strategy-filter',
                        multi=True,
                        placeholder='Select strategies to analyze'
                    )
                ], className='six columns'),
            ], className='row', style={'marginBottom': 30}),
            
            # Performance Analysis
            html.Div([
                html.Div([
                    html.H3("Returns Distribution"),
                    dcc.Graph(id='returns-histogram'),
                ], className='six columns'),
                
                html.Div([
                    html.H3("Rolling Sharpe Ratio"),
                    dcc.Graph(id='rolling-sharpe'),
                ], className='six columns'),
            ], className='row'),
            
            # Risk Analysis
            html.Div([
                html.Div([
                    html.H3("Drawdown Analysis"),
                    dcc.Graph(id='drawdown-chart'),
                ], className='six columns'),
                
                html.Div([
                    html.H3("Position Correlation Matrix"),
                    dcc.Graph(id='correlation-matrix'),
                ], className='six columns'),
            ], className='row', style={'marginTop': 30}),
            
            # Strategy Comparison
            html.Div([
                html.H3("Strategy Performance Comparison"),
                dcc.Graph(id='strategy-comparison'),
            ], style={'marginTop': 30}),
            
            # Auto-refresh
            dcc.Interval(
                id='analytics-interval',
                interval=10000,  # Update every 10 seconds
                n_intervals=0
            )
        ])
    
    def setup_callbacks(self):
        """Setup analytics dashboard callbacks"""
        
        @self.app.callback(
            Output('strategy-filter', 'options'),
            [Input('analytics-interval', 'n_intervals')]
        )
        def update_strategy_options(n):
            try:
                status = self.strategy_engine.get_engine_status()
                strategies = status['strategies']
                
                return [{'label': name, 'value': name} for name in strategies.keys()]
                
            except Exception as e:
                return []
        
        @self.app.callback(
            Output('returns-histogram', 'figure'),
            [Input('analytics-interval', 'n_intervals'),
             Input('period-dropdown', 'value')]
        )
        def update_returns_histogram(n, period):
            try:
                performance_data = self.strategy_engine.paper_account.export_performance_data()
                portfolio_history = performance_data['portfolio_history']
                
                if not portfolio_history or len(portfolio_history) < 2:
                    return {'data': [], 'layout': {'title': 'Insufficient Data'}}
                
                df = pd.DataFrame(portfolio_history)
                df['returns'] = df['portfolio_value'].pct_change().dropna()
                
                fig = px.histogram(df, x='returns', nbins=50, title='Portfolio Returns Distribution')
                fig.update_layout(xaxis_title='Returns', yaxis_title='Frequency')
                
                return fig
                
            except Exception as e:
                return {'data': [], 'layout': {'title': f'Error: {str(e)}'}}
    
    def run_server(self, debug=False, port=8051, host='127.0.0.1'):
        """Run the analytics dashboard server"""
        print(f"Starting Portfolio Analytics Dashboard at http://{host}:{port}")
        self.app.run(debug=debug, port=port, host=host)