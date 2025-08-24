import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TradingVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = {
            'buy': '#2E8B57',
            'sell': '#DC143C',
            'portfolio': '#1f77b4',
            'benchmark': '#ff7f0e',
            'profit': '#2ca02c',
            'loss': '#d62728'
        }
    
    def plot_price_and_signals(self, 
                              data: pd.DataFrame, 
                              signals: pd.DataFrame,
                              symbol: str,
                              indicators: Optional[List[str]] = None,
                              save_path: Optional[str] = None) -> go.Figure:
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=[f'{symbol} Price and Signals', 'Volume', 'Indicators'],
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Price chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing_line_color=self.colors['profit'],
                decreasing_line_color=self.colors['loss']
            ),
            row=1, col=1
        )
        
        # Buy signals
        buy_signals = signals[signals['Signal'] == 1]
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['Price'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=10,
                        color=self.colors['buy']
                    ),
                    name='Buy Signal'
                ),
                row=1, col=1
            )
        
        # Sell signals
        sell_signals = signals[signals['Signal'] == -1]
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['Price'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=10,
                        color=self.colors['sell']
                    ),
                    name='Sell Signal'
                ),
                row=1, col=1
            )
        
        # Volume
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'] if 'Volume' in data.columns else [],
                name='Volume',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Indicators
        if indicators:
            for indicator in indicators:
                if indicator in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data[indicator],
                            mode='lines',
                            name=indicator,
                            line=dict(width=1)
                        ),
                        row=3, col=1
                    )
        
        fig.update_layout(
            title=f'{symbol} Trading Analysis',
            xaxis_title='Date',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_xaxes(rangeslider_visible=False)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_portfolio_performance(self, 
                                 portfolio_values: pd.Series,
                                 benchmark: Optional[pd.Series] = None,
                                 trades: Optional[pd.DataFrame] = None,
                                 save_path: Optional[str] = None) -> go.Figure:
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=['Portfolio Value Over Time', 'Drawdown'],
            row_heights=[0.7, 0.3]
        )
        
        # Portfolio performance
        portfolio_pct = (portfolio_values / portfolio_values.iloc[0] - 1) * 100
        
        fig.add_trace(
            go.Scatter(
                x=portfolio_values.index,
                y=portfolio_pct,
                mode='lines',
                name='Portfolio',
                line=dict(color=self.colors['portfolio'], width=2)
            ),
            row=1, col=1
        )
        
        # Benchmark comparison
        if benchmark is not None:
            benchmark_pct = (benchmark / benchmark.iloc[0] - 1) * 100
            fig.add_trace(
                go.Scatter(
                    x=benchmark.index,
                    y=benchmark_pct,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color=self.colors['benchmark'], width=2)
                ),
                row=1, col=1
            )
        
        # Trade markers
        if trades is not None and not trades.empty:
            # Winning trades
            winning_trades = trades[trades['PnL'] > 0]
            if not winning_trades.empty:
                fig.add_trace(
                    go.Scatter(
                        x=winning_trades['Exit_Date'],
                        y=[portfolio_pct.loc[date] if date in portfolio_pct.index else 0 
                           for date in winning_trades['Exit_Date']],
                        mode='markers',
                        marker=dict(
                            symbol='circle',
                            size=6,
                            color=self.colors['profit']
                        ),
                        name='Winning Trades'
                    ),
                    row=1, col=1
                )
            
            # Losing trades
            losing_trades = trades[trades['PnL'] < 0]
            if not losing_trades.empty:
                fig.add_trace(
                    go.Scatter(
                        x=losing_trades['Exit_Date'],
                        y=[portfolio_pct.loc[date] if date in portfolio_pct.index else 0 
                           for date in losing_trades['Exit_Date']],
                        mode='markers',
                        marker=dict(
                            symbol='x',
                            size=6,
                            color=self.colors['loss']
                        ),
                        name='Losing Trades'
                    ),
                    row=1, col=1
                )
        
        # Drawdown
        cumulative = portfolio_values / portfolio_values.expanding().max()
        drawdown = (1 - cumulative) * 100
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown,
                mode='lines',
                fill='tonexty',
                name='Drawdown',
                line=dict(color='red', width=1),
                fillcolor='rgba(255, 0, 0, 0.3)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Portfolio Performance Analysis',
            height=600,
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text='Return (%)', row=1, col=1)
        fig.update_yaxes(title_text='Drawdown (%)', row=2, col=1)
        fig.update_xaxes(title_text='Date', row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_risk_metrics(self, 
                         risk_metrics: Dict[str, float],
                         save_path: Optional[str] = None) -> go.Figure:
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Risk Metrics', 'Performance Metrics', 'Trade Statistics', 'Risk Distribution']
        )
        
        # Risk metrics radar chart
        risk_categories = ['VaR 95%', 'Volatility', 'Max Drawdown', 'Beta']
        risk_values = [
            min(risk_metrics.get('var_95', 0) * 100, 100),
            min(risk_metrics.get('volatility', 0) * 100, 100),
            min(risk_metrics.get('max_drawdown', 0) * 100, 100),
            min(abs(risk_metrics.get('beta', 1)) * 50, 100)
        ]
        
        fig.add_trace(
            go.Scatterpolar(
                r=risk_values,
                theta=risk_categories,
                fill='toself',
                name='Risk Profile',
                line_color='red'
            ),
            row=1, col=1
        )
        
        # Performance metrics
        perf_metrics = ['Sharpe Ratio', 'Total Return', 'Win Rate']
        perf_values = [
            max(risk_metrics.get('sharpe_ratio', 0) * 20, 0),
            max(risk_metrics.get('total_return_pct', 0), 0),
            risk_metrics.get('win_rate_pct', 50)
        ]
        
        fig.add_trace(
            go.Bar(
                x=perf_metrics,
                y=perf_values,
                marker_color=[self.colors['profit'] if v > 0 else self.colors['loss'] for v in perf_values],
                name='Performance'
            ),
            row=1, col=2
        )
        
        # Trade statistics
        trade_stats = ['Total Trades', 'Winning Trades', 'Losing Trades']
        trade_values = [
            risk_metrics.get('total_trades', 0),
            risk_metrics.get('winning_trades', 0),
            risk_metrics.get('losing_trades', 0)
        ]
        
        fig.add_trace(
            go.Bar(
                x=trade_stats,
                y=trade_values,
                marker_color=['blue', self.colors['profit'], self.colors['loss']],
                name='Trades'
            ),
            row=2, col=1
        )
        
        # Risk distribution (hypothetical)
        returns_dist = np.random.normal(0, risk_metrics.get('volatility', 0.15) / np.sqrt(252), 1000)
        
        fig.add_trace(
            go.Histogram(
                x=returns_dist,
                nbinsx=30,
                name='Return Distribution',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Risk Analysis Dashboard',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_strategy_comparison(self, 
                               results: Dict[str, Dict],
                               save_path: Optional[str] = None) -> go.Figure:
        
        strategies = list(results.keys())
        metrics = ['Total_Return_Pct', 'Sharpe_Ratio', 'Max_Drawdown_Pct', 'Win_Rate_Pct']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)']
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, metric in enumerate(metrics):
            values = [results[strategy]['metrics'].get(metric, 0) for strategy in strategies]
            colors = [self.colors['profit'] if v > 0 else self.colors['loss'] 
                     if metric != 'Max_Drawdown_Pct' else self.colors['loss'] for v in values]
            
            fig.add_trace(
                go.Bar(
                    x=strategies,
                    y=values,
                    marker_color=colors,
                    name=metric,
                    showlegend=False
                ),
                row=positions[i][0], col=positions[i][1]
            )
        
        fig.update_layout(
            title='Strategy Performance Comparison',
            height=600,
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_correlation_heatmap(self, 
                               correlation_matrix: pd.DataFrame,
                               save_path: Optional[str] = None) -> go.Figure:
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdYlBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={'size': 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Asset Correlation Matrix',
            xaxis_title='Assets',
            yaxis_title='Assets',
            height=600,
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_performance_report(self, 
                                results: Dict,
                                save_path: str = 'performance_report.html'):
        
        # Create comprehensive HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Strategy Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .metrics-table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .metrics-table th {{ background-color: #f2f2f2; }}
                .positive {{ color: #2ca02c; }}
                .negative {{ color: #d62728; }}
                .section {{ margin-bottom: 40px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Algorithmic Trading Strategy Performance Report</h1>
                <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Performance Summary</h2>
                <table class="metrics-table">
                    <tr><th>Metric</th><th>Value</th></tr>
        """
        
        metrics = results.get('metrics', {})
        for metric, value in metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
                css_class = "positive" if value > 0 and "Loss" not in metric and "Drawdown" not in metric else "negative" if value < 0 or "Loss" in metric or "Drawdown" in metric else ""
            else:
                formatted_value = str(value)
                css_class = ""
            
            html_content += f'<tr><td>{metric.replace("_", " ").title()}</td><td class="{css_class}">{formatted_value}</td></tr>'
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Trade Analysis</h2>
                <p>Detailed trade statistics and performance metrics are displayed in the charts above.</p>
            </div>
            
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        print(f"Performance report saved to {save_path}")
        
        return save_path