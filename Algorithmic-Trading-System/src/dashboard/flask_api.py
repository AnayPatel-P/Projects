from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import json
from datetime import datetime
from typing import Dict, List, Optional
import threading
import time

from ..real_time.live_strategy_engine import LiveStrategyEngine

class TradingSystemAPI:
    """
    Flask API for the trading system with WebSocket support
    """
    
    def __init__(self, strategy_engine: LiveStrategyEngine):
        self.strategy_engine = strategy_engine
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/')
        def dashboard():
            """Serve basic web dashboard"""
            return render_template_string(self.get_html_template())
        
        @self.app.route('/api/status')
        def get_status():
            """Get current system status"""
            try:
                status = self.strategy_engine.get_engine_status()
                return jsonify({
                    'success': True,
                    'data': status,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/account')
        def get_account():
            """Get account summary"""
            try:
                account_summary = self.strategy_engine.paper_account.get_account_summary()
                return jsonify({
                    'success': True,
                    'data': account_summary,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/positions')
        def get_positions():
            """Get current positions"""
            try:
                positions = self.strategy_engine.paper_account.get_positions()
                return jsonify({
                    'success': True,
                    'data': positions,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/trades')
        def get_trades():
            """Get trade history"""
            try:
                limit = request.args.get('limit', 50, type=int)
                trades = self.strategy_engine.paper_account.get_trades(limit=limit)
                return jsonify({
                    'success': True,
                    'data': trades,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/signals')
        def get_signals():
            """Get recent signals"""
            try:
                status = self.strategy_engine.get_engine_status()
                signals = status.get('recent_signals', [])
                return jsonify({
                    'success': True,
                    'data': signals,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/performance')
        def get_performance():
            """Get comprehensive performance data"""
            try:
                performance_data = self.strategy_engine.export_performance_data()
                return jsonify({
                    'success': True,
                    'data': performance_data,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/strategies')
        def get_strategies():
            """Get strategy information"""
            try:
                status = self.strategy_engine.get_engine_status()
                strategies = status.get('strategies', {})
                return jsonify({
                    'success': True,
                    'data': strategies,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/strategies/<strategy_name>/activate', methods=['POST'])
        def activate_strategy(strategy_name):
            """Activate a strategy"""
            try:
                self.strategy_engine.activate_strategy(strategy_name)
                return jsonify({
                    'success': True,
                    'message': f'Strategy {strategy_name} activated',
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/strategies/<strategy_name>/deactivate', methods=['POST'])
        def deactivate_strategy(strategy_name):
            """Deactivate a strategy"""
            try:
                self.strategy_engine.deactivate_strategy(strategy_name)
                return jsonify({
                    'success': True,
                    'message': f'Strategy {strategy_name} deactivated',
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/engine/start', methods=['POST'])
        def start_engine():
            """Start the trading engine"""
            try:
                data = request.get_json() or {}
                symbols = data.get('symbols', ['AAPL', 'GOOGL', 'MSFT'])
                
                if not self.strategy_engine.is_running:
                    self.strategy_engine.start_engine(symbols)
                
                return jsonify({
                    'success': True,
                    'message': 'Engine started',
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/engine/stop', methods=['POST'])
        def stop_engine():
            """Stop the trading engine"""
            try:
                self.strategy_engine.stop_engine()
                return jsonify({
                    'success': True,
                    'message': 'Engine stopped',
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/health')
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'engine_running': self.strategy_engine.is_running if self.strategy_engine else False
            })
    
    def get_html_template(self):
        """Get HTML template for web dashboard"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Live Trading Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .metrics {
            display: flex;
            justify-content: space-around;
            margin-bottom: 20px;
        }
        .metric {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            min-width: 200px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric h3 {
            margin: 0;
            color: #2c3e50;
        }
        .metric .value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .positive { color: green; }
        .negative { color: red; }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .table-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        .status-running { color: green; font-weight: bold; }
        .status-stopped { color: red; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Live Algorithmic Trading Dashboard</h1>
        <p>Real-time monitoring and control system</p>
    </div>
    
    <div class="metrics" id="metrics">
        <!-- Metrics will be populated by JavaScript -->
    </div>
    
    <div class="chart-container">
        <h2>Portfolio Performance</h2>
        <div id="portfolio-chart"></div>
    </div>
    
    <div class="table-container">
        <h2>Current Positions</h2>
        <div id="positions-table"></div>
    </div>
    
    <div class="table-container">
        <h2>Recent Trades</h2>
        <div id="trades-table"></div>
    </div>
    
    <div class="table-container">
        <h2>Strategy Status</h2>
        <div id="strategies-table"></div>
    </div>

    <script>
        // Dashboard JavaScript
        let refreshInterval;
        
        async function fetchData(endpoint) {
            try {
                const response = await fetch(`/api/${endpoint}`);
                const data = await response.json();
                return data.success ? data.data : null;
            } catch (error) {
                console.error(`Error fetching ${endpoint}:`, error);
                return null;
            }
        }
        
        function formatCurrency(value) {
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD'
            }).format(value);
        }
        
        function formatPercentage(value) {
            return value.toFixed(2) + '%';
        }
        
        async function updateMetrics() {
            const account = await fetchData('account');
            const status = await fetchData('status');
            
            if (!account || !status) return;
            
            const metricsHtml = `
                <div class="metric">
                    <h3>Engine Status</h3>
                    <div class="value ${status.is_running ? 'status-running' : 'status-stopped'}">
                        ${status.is_running ? '🟢 RUNNING' : '🔴 STOPPED'}
                    </div>
                </div>
                <div class="metric">
                    <h3>Portfolio Value</h3>
                    <div class="value">${formatCurrency(account.portfolio_value)}</div>
                </div>
                <div class="metric">
                    <h3>Total Return</h3>
                    <div class="value ${account.total_return_pct >= 0 ? 'positive' : 'negative'}">
                        ${account.total_return_pct >= 0 ? '+' : ''}${formatPercentage(account.total_return_pct)}
                    </div>
                </div>
                <div class="metric">
                    <h3>Cash Balance</h3>
                    <div class="value">${formatCurrency(account.cash_balance)}</div>
                </div>
            `;
            
            document.getElementById('metrics').innerHTML = metricsHtml;
        }
        
        async function updatePortfolioChart() {
            const performance = await fetchData('performance');
            
            if (!performance || !performance.account_data.portfolio_history) return;
            
            const history = performance.account_data.portfolio_history;
            
            const trace = {
                x: history.map(h => h.timestamp),
                y: history.map(h => h.portfolio_value),
                type: 'scatter',
                mode: 'lines',
                name: 'Portfolio Value',
                line: { color: '#3498db', width: 2 }
            };
            
            const layout = {
                title: 'Portfolio Value Over Time',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Value ($)' },
                height: 400
            };
            
            Plotly.newPlot('portfolio-chart', [trace], layout);
        }
        
        async function updatePositionsTable() {
            const positions = await fetchData('positions');
            
            if (!positions) return;
            
            let html = '<table><thead><tr><th>Symbol</th><th>Quantity</th><th>Avg Cost</th><th>Market Value</th><th>P&L</th></tr></thead><tbody>';
            
            Object.entries(positions).forEach(([symbol, pos]) => {
                const pnlClass = pos.unrealized_pnl >= 0 ? 'positive' : 'negative';
                html += `
                    <tr>
                        <td>${symbol}</td>
                        <td>${pos.quantity}</td>
                        <td>${formatCurrency(pos.avg_cost)}</td>
                        <td>${formatCurrency(pos.market_value)}</td>
                        <td class="${pnlClass}">${formatCurrency(pos.unrealized_pnl)}</td>
                    </tr>
                `;
            });
            
            html += '</tbody></table>';
            document.getElementById('positions-table').innerHTML = html;
        }
        
        async function updateTradesTable() {
            const trades = await fetchData('trades?limit=10');
            
            if (!trades) return;
            
            let html = '<table><thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Quantity</th><th>Price</th></tr></thead><tbody>';
            
            trades.forEach(trade => {
                const time = new Date(trade.timestamp).toLocaleTimeString();
                html += `
                    <tr>
                        <td>${time}</td>
                        <td>${trade.symbol}</td>
                        <td>${trade.side.toUpperCase()}</td>
                        <td>${trade.quantity}</td>
                        <td>${formatCurrency(trade.price)}</td>
                    </tr>
                `;
            });
            
            html += '</tbody></table>';
            document.getElementById('trades-table').innerHTML = html;
        }
        
        async function updateStrategiesTable() {
            const strategies = await fetchData('strategies');
            
            if (!strategies) return;
            
            let html = '<table><thead><tr><th>Strategy</th><th>Status</th><th>Signals</th><th>Trades</th></tr></thead><tbody>';
            
            Object.entries(strategies).forEach(([name, info]) => {
                const statusClass = info.active ? 'status-running' : 'status-stopped';
                const statusText = info.active ? 'Active' : 'Inactive';
                
                html += `
                    <tr>
                        <td>${name}</td>
                        <td class="${statusClass}">${statusText}</td>
                        <td>${info.signals}</td>
                        <td>${info.trades}</td>
                    </tr>
                `;
            });
            
            html += '</tbody></table>';
            document.getElementById('strategies-table').innerHTML = html;
        }
        
        async function updateDashboard() {
            await Promise.all([
                updateMetrics(),
                updatePortfolioChart(),
                updatePositionsTable(),
                updateTradesTable(),
                updateStrategiesTable()
            ]);
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            updateDashboard();
            refreshInterval = setInterval(updateDashboard, 5000); // Update every 5 seconds
        });
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', function() {
            if (refreshInterval) {
                clearInterval(refreshInterval);
            }
        });
    </script>
</body>
</html>
        """
    
    def run_server(self, debug=False, host='127.0.0.1', port=5000):
        """Run the Flask server"""
        print(f"Starting Flask API server at http://{host}:{port}")
        self.app.run(debug=debug, host=host, port=port, threaded=True)