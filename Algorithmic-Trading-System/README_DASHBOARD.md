# 🌐 Live Trading Dashboard Guide

This document provides comprehensive instructions for using the real-time web dashboards integrated with the algorithmic trading system.

## 📊 Dashboard Options

The system provides three different dashboard implementations:

### 1. **Dash Dashboard** (Primary)
- **Technology**: Plotly Dash
- **Features**: Real-time updates, interactive charts, advanced analytics
- **Best for**: Production monitoring, detailed analysis

### 2. **Streamlit Dashboard** 
- **Technology**: Streamlit
- **Features**: Clean UI, automatic refresh, easy customization
- **Best for**: Quick prototyping, presentations

### 3. **Flask Web Dashboard**
- **Technology**: Flask + REST API + HTML/JavaScript
- **Features**: Lightweight, REST API access, custom frontend
- **Best for**: Integration with other systems, API access

## 🚀 Quick Start

### Option 1: Dash Dashboard (Recommended)

```bash
# Run the complete live trading system with Dash dashboard
python examples/run_live_trading_demo.py
```

Access at: http://localhost:8050

### Option 2: Streamlit Dashboard

```bash
# Run with Streamlit dashboard
python examples/run_streamlit_dashboard.py
```

Access at: http://localhost:8501

### Option 3: Flask Dashboard

```bash
# Run with Flask web dashboard
python examples/run_flask_dashboard.py
```

Access at: http://localhost:5000

## 📈 Dashboard Features

### Real-Time Metrics
- **Portfolio Value**: Current total portfolio value
- **Total Return**: Percentage return since inception
- **Cash Balance**: Available cash for trading
- **Active Positions**: Number of open positions
- **Engine Status**: Real-time system status

### Interactive Charts
- **Portfolio Performance**: Real-time portfolio value chart
- **Trading Signals**: Visual display of buy/sell signals
- **Price Charts**: Live price feeds with technical indicators
- **Returns Distribution**: Historical return analysis
- **Drawdown Analysis**: Risk analysis visualizations

### Data Tables
- **Current Positions**: All open positions with P&L
- **Recent Trades**: Trade execution history
- **Strategy Performance**: Individual strategy metrics
- **Signal History**: Detailed signal information

### Risk Analytics
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Volatility**: Portfolio volatility metrics
- **Correlation Analysis**: Position correlation matrix

## 🛠️ Technical Implementation

### Architecture

```
Live Trading System
├── Strategy Engine (Core)
├── Paper Trading Account
├── Live Data Feed
└── Dashboard Layer
    ├── Dash Dashboard
    ├── Streamlit Dashboard
    └── Flask API Dashboard
```

### Key Components

**LiveTradingDashboard** (`src/dashboard/live_dashboard.py`)
- Real-time data collection from strategy engine
- Plotly charts with automatic updates
- Interactive controls and filters
- Multi-threading for data collection

**StreamlitTradingDashboard** (`src/dashboard/streamlit_dashboard.py`)
- Streamlit-based interface
- Automatic refresh capabilities
- Clean, modern UI design
- Easy customization

**TradingSystemAPI** (`src/dashboard/flask_api.py`)
- RESTful API endpoints
- JSON data exchange
- CORS support for cross-origin requests
- Built-in HTML dashboard

### Data Flow

```
Market Data → Strategy Engine → Dashboard
                ↑                    ↓
         Paper Account ←→ Real-time Updates
```

## 🔧 Configuration

### Environment Setup

```bash
# Install required packages
pip install -r requirements.txt

# Ensure all dashboard dependencies are available
pip install dash plotly streamlit flask flask-cors
```

### Customization Options

**Dashboard Refresh Rate**
```python
# Dash Dashboard - modify in live_dashboard.py
dcc.Interval(
    id='interval-component',
    interval=2000,  # 2 seconds
    n_intervals=0
)

# Streamlit Dashboard - modify in streamlit_dashboard.py
time.sleep(5)  # 5 seconds
st.rerun()
```

**Chart Styling**
```python
# Modify Plotly chart configurations
fig.update_layout(
    title="Custom Chart Title",
    template="plotly_dark",  # or "plotly_white"
    height=500
)
```

## 📊 API Endpoints (Flask Dashboard)

### Core Endpoints

- `GET /api/status` - Engine and system status
- `GET /api/account` - Account summary and balance
- `GET /api/positions` - Current portfolio positions
- `GET /api/trades?limit=50` - Recent trade history
- `GET /api/signals` - Recent trading signals
- `GET /api/performance` - Comprehensive performance data
- `GET /api/strategies` - Strategy information and status

### Control Endpoints

- `POST /api/engine/start` - Start trading engine
- `POST /api/engine/stop` - Stop trading engine
- `POST /api/strategies/{name}/activate` - Activate strategy
- `POST /api/strategies/{name}/deactivate` - Deactivate strategy

### Example API Usage

```python
import requests

# Get current account status
response = requests.get('http://localhost:5000/api/account')
account_data = response.json()

# Get recent trades
response = requests.get('http://localhost:5000/api/trades?limit=20')
trades = response.json()

# Start trading engine
response = requests.post('http://localhost:5000/api/engine/start', 
                        json={'symbols': ['AAPL', 'GOOGL', 'MSFT']})
```

## 🎯 Dashboard Usage Tips

### Monitoring Best Practices

1. **Real-Time Monitoring**
   - Keep dashboard open during trading hours
   - Monitor position sizes and risk metrics
   - Watch for system alerts and status changes

2. **Performance Analysis**
   - Review daily/weekly performance summaries
   - Analyze strategy effectiveness
   - Monitor risk-adjusted returns

3. **Risk Management**
   - Set up alerts for drawdown limits
   - Monitor position concentration
   - Track volatility metrics

### Troubleshooting

**Dashboard Not Loading**
```bash
# Check if trading system is running
curl http://localhost:5000/api/health

# Verify all dependencies are installed
pip install -r requirements.txt

# Check for port conflicts
lsof -i :8050  # Dash
lsof -i :8501  # Streamlit
lsof -i :5000  # Flask
```

**Data Not Updating**
- Verify strategy engine is running
- Check data feed connections
- Ensure WebSocket connections are active
- Review browser console for JavaScript errors

**Performance Issues**
- Reduce update frequency for large datasets
- Limit historical data retention
- Use data sampling for high-frequency updates

## 🚀 Advanced Features

### Custom Indicators
Add custom technical indicators to charts:

```python
# In live_dashboard.py, modify the chart creation
fig.add_trace(go.Scatter(
    x=df['timestamp'],
    y=df['custom_indicator'],
    mode='lines',
    name='Custom Indicator'
))
```

### Alert System
Implement custom alerts:

```python
# Add to dashboard callbacks
if account_summary['total_return_pct'] < -5.0:
    # Trigger alert system
    send_alert("Portfolio down 5%")
```

### Export Functionality
Add data export capabilities:

```python
# Export performance data
@app.callback(...)
def export_data():
    performance_data = strategy_engine.export_performance_data()
    return dcc.send_data_frame(df.to_csv, "performance_data.csv")
```

## 📱 Mobile Responsiveness

All dashboards are designed to work on mobile devices:

- **Dash Dashboard**: Responsive grid layout
- **Streamlit Dashboard**: Automatic mobile optimization
- **Flask Dashboard**: Bootstrap-based responsive design

## 🔒 Security Considerations

- Dashboards run on localhost by default
- For production deployment:
  - Use HTTPS
  - Implement authentication
  - Add API rate limiting
  - Use secure WebSocket connections

## 📈 Next Steps

1. **Enhanced Analytics**: Add more sophisticated risk metrics
2. **Alert System**: Implement email/SMS notifications
3. **Multi-Account Support**: Support multiple trading accounts
4. **Historical Analysis**: Deep historical performance analysis
5. **Portfolio Optimization**: Real-time portfolio rebalancing

---

The dashboard system provides comprehensive real-time monitoring and control capabilities for your algorithmic trading system. Choose the dashboard that best fits your needs and customize it according to your requirements.