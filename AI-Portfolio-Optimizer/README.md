# 📊 AI Portfolio Optimizer

An AI-powered portfolio optimization tool that helps users allocate assets based on Modern Portfolio Theory (MPT). Built with **FastAPI**, **React**, and **PyPortfolioOpt**, this full-stack application allows users to select a risk level and input any U.S. stock tickers to receive an optimized allocation — along with expected return, volatility, and Sharpe ratio.

---

## 🌐 Live Demo + Repos

- 🌍 [Live App](celebrated-florentine-798645.netlify.app)
- 💻 [Frontend Repo](https://github.com/AnayPatel-P/ai-portfolio-frontend)
- 🔌 [Backend Repo](https://github.com/AnayPatel-P/ai-portfolio-backend)

---

## 🚀 Features

- 🎯 Select a risk level: Low, Medium, or High
- ⌨️ Enter any valid U.S. stock tickers
- 🧠 Backend computes optimal allocation using PyPortfolioOpt
- 📈 Displays expected return, volatility, and Sharpe ratio
- 📊 Portfolio weights calculated from real-time yFinance data
- 📉 View historical price trends for selected tickers (line chart)
- 📥 Export results as CSV for Power BI
- 🔌 Full-stack architecture: React (frontend) + FastAPI (backend)
- 🌍 Hosted on Netlify (frontend) and Render (backend)

---

## 🧠 Tech Stack

| Layer      | Technology                               |
|------------|-------------------------------------------|
| Frontend   | React, TailwindCSS, Axios, Vite, Recharts |
| Backend    | FastAPI, PyPortfolioOpt, yFinance, Pandas |
| API Format | REST (JSON)                               |
| Deployment | Netlify (frontend) + Render (backend)     |

---

## 🛠 How It Works

1. User enters a comma-separated list of stock tickers and selects a risk level
2. Frontend sends a POST request to `/optimize` on the backend
3. Backend fetches historical stock data from yFinance and computes optimized weights via PyPortfolioOpt
4. The response includes:
   - Optimized weights
   - Expected return, volatility, Sharpe ratio
   - Historical price series
5. Frontend renders:
   - Allocation weights
   - Line chart of price history
   - Downloadable CSV

---

## 📂 Project Structure

```
AI-Portfolio-Optimizer/
├── backend/
│   ├── main.py
│   ├── optimizer.py
│   ├── data_loader.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── PortfolioForm.jsx
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── index.html
│   └── tailwind.config.js
```

---

## 🚀 **Quick Start**

### **Option 1: Easy Startup (Recommended)**
```bash
# Navigate to project directory
cd AI-Portfolio-Optimizer

# Run the startup script
./start.sh
```

### **Option 2: Manual Setup**

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

### **Access the Application**
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8001
- **API Documentation**: http://localhost:8001/docs

---

## ✨ **Enhanced Features**

### **New in This Version:**
- ✅ **Real-time Input Validation** with visual feedback
- ✅ **Interactive Pie Chart** for portfolio allocation
- ✅ **Portfolio History** with localStorage persistence
- ✅ **Advanced Risk Metrics** (VaR, Max Drawdown, Diversification Ratio)
- ✅ **Normalized Price Charts** showing percentage returns
- ✅ **Professional UI/UX** with loading states and better error handling
- ✅ **Enhanced Security** with environment-based CORS configuration
- ✅ **Performance Optimizations** with TTL caching
- ✅ **Comprehensive Logging** throughout the application
- ✅ **Improved Port Configuration** with automatic conflict resolution

### **Backend Improvements:**
- 🔒 Environment-based CORS security with multiple port support
- ⚡ TTL caching for price data (1-hour cache)
- 📊 Additional risk metrics and analytics
- 🛡️ Comprehensive error handling and validation
- 📝 Structured logging with configurable levels
- 🔧 Flexible port configuration (runs on port 8001 to avoid conflicts)

### **Frontend Improvements:**
- 🎨 Modern UI with improved visual design
- 📊 Interactive pie chart for allocation visualization
- 💾 Portfolio history with localStorage persistence
- ⚡ Real-time input validation and feedback
- 📈 Normalized percentage-based price charts

---

## 🛠️ Troubleshooting

### Port Conflicts
If you encounter "Address already in use" errors:

1. **Check what's running on the ports:**
   ```bash
   lsof -i :8001  # Check backend port
   lsof -i :5173  # Check frontend port
   ```

2. **Kill existing processes:**
   ```bash
   pkill -f "uvicorn main:app"  # Kill backend
   pkill -f "vite"              # Kill frontend
   ```

3. **Restart the application:**
   ```bash
   ./start.sh
   ```

### Frontend Connection Issues
If the frontend shows "Failed to fetch optimization results":
- Ensure both backend (port 8001) and frontend (port 5173) are running
- Check browser console for CORS errors
- Verify the application started without errors

---

## 🔮 Completed Improvements

- [x] ~~Normalize chart data for comparison (% return vs raw price)~~
- [x] ~~Add pie/bar chart to visualize allocation~~
- [x] ~~Export results to Excel or connect live to Power BI~~
- [x] ~~Save portfolio history using localStorage or backend~~
- [x] **Enhanced error handling and validation**
- [x] **Advanced risk metrics and analytics**
- [x] **Performance optimizations with caching**
- [x] **Professional UI/UX improvements**

---

## 👨‍💻 Built by [Anay Patel](https://www.linkedin.com/in/anaypatel26)

### **Enhanced Version with Enterprise-Grade Features**
