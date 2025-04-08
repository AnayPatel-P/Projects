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

## 🔮 Future Improvements

- [ ] Normalize chart data for comparison (% return vs raw price)
- [ ] Add pie/bar chart to visualize allocation
- [ ] Export results to Excel or connect live to Power BI
- [ ] Save portfolio history using localStorage or backend

---

## 👨‍💻 Built by [Anay Patel](https://www.linkedin.com/in/anaypatel26)
```

---
