
# 📊 AI Portfolio Optimizer

An AI-powered portfolio optimization tool that helps users allocate assets based on modern portfolio theory (MPT). Built with **FastAPI**, **React**, and **PyPortfolioOpt**, this full-stack application allows users to select their risk level and receive an optimized allocation of US equities — along with expected return, volatility, and Sharpe ratio.

---

## 🚀 Features

- ✅ Optimize a portfolio of US equities (10 tickers)
- 🎯 Select a risk profile: low, medium, or high
- 📈 Returns expected return, volatility, and Sharpe ratio
- 📊 Portfolio weights generated using the Efficient Frontier
- ⚡ Frontend built with React + TailwindCSS
- 🔌 Backend built with FastAPI + PyPortfolioOpt
- 🔁 Ready for Power BI or Excel integration

---

## 🧠 Tech Stack

| Layer      | Technology                               |
|------------|-------------------------------------------|
| Frontend   | React, TailwindCSS, Axios                 |
| Backend    | FastAPI, PyPortfolioOpt, yfinance, Pandas |
| API Format | REST (JSON)                               |
| Deployment | Vercel (frontend), Render/EC2 (backend)   |

---

## 📦 Project Structure

```
AI-Portfolio-Optimizer/
├── backend/
│   ├── main.py
│   ├── optimizer.py
│   ├── data_loader.py
│   └── requirements.txt
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   └── PortfolioForm.jsx
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── package.json
│   └── vite.config.js
└── README.md
```

---

## 🛠️ Getting Started

### 🔧 Backend Setup (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

API will run at: [http://localhost:8000](http://localhost:8000)  
Test it at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 🌐 Frontend Setup (React + Tailwind)

```bash
cd frontend
npm install
npm run dev
```

App runs at: [http://localhost:5173](http://localhost:5173)

---

## 📈 Example Response

```json
{
  "weights": {
    "AAPL": 0.0636,
    "AMZN": 0.0928,
    "GOOGL": 0.0721,
    "JNJ": 0.2390,
    "JPM": 0.0855,
    "META": 0.0267,
    "MSFT": 0.0671,
    "NVDA": 0.0,
    "PG": 0.2218,
    "XOM": 0.1310
  },
  "expected_return": 0.1226,
  "expected_volatility": 0.1837,
  "sharpe_ratio": 0.6674
}
```

---

## 📸 Screenshots

> _Add screenshots here of the frontend UI and Swagger API once available._

---

## 🧠 Future Ideas

- [ ] Allow users to enter their own ticker symbols
- [ ] Save & compare portfolios
- [ ] Export results to CSV or Power BI
- [ ] Add historical performance charting
- [ ] Deploy backend via Docker

---

## 📄 License

MIT License

---

## 👨‍💻 Built by [Anay Patel](https://www.linkedin.com/in/anaypatel18)
```

---

