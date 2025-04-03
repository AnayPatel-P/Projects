# 📊 AI Portfolio Optimizer

An AI-powered portfolio optimization tool that helps users allocate assets based on modern portfolio theory (MPT). Built with **FastAPI**, **React**, and **PyPortfolioOpt**, this full-stack application allows users to select their risk level and receive an optimized allocation of US equities — along with expected return, volatility, and Sharpe ratio.

---

## 🌐 Live Demo

👉 [Try the App](https://ai-portfolio-frontend.netlify.app)

---

## 🚀 Features

- 🎯 Select a risk level: Low, Medium, or High
- 🧠 Backend computes optimized portfolio using PyPortfolioOpt
- 📈 Returns expected return, volatility, and Sharpe ratio
- 📊 Portfolio weights calculated from real-time yFinance data
- 🔌 Full-stack: React (frontend) + FastAPI (backend)
- 🌍 Deployed on Netlify (frontend) and Render (backend)

---

## 🧠 Tech Stack

| Layer      | Technology                               |
|------------|-------------------------------------------|
| Frontend   | React, TailwindCSS, Axios, Vite           |
| Backend    | FastAPI, PyPortfolioOpt, yFinance, Pandas |
| API Format | REST (JSON)                               |
| Deployment | Netlify (frontend) + Render (backend)     |

---

## 🛠 How It Works

1. User selects a risk level from the frontend
2. Frontend sends a POST request to `/optimize` on the backend
3. Backend fetches US equity prices, computes portfolio weights via PyPortfolioOpt
4. Result is sent back and displayed with stats & weights

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

- [ ] Allow custom ticker input
- [ ] Add pie chart of weights
- [ ] CSV / Excel / Power BI export
- [ ] Persistent portfolio storage

---

## 👨‍💻 Built by [Anay Patel](https://www.linkedin.com/in/anaypatel26)
```

---