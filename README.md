# 🧠 Projects Directory

Welcome to the Projects directory of my GitHub! Here, you'll find a collection of applied AI/ML and data science projects that demonstrate my interests and capabilities in quantitative research, sports analytics, and explainable AI. Each subproject contains its own README, codebase, and documentation.

---

## 📊 [AI Portfolio Optimizer](./ai-portfolio-optimizer)

A web-based application that allows users to input a custom list of tickers and receive:

* Optimized portfolio allocation using Modern Portfolio Theory (MPT)
* Sharpe Ratio maximization
* Visualizations of historical performance (normalized price charts)
* Power BI integration for rich analytics dashboards

**Tech Stack:** Python, React, yFinance, Plotly, Power BI

---

## 🏈 [College Football Rank Predictor](./college-football-rank-predictor)

A machine learning project that predicts NCAA college football team rankings using historical performance data.

* Models trained on 2013–2022 statistics
* Predicts 2023 final rankings
* Evaluates accuracy using ranking-based metrics
* Potential for expansion into betting edge simulation and team clustering

**Tech Stack:** Python (Pandas, Scikit-learn, XGBoost), Jupyter

---

## 🔍 [Explainable AI Platform](./explainable-ai-platform)

An interpretable ML framework that helps visualize and explain model predictions using SHAP and LIME.

* Use case: Predicting credit scores and loan default risks
* Interactive plots showing feature importance
* Supports both tree-based and linear models

**Tech Stack:** Python, SHAP, LIME, Streamlit

---

## 💡 [Stock Recommender](./Stock_Recommender)

An end-to-end risk-aware stock recommendation system for S\&P 500 equities:

* **Data Ingestion:** Scrapes S\&P 500 tickers and downloads 5 years of historical prices via yfinance
* **Feature Engineering:** Computes daily returns, annualized volatility, momentum, and rolling Sharpe ratios
* **Modeling:** Trains a two-headed TensorFlow neural network to predict next-month return and volatility
* **Inference UI:** Streamlit app allowing users to select a risk preference (α ∈ \[0,1]) and view top N stock picks
* **Backtesting:** Monthly rebalance evaluation measuring realized returns & volatility over a 21-day horizon
* **Containerization:** Docker support for easy deployment

**Tech Stack:** Python, TensorFlow, Pandas, Streamlit, Docker

---

Each project is built with a strong emphasis on clarity, reproducibility, and real-world relevance. Feel free to explore and reach out if you’d like to collaborate or discuss ideas!

— Anay Patel
