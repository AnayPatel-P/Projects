# 🧠 Projects Directory

Welcome to the Projects directory of my GitHub! Here, you'll find a collection of applied AI/ML and quantitative finance projects that demonstrate my interests and capabilities in quantitative research, portfolio optimization, sports analytics, and explainable AI. Each subproject contains its own README, codebase, and documentation.

---


## 📊 [AI Portfolio Optimizer](./AI-Portfolio-Optimizer)

A full-stack web application that provides professional-grade portfolio optimization using Modern Portfolio Theory. Features include:

* **Advanced Portfolio Analytics:** Optimized allocation with risk metrics (VaR, Max Drawdown, Diversification Ratio)
* **Interactive Visualizations:** Pie charts for allocation, normalized price performance charts
* **Real-time Data Processing:** Live stock data via yFinance with TTL caching for performance
* **Portfolio History:** Save and reload previous optimizations with localStorage persistence
* **Export Capabilities:** CSV export for Power BI and Excel integration
* **Professional UI/UX:** Real-time validation, loading states, error handling

**Tech Stack:** FastAPI, React, PyPortfolioOpt, TailwindCSS, Recharts, yFinance

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

An end-to-end risk-aware stock recommendation system for S&P 500 equities:

* **Data Ingestion:** Scrapes S&P 500 tickers and downloads 5 years of historical prices via yfinance
* **Feature Engineering:** Computes daily returns, annualized volatility, momentum, and rolling Sharpe ratios
* **Modeling:** Trains a two-headed TensorFlow neural network to predict next-month return and volatility
* **Inference UI:** Streamlit app allowing users to select a risk preference (α ∈ [0,1]) and view top N stock picks
* **Backtesting:** Monthly rebalance evaluation measuring realized returns & volatility over a 21-day horizon
* **Containerization:** Docker support for easy deployment

**Tech Stack:** Python, TensorFlow, Pandas, Streamlit, Docker

---

## 🎯 Project Highlights

### **Quantitative Finance & Portfolio Management**
- **AI Portfolio Optimizer**: Modern Portfolio Theory implementation with advanced analytics and interactive optimization
- **Stock Recommender**: Neural network-based stock selection with risk management and backtesting

### **Machine Learning & AI**
- **Explainable AI Platform**: Model interpretability using SHAP/LIME techniques
- **College Football Predictor**: Sports analytics with ranking prediction models

### **Technical Capabilities**
- **Full-Stack Development**: FastAPI backends with React frontends, professional UI/UX
- **Production Systems**: Docker deployment, comprehensive testing, enterprise configuration
- **Interactive Dashboards**: Streamlit and React-based user interfaces with real-time data
- **Advanced Analytics**: Risk metrics, portfolio optimization, performance attribution, statistical analysis
- **Data Engineering**: Multi-source data ingestion, caching, processing, and validation

### **Key Technical Skills Demonstrated**
- **Python Ecosystem**: Pandas, NumPy, Scikit-learn, TensorFlow, FastAPI, Streamlit
- **Frontend Development**: React, TailwindCSS, Recharts, responsive design
- **Financial Engineering**: Portfolio optimization, risk management, Modern Portfolio Theory
- **Data Visualization**: Interactive charts, real-time dashboards, performance analytics
- **Software Engineering**: Full-stack development, API design, testing frameworks, configuration management
- **Machine Learning**: Supervised learning, model interpretation, feature engineering

---

Each project is built with a strong emphasis on clarity, reproducibility, and real-world relevance. The projects range from educational implementations to production-ready systems suitable for commercial use. Feel free to explore and reach out if you'd like to collaborate or discuss ideas!

— Anay Patel