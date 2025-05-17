# Stock Recommender

An interactive, risk‑aware stock recommendation system for S\&P 500 equities. Built with Python, TensorFlow, and Streamlit, this project ingests historical price data, engineers risk and momentum features, trains a multi‑output neural network, and provides monthly backtesting and a live UI for new investors to get tailored stock suggestions.

---

## Features

* **Data Fetching**: Scrape S\&P 500 tickers and download 5 years of daily prices via `yfinance`.
* **Feature Engineering**: Compute daily returns, annualized volatility, 60‑day momentum, and rolling Sharpe ratio.
* **Modeling**: Two‑headed TensorFlow model predicting next‑month return and volatility.
* **Inference UI**: Streamlit app to select a risk preference (α ∈ \[0,1]) and see top N stock recommendations.
* **Backtesting**: Monthly rebalance backtest measuring realized returns & volatility over a 21‑day horizon.
* **Containerized**: Docker support for consistent deployment.

---

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)

   * [1. Data Collection](#1-data-collection)
   * [2. Feature Engineering](#2-feature-engineering)
   * [3. Training](#3-training)
   * [4. Live App](#4-live-app)
   * [5. Backtesting](#5-backtesting)
3. [Docker](#docker)
4. [Project Structure](#project-structure)
5. [Future Work](#future-work)
6. [Contributing](#contributing)

---

## Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/stock-recommender.git
   cd stock-recommender
   ```
2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Data Collection

Fetch S\&P 500 tickers and download raw price CSVs:

```bash
python src/data_fetch.py
```

Data will be saved under `data/raw/`.

### 2. Feature Engineering

Compute and save rolling features:

```bash
python src/features.py
```

Outputs: `data/processed/features.csv` and `data/processed/latest_features.csv`.

### 3. Training

Train the TensorFlow model on historical features:

```bash
python src/train.py
```

Best model weights saved to `models/risk_model.h5`.

### 4. Live App

Launch the Streamlit recommendation UI:

```bash
streamlit run src/app.py
```

Open [http://localhost:8501](http://localhost:8501) to interactively choose your risk level and view top picks.

### 5. Backtesting

Run a monthly backtest of recommendations:

```bash
python src/backtest.py
```

Results saved to `data/backtest/backtest_results.csv` and a cumulative return plot.

---

## Docker

Build and run in Docker for a consistent environment:

```bash
# Build the image
docker build -t sp500-recommender .

# Run the container
docker run -p 8501:8501 sp500-recommender
```

Then visit [http://localhost:8501](http://localhost:8501).

---

## Project Structure

```
stock-recommender/
├── Dockerfile
├── requirements.txt
├── src/
│   ├── data_fetch.py       # Download raw price data
│   ├── features.py         # Compute rolling features
│   ├── model.py            # TensorFlow model definition
│   ├── train.py            # Model training script
│   ├── recommend.py        # Inference engine
│   ├── app.py              # Streamlit UI
│   └── backtest.py         # Backtesting script
├── data/
│   ├── raw/                # Raw CSV price history
│   ├── processed/          # Engineered feature tables
│   └── backtest/           # Backtest results
├── models/                 # Saved model weights
└── README.md
```

---

## Future Work

* **Backtesting enhancements**: rolling-window performance, transaction costs, and portfolio weighting schemes.
* **Hyperparameter tuning**: grid or Bayesian search for model architecture and training settings.
* **Deployment**: CI/CD pipelines, cloud deployment (Heroku, AWS ECS, Streamlit Cloud).
* **Expanded UI**: user authentication, historical charting, and portfolio tracking.

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes, feature ideas, or documentation improvements.

---

*© 2025 Anay Patel*
