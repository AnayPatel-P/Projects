# Monte Carlo American Option Pricer

**Description:**
A Python implementation of the Longstaff–Schwartz Least-Squares Monte Carlo algorithm to price American-style options under Geometric Brownian Motion with variance reduction, convergence analysis, and interactive exploration.

## Features

* **LSM Pricing:** Price American puts and calls via regression on simulated paths.
* **Variance Reduction:** Antithetic variates and control variates to reduce Monte Carlo variance.
* **Analytics & Visualization:**

  * Plot the early-exercise boundary over time.
  * Show Monte Carlo convergence of the price estimator.
* **Interactive Notebook:** Jupyter notebook with sliders and widgets for parameter tuning.
* **Streamlit App:** Lightweight web UI for real-time exploration.

## Repo Structure

```
monte-carlo-american-option/
├── README.md
├── requirements.txt
├── LICENSE
├── src/
│   ├── __init__.py
│   ├── gbm.py
│   ├── lsm.py
│   ├── variates.py
│   └── analytics.py
├── tests/
│   ├── test_gbm.py
│   ├── test_lsm.py
│   └── test_variates.py
├── notebooks/
│   └── american_option_lsm.ipynb
├── streamlit_app.py
└── .github/
    └── workflows/ci.yml
```

## Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/YourUsername/monte-carlo-american-option.git
   cd monte-carlo-american-option
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running Tests

Execute unit tests with pytest:

```bash
pytest --maxfail=1 --disable-warnings -q
```

## Usage

### Jupyter Notebook Exploration

1. Open the notebook:

   ```bash
   jupyter lab notebooks/american_option_lsm.ipynb
   ```
2. Run all cells.
3. Adjust sliders or parameters inline to see updated boundary and convergence plots.

### Streamlit App

1. Launch Streamlit:

   ```bash
   streamlit run streamlit_app.py
   ```
2. Use sidebar controls to modify parameters and view:

   * **Early-Exercise Boundary** plot
   * **Monte Carlo Convergence** plot

## Parameter Definitions

* **S0**: Initial asset price.
* **K**: Strike price.
* **r**: Annualized risk-free rate.
* **σ** (`sigma`): Annualized volatility.
* **T**: Time to maturity (years).
* **n\_steps**: Number of discrete time steps in the simulation.
* **n\_paths**: Number of simulated paths.
* **path\_counts**: List of path counts for convergence analysis.
* **option\_type**: "put" or "call".
* **random\_seed**: Seed for RNG reproducibility.

## Contributing

Contributions welcome! Please submit issues or pull requests.

## License

MIT License
