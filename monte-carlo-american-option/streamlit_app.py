import streamlit as st
import numpy as np
from src.analytics import plot_exercise_boundary, plot_convergence

st.title("American Option Monte Carlo LSM Explorer")

# Sidebar inputs
S0 = st.sidebar.slider("Initial Price (S0)", 50.0, 150.0, 100.0, step=5.0)
K = st.sidebar.slider("Strike Price (K)", 50.0, 150.0, 100.0, step=5.0)
r = st.sidebar.slider("Risk-Free Rate (r)", 0.0, 0.1, 0.05, step=0.005)
sigma = st.sidebar.slider("Volatility (σ)", 0.0, 1.0, 0.2, step=0.05)
T = st.sidebar.slider("Time to Maturity (T, years)", 0.1, 2.0, 1.0, step=0.1)
n_steps = st.sidebar.slider("Time Steps (n_steps)", 10, 100, 50, step=10)
n_paths = st.sidebar.slider("Number of Paths (n_paths)", 1000, 50000, 20000, step=1000)
option_type = st.sidebar.selectbox("Option Type", ["put", "call"])
random_seed = 42

# Early-Exercise Boundary
st.header("Early-Exercise Boundary")
fig1 = plot_exercise_boundary(
    S0=S0, K=K, r=r, sigma=sigma,
    T=T, n_steps=n_steps, n_paths=n_paths,
    option_type=option_type, random_seed=random_seed
)
st.pyplot(fig1)

# Convergence
st.header("Monte Carlo Convergence")
# define a few path counts for convergence
path_counts = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000]
fig2 = plot_convergence(
    S0=S0, K=K, r=r, sigma=sigma,
    T=T, n_steps=n_steps, path_counts=path_counts,
    option_type=option_type, random_seed=random_seed
)
st.pyplot(fig2)
