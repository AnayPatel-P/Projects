# src/app.py

import streamlit as st
import pandas as pd
from recommend import recommend

st.set_page_config(page_title="S&P 500 Risk-Based Recommender")
st.title("S&P 500 Risk-Based Stock Recommender")

# Inputs
data_path = "data/processed/latest_features.csv"
features = pd.read_csv(data_path, index_col=0)

risk = st.slider(
    "Select your risk preference (alpha):",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

top_n = st.number_input(
    "Number of stocks to recommend:",
    min_value=1,
    max_value=50,
    value=10
)

if st.button("Recommend Stocks"):
    picks = recommend(features, alpha=risk, top_n=top_n)
    st.subheader("Top Picks")
    st.dataframe(picks.set_index("ticker"))

    st.write("### Interpretation:")
    st.write(
        "*Higher `pred_return` indicates expected higher returns;*  \
         *lower `pred_volatility` indicates lower risk.*"
    )

# Footer
st.markdown("---")
st.markdown("Built with TensorFlow & Streamlit by Anay Patel")