# src/train.py

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

# Import the model builder from local module
from model import build_risk_model

# Paths
DATA_DIR       = "data/processed"
RAW_DIR        = "data/raw"
FEATURES_PATH  = os.path.join(DATA_DIR, "features.csv")
MODEL_DIR      = "models"

# Hyperparameters
EPOCHS          = 50
BATCH_SIZE      = 32
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "risk_model.h5")

os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Load computed features
df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])

# 2. Generate prediction labels (next-month return & volatility)
all_labels = []
for ticker, grp in df.groupby("ticker"):
    raw_csv = os.path.join(RAW_DIR, f"{ticker}.csv")
    if not os.path.exists(raw_csv):
        continue
    raw = pd.read_csv(raw_csv, index_col=0, parse_dates=True)
    raw.index.name = "Date"

    # Coerce Close to numeric and drop invalid rows
    raw["Close"] = pd.to_numeric(raw["Close"], errors="coerce")
    raw = raw.dropna(subset=["Close"])

    # Next-month (~21 trading days) return
    ret = raw["Close"].pct_change(periods=21).shift(-21)
    # Next-month volatility: annualized rolling std of daily returns
    daily_ret = raw["Close"].pct_change()
    vol = daily_ret.rolling(60).std() * np.sqrt(252)
    vol = vol.shift(-21)

    labels = pd.DataFrame({
        "ticker": ticker,
        "date": ret.index,
        "target_return": ret.values,
        "target_volatility": vol.values
    })
    all_labels.append(labels)

labels_df = pd.concat(all_labels, ignore_index=True)
# Ensure 'date' dtype matches features
labels_df["date"] = pd.to_datetime(labels_df["date"])

# 3. Merge features and labels on ticker & date
data = pd.merge(
    df,
    labels_df,
    on=["ticker", "date"],
    how="inner"
).dropna()

# 4. Split into train and validation (time-based split)
data = data.sort_values("date")
max_date = data["date"].max()
val_start = max_date - pd.DateOffset(months=6)
train_df = data[data["date"] < val_start]
val_df   = data[data["date"] >= val_start]

# 5. Prepare features & targets
feature_cols = ["return", "volatility", "momentum", "sharpe"]
X_train = train_df[feature_cols].values
y_train = [
    train_df["target_return"].values,
    train_df["target_volatility"].values
]
X_val = val_df[feature_cols].values
y_val = [
    val_df["target_return"].values,
    val_df["target_volatility"].values
]

# 6. Build and compile model
model = build_risk_model(input_dim=len(feature_cols))
model.compile(
    optimizer="adam",
    loss={
        "pred_return": "mse",
        "pred_volatility": "mse"
    },
    loss_weights={
        "pred_return": 1.0,
        "pred_volatility": 1.0
    }
)

# 7. Setup checkpoint
checkpoint = ModelCheckpoint(
    filepath=CHECKPOINT_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# 8. Train
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint]
)

print(f"\n✅ Training complete. Best model saved to {CHECKPOINT_PATH}")
