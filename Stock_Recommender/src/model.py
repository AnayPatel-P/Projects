# src/model.py

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_risk_model(input_dim: int = 4) -> Model:
    """
    Builds a two-head TensorFlow model:
      - pred_return: linear output for next-month return prediction
      - pred_volatility: softplus output for next-month volatility prediction
    """
    inputs = layers.Input(shape=(input_dim,), name="features")
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(32, activation="relu")(x)

    pred_return = layers.Dense(1, activation="linear", name="pred_return")(x)
    pred_volatility = layers.Dense(1, activation="softplus", name="pred_volatility")(x)

    model = Model(inputs=inputs, outputs=[pred_return, pred_volatility])
    return model


if __name__ == "__main__":
    # Sanity check: build and summarize the model
    model = build_risk_model()
    model.summary()