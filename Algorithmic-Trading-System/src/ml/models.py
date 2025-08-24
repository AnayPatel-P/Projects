import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, MultiHeadAttention
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

# Traditional ML
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Model Selection and Evaluation
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import optuna

class LSTMPredictor:
    def __init__(self, sequence_length: int = 60, lstm_units: List[int] = [50, 50],
                 dropout_rate: float = 0.2, dense_units: List[int] = [25]):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        
    def _create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple) -> Sequential:
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units=self.lstm_units[0], 
                      return_sequences=len(self.lstm_units) > 1,
                      input_shape=input_shape))
        model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i in range(1, len(self.lstm_units)):
            return_sequences = i < len(self.lstm_units) - 1
            model.add(LSTM(units=self.lstm_units[i], return_sequences=return_sequences))
            model.add(Dropout(self.dropout_rate))
        
        # Dense layers
        for units in self.dense_units:
            model.add(Dense(units=units, activation='relu'))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        
        return model
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
             validation_split: float = 0.2, epochs: int = 100,
             batch_size: int = 32, verbose: int = 1) -> Dict:
        
        # Scale features and target
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
        
        # Build model
        self.model = self.build_model((self.sequence_length, X.shape[1]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)
        ]
        
        # Train model
        history = self.model.fit(
            X_seq, y_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history.history
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        # Create sequences
        predictions = []
        for i in range(self.sequence_length, len(X_scaled)):
            seq = X_scaled[i-self.sequence_length:i].reshape(1, self.sequence_length, -1)
            pred = self.model.predict(seq, verbose=0)[0, 0]
            predictions.append(pred)
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        return predictions
    
    def save_model(self, filepath: str):
        if self.model is not None:
            self.model.save(f"{filepath}_lstm.h5")
            joblib.dump(self.scaler, f"{filepath}_target_scaler.pkl")
            joblib.dump(self.feature_scaler, f"{filepath}_feature_scaler.pkl")
    
    def load_model(self, filepath: str):
        self.model = tf.keras.models.load_model(f"{filepath}_lstm.h5")
        self.scaler = joblib.load(f"{filepath}_target_scaler.pkl")
        self.feature_scaler = joblib.load(f"{filepath}_feature_scaler.pkl")

class TransformerPredictor:
    def __init__(self, sequence_length: int = 60, d_model: int = 64, 
                 num_heads: int = 8, num_layers: int = 4, dropout_rate: float = 0.1):
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
    
    def build_model(self, input_shape: Tuple) -> Model:
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Embedding layer to project features to d_model dimensions
        x = Dense(self.d_model)(inputs)
        
        # Multi-head attention layers
        for _ in range(self.num_layers):
            # Multi-head attention
            attn_output = MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads,
                dropout=self.dropout_rate
            )(x, x)
            
            # Residual connection and layer norm
            x = tf.keras.layers.Add()([x, attn_output])
            x = tf.keras.layers.LayerNormalization()(x)
            
            # Feed-forward network
            ffn = Sequential([
                Dense(self.d_model * 4, activation='relu'),
                Dropout(self.dropout_rate),
                Dense(self.d_model)
            ])
            
            ffn_output = ffn(x)
            x = tf.keras.layers.Add()([x, ffn_output])
            x = tf.keras.layers.LayerNormalization()(x)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Output layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return model
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        # Similar to LSTM training but with transformer architecture
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
        
        # Build model
        self.model = self.build_model((self.sequence_length, X.shape[1]))
        
        # Training logic similar to LSTM
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=8, min_lr=0.0001)
        ]
        
        history = self.model.fit(
            X_seq, y_seq,
            validation_split=kwargs.get('validation_split', 0.2),
            epochs=kwargs.get('epochs', 80),
            batch_size=kwargs.get('batch_size', 32),
            callbacks=callbacks,
            verbose=kwargs.get('verbose', 1)
        )
        
        return history.history
    
    def _create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)

class EnsemblePredictor:
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scalers = {}
        
    def add_model(self, name: str, model, weight: float = 1.0):
        self.models[name] = model
        self.weights[name] = weight
    
    def train_traditional_models(self, X: pd.DataFrame, y: pd.Series,
                                cv_folds: int = 5) -> Dict:
        
        # Initialize models
        models_to_train = {
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            'catboost': CatBoostRegressor(iterations=100, random_state=42, verbose=False),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'elastic_net': ElasticNet(alpha=1.0, random_state=42)
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        results = {}
        
        for name, model in models_to_train.items():
            print(f"Training {name}...")
            
            cv_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Handle missing values
                X_train = X_train.fillna(X_train.median())
                X_val = X_val.fillna(X_train.median())
                
                # Train model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                # Calculate score
                score = mean_squared_error(y_val, y_pred)
                cv_scores.append(score)
            
            avg_score = np.mean(cv_scores)
            results[name] = {
                'model': model,
                'cv_score': avg_score,
                'cv_std': np.std(cv_scores)
            }
            
            print(f"{name} - CV Score: {avg_score:.6f} (+/- {np.std(cv_scores):.6f})")
        
        # Add best models to ensemble
        for name, result in results.items():
            if result['cv_score'] < 1e10:  # Filter out poor performers
                weight = 1.0 / (1.0 + result['cv_score'])  # Inverse error weighting
                self.add_model(name, result['model'], weight)
        
        return results
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series,
                                model_name: str, n_trials: int = 50) -> Dict:
        
        def objective(trial):
            if model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
                model = xgb.XGBRegressor(**params, random_state=42)
                
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
                model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
            
            else:
                raise ValueError(f"Hyperparameter optimization not implemented for {model_name}")
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                X_train = X_train.fillna(X_train.median())
                X_val = X_val.fillna(X_train.median())
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = mean_squared_error(y_val, y_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"Best parameters for {model_name}: {study.best_params}")
        print(f"Best score: {study.best_value:.6f}")
        
        return study.best_params
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise ValueError("No models in ensemble. Train models first.")
        
        predictions = []
        total_weight = sum(self.weights.values())
        
        for name, model in self.models.items():
            # Handle missing values
            X_clean = X.fillna(X.median())
            
            # Make prediction
            pred = model.predict(X_clean)
            weight = self.weights[name] / total_weight
            predictions.append(pred * weight)
        
        # Combine weighted predictions
        ensemble_pred = np.sum(predictions, axis=0)
        return ensemble_pred
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_dict[name] = np.abs(model.coef_)
        
        if importance_dict:
            importance_df = pd.DataFrame(importance_dict, index=feature_names)
            importance_df['average'] = importance_df.mean(axis=1)
            return importance_df.sort_values('average', ascending=False)
        
        return pd.DataFrame()

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                           model_name: str = "Model") -> Dict:
        
        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'Direction_Accuracy': self._direction_accuracy(y_true, y_pred)
        }
        
        self.metrics[model_name] = metrics
        return metrics
    
    def _direction_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) < 2:
            return 0.0
            
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        return np.mean(true_direction == pred_direction) * 100
    
    def compare_models(self) -> pd.DataFrame:
        if not self.metrics:
            return pd.DataFrame()
        
        comparison_df = pd.DataFrame(self.metrics).T
        return comparison_df.round(4)
    
    def walk_forward_validation(self, model, X: pd.DataFrame, y: pd.Series,
                               initial_train_size: int = 252,
                               step_size: int = 21) -> Dict:
        
        predictions = []
        actuals = []
        
        for start in range(initial_train_size, len(X) - step_size, step_size):
            # Training data
            X_train = X.iloc[:start]
            y_train = y.iloc[:start]
            
            # Test data
            X_test = X.iloc[start:start+step_size]
            y_test = y.iloc[start:start+step_size]
            
            # Handle missing values
            X_train = X_train.fillna(X_train.median())
            X_test = X_test.fillna(X_train.median())
            
            # Train and predict
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                # For deep learning models that need different training
                y_pred = model.predict(X_test)
            
            predictions.extend(y_pred)
            actuals.extend(y_test.values)
        
        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        return self.evaluate_predictions(actuals, predictions, "Walk_Forward")