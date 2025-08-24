import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from .base_strategy import BaseStrategy, TradingSignal, SignalType
from ..ml.feature_engineering import AdvancedFeatureEngineer
from ..ml.models import LSTMPredictor, EnsemblePredictor, ModelEvaluator

class MLPredictionStrategy(BaseStrategy):
    def __init__(self, 
                 model_type: str = 'lstm',
                 prediction_horizon: int = 1,
                 confidence_threshold: float = 0.6,
                 feature_selection_k: int = 50):
        super().__init__(f"ML Prediction Strategy ({model_type})")
        self.model_type = model_type
        self.prediction_horizon = prediction_horizon
        self.confidence_threshold = confidence_threshold
        self.feature_selection_k = feature_selection_k
        
        self.feature_engineer = AdvancedFeatureEngineer()
        self.model = None
        self.selected_features = None
        self.is_trained = False
        
        self.parameters = {
            'model_type': model_type,
            'prediction_horizon': prediction_horizon,
            'confidence_threshold': confidence_threshold,
            'feature_selection_k': feature_selection_k
        }
    
    def set_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.parameters:
                setattr(self, key, value)
                self.parameters[key] = value
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        print("Preparing advanced features for ML strategy...")
        
        # Engineer comprehensive feature set
        featured_data = self.feature_engineer.engineer_all_features(data)
        
        # Create target variable (future returns)
        target = featured_data['Close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        
        # Select best features
        if self.selected_features is None:
            featured_data_clean = featured_data.dropna()
            target_clean = target.dropna()
            
            # Align data
            common_index = featured_data_clean.index.intersection(target_clean.index)
            if len(common_index) > 100:  # Ensure sufficient data
                X_for_selection = featured_data_clean.loc[common_index]
                y_for_selection = target_clean.loc[common_index]
                
                try:
                    selected_data, self.selected_features = self.feature_engineer.select_features(
                        X_for_selection, y_for_selection, 
                        method='mutual_info', k=self.feature_selection_k
                    )
                    print(f"Selected {len(self.selected_features)} features")
                except Exception as e:
                    print(f"Feature selection failed: {e}")
                    # Fallback to using all numeric features
                    numeric_cols = featured_data.select_dtypes(include=[np.number]).columns.tolist()
                    self.selected_features = numeric_cols[:self.feature_selection_k]
        
        # Return only selected features
        if self.selected_features:
            return featured_data[self.selected_features]
        else:
            # Fallback to basic features
            basic_features = ['Close', 'Volume', 'RSI_14', 'MACD_12_26', 'SMA_20', 'SMA_50']
            available_features = [f for f in basic_features if f in featured_data.columns]
            return featured_data[available_features] if available_features else featured_data.iloc[:, :20]
    
    def train_model(self, data: pd.DataFrame, validation_split: float = 0.2) -> Dict:
        print(f"Training {self.model_type} model...")
        
        # Prepare features and target
        features = self.prepare_features(data)
        target = data['Close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        
        # Clean data
        features_clean = features.dropna()
        target_clean = target.dropna()
        
        # Align data
        common_index = features_clean.index.intersection(target_clean.index)
        if len(common_index) < 100:
            raise ValueError("Insufficient data for training. Need at least 100 samples.")
        
        X = features_clean.loc[common_index]
        y = target_clean.loc[common_index]
        
        print(f"Training data shape: {X.shape}")
        
        # Initialize and train model
        if self.model_type == 'lstm':
            self.model = LSTMPredictor(
                sequence_length=min(60, len(X) // 4),
                lstm_units=[50, 30],
                dropout_rate=0.3,
                dense_units=[25, 10]
            )
            
            try:
                history = self.model.train(X, y, validation_split=validation_split, 
                                         epochs=50, batch_size=32, verbose=1)
                self.is_trained = True
                return history
            except Exception as e:
                print(f"LSTM training failed: {e}")
                # Fallback to ensemble
                self.model_type = 'ensemble'
        
        if self.model_type == 'ensemble' or not self.is_trained:
            self.model = EnsemblePredictor()
            
            try:
                results = self.model.train_traditional_models(X, y, cv_folds=3)
                self.is_trained = True
                print("Ensemble training completed")
                return results
            except Exception as e:
                print(f"Ensemble training failed: {e}")
                self.is_trained = False
                return {}
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        if not self.is_trained or self.model is None:
            print("Model not trained. Training now...")
            try:
                self.train_model(data)
            except Exception as e:
                print(f"Training failed: {e}")
                return []
        
        # Prepare features
        features = self.prepare_features(data)
        
        if features.empty:
            return []
        
        # Generate predictions
        try:
            predictions = self.model.predict(features)
            
            # Handle case where predictions might be shorter than data
            if len(predictions) < len(data):
                # Pad predictions to match data length
                padding = len(data) - len(predictions)
                predictions = np.concatenate([np.zeros(padding), predictions])
            elif len(predictions) > len(data):
                predictions = predictions[:len(data)]
                
        except Exception as e:
            print(f"Prediction failed: {e}")
            return []
        
        # Generate signals based on predictions
        signals = []
        
        for i in range(len(predictions)):
            if i >= len(data) - self.prediction_horizon:  # Skip last few points
                continue
                
            current_date = data.index[i]
            current_price = data.loc[current_date, 'Close']
            predicted_return = predictions[i]
            
            # Calculate confidence based on prediction magnitude
            confidence = min(abs(predicted_return) * 10, 1.0)  # Scale and cap at 1.0
            confidence = max(confidence, 0.1)  # Minimum confidence
            
            # Generate signal if confidence is above threshold
            if confidence >= self.confidence_threshold:
                if predicted_return > 0.005:  # Positive return threshold (0.5%)
                    signal_type = SignalType.BUY
                elif predicted_return < -0.005:  # Negative return threshold (-0.5%)
                    signal_type = SignalType.SELL
                else:
                    continue  # No strong signal
                
                signal = TradingSignal(
                    timestamp=current_date,
                    symbol=data.attrs.get('symbol', 'UNKNOWN'),
                    signal=signal_type,
                    price=current_price,
                    confidence=confidence,
                    metadata={
                        'predicted_return': predicted_return,
                        'model_type': self.model_type,
                        'horizon': self.prediction_horizon
                    }
                )
                signals.append(signal)
        
        print(f"Generated {len(signals)} ML-based signals")
        return signals

class MLEnsembleStrategy(BaseStrategy):
    def __init__(self, 
                 models: List[str] = ['lstm', 'ensemble'],
                 voting_method: str = 'weighted',
                 min_agreement: int = 2):
        super().__init__("ML Ensemble Strategy")
        self.models = {}
        self.voting_method = voting_method
        self.min_agreement = min_agreement
        
        # Initialize individual ML strategies
        for model_name in models:
            self.models[model_name] = MLPredictionStrategy(
                model_type=model_name,
                prediction_horizon=1,
                confidence_threshold=0.5
            )
        
        self.parameters = {
            'voting_method': voting_method,
            'min_agreement': min_agreement,
            'models': models
        }
    
    def set_parameters(self, **kwargs):
        self.parameters.update(kwargs)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def train_all_models(self, data: pd.DataFrame) -> Dict:
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name} model...")
            try:
                result = model.train_model(data)
                results[name] = result
                print(f"✅ {name} model trained successfully")
            except Exception as e:
                print(f"❌ {name} model training failed: {e}")
                results[name] = None
        
        return results
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        # Get signals from all models
        all_signals = {}
        for name, model in self.models.items():
            try:
                signals = model.generate_signals(data)
                # Convert to DataFrame for easier processing
                if signals:
                    signal_df = pd.DataFrame([{
                        'date': s.timestamp,
                        'signal': s.signal.value,
                        'confidence': s.confidence,
                        'price': s.price
                    } for s in signals])
                    signal_df.set_index('date', inplace=True)
                    all_signals[name] = signal_df
            except Exception as e:
                print(f"Error generating signals for {name}: {e}")
                continue
        
        if not all_signals:
            return []
        
        # Combine signals using voting
        ensemble_signals = self._combine_signals(all_signals, data)
        return ensemble_signals
    
    def _combine_signals(self, all_signals: Dict[str, pd.DataFrame], 
                        data: pd.DataFrame) -> List[TradingSignal]:
        
        # Get all unique dates
        all_dates = set()
        for signal_df in all_signals.values():
            all_dates.update(signal_df.index)
        
        ensemble_signals = []
        
        for date in sorted(all_dates):
            if date not in data.index:
                continue
                
            current_price = data.loc[date, 'Close']
            
            # Collect votes from all models
            votes = []
            confidences = []
            
            for name, signal_df in all_signals.items():
                if date in signal_df.index:
                    signal_value = signal_df.loc[date, 'signal']
                    confidence = signal_df.loc[date, 'confidence']
                    
                    votes.append(signal_value)
                    confidences.append(confidence)
            
            if len(votes) < self.min_agreement:
                continue
            
            # Determine ensemble signal
            if self.voting_method == 'majority':
                # Simple majority vote
                buy_votes = sum(1 for v in votes if v == 1)
                sell_votes = sum(1 for v in votes if v == -1)
                
                if buy_votes > sell_votes and buy_votes >= self.min_agreement:
                    signal_type = SignalType.BUY
                    confidence = np.mean([c for v, c in zip(votes, confidences) if v == 1])
                elif sell_votes > buy_votes and sell_votes >= self.min_agreement:
                    signal_type = SignalType.SELL
                    confidence = np.mean([c for v, c in zip(votes, confidences) if v == -1])
                else:
                    continue
                    
            elif self.voting_method == 'weighted':
                # Weighted vote by confidence
                weighted_vote = sum(v * c for v, c in zip(votes, confidences))
                total_confidence = sum(confidences)
                
                if total_confidence == 0:
                    continue
                
                avg_vote = weighted_vote / total_confidence
                confidence = np.mean(confidences)
                
                if avg_vote > 0.3:  # Positive threshold
                    signal_type = SignalType.BUY
                elif avg_vote < -0.3:  # Negative threshold
                    signal_type = SignalType.SELL
                else:
                    continue
            else:
                continue
            
            # Create ensemble signal
            signal = TradingSignal(
                timestamp=date,
                symbol=data.attrs.get('symbol', 'UNKNOWN'),
                signal=signal_type,
                price=current_price,
                confidence=confidence,
                metadata={
                    'ensemble_vote': avg_vote if self.voting_method == 'weighted' else None,
                    'num_votes': len(votes),
                    'voting_method': self.voting_method
                }
            )
            ensemble_signals.append(signal)
        
        print(f"Generated {len(ensemble_signals)} ensemble signals from {len(all_signals)} models")
        return ensemble_signals

class AdaptiveMLStrategy(BaseStrategy):
    def __init__(self, 
                 retraining_frequency: int = 60,
                 performance_window: int = 20,
                 adaptation_threshold: float = 0.1):
        super().__init__("Adaptive ML Strategy")
        self.retraining_frequency = retraining_frequency
        self.performance_window = performance_window
        self.adaptation_threshold = adaptation_threshold
        
        self.base_strategy = MLPredictionStrategy(model_type='ensemble')
        self.last_training_date = None
        self.performance_tracker = []
        
        self.parameters = {
            'retraining_frequency': retraining_frequency,
            'performance_window': performance_window,
            'adaptation_threshold': adaptation_threshold
        }
    
    def set_parameters(self, **kwargs):
        self.parameters.update(kwargs)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def should_retrain(self, current_date: pd.Timestamp) -> bool:
        # Check if it's time to retrain based on frequency
        if self.last_training_date is None:
            return True
        
        days_since_training = (current_date - self.last_training_date).days
        if days_since_training >= self.retraining_frequency:
            return True
        
        # Check if performance has degraded
        if len(self.performance_tracker) >= self.performance_window:
            recent_performance = np.mean(self.performance_tracker[-self.performance_window:])
            overall_performance = np.mean(self.performance_tracker)
            
            if (overall_performance - recent_performance) > self.adaptation_threshold:
                print("Performance degradation detected. Triggering retraining.")
                return True
        
        return False
    
    def update_performance(self, actual_return: float, predicted_return: float):
        # Simple performance metric: negative absolute error
        error = abs(actual_return - predicted_return)
        performance = -error  # Negative because lower error is better
        
        self.performance_tracker.append(performance)
        
        # Keep only recent performance history
        if len(self.performance_tracker) > self.performance_window * 3:
            self.performance_tracker = self.performance_tracker[-self.performance_window * 2:]
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        current_date = data.index[-1]
        
        # Check if retraining is needed
        if self.should_retrain(current_date):
            print(f"Retraining model on {current_date}")
            try:
                # Use recent data for training (last 2 years or available data)
                training_data = data.tail(min(504, len(data)))  # ~2 years of daily data
                self.base_strategy.train_model(training_data)
                self.last_training_date = current_date
                print("Retraining completed successfully")
            except Exception as e:
                print(f"Retraining failed: {e}")
        
        # Generate signals using the current model
        return self.base_strategy.generate_signals(data)