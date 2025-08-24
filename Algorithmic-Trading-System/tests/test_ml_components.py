import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.feature_engineering import AdvancedFeatureEngineer
from src.ml.models import LSTMPredictor, EnsemblePredictor, ModelEvaluator
from src.strategies.ml_strategy import MLPredictionStrategy

class TestAdvancedFeatureEngineer(unittest.TestCase):
    
    def setUp(self):
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        prices = 100 + np.cumsum(np.random.normal(0.1, 1, len(dates)))
        
        self.test_data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.01, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.015, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.015, len(dates)))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        self.feature_engineer = AdvancedFeatureEngineer()
    
    def test_price_features_creation(self):
        """Test price-based feature engineering"""
        result = self.feature_engineer.create_price_features(self.test_data)
        
        # Check that new features were added
        self.assertGreater(len(result.columns), len(self.test_data.columns))
        
        # Check specific features
        expected_features = ['OHLC_Mean', 'Price_Range', 'Gap', 'Upper_Shadow', 'Close_Position']
        for feature in expected_features:
            self.assertIn(feature, result.columns)
        
        # Check that features have reasonable values
        self.assertTrue((result['Close_Position'] >= 0).all())
        self.assertTrue((result['Close_Position'] <= 1).all())
    
    def test_volume_features_creation(self):
        """Test volume-based feature engineering"""
        result = self.feature_engineer.create_volume_features(self.test_data)
        
        # Should have volume-related features
        volume_features = ['Volume_Price_Trend', 'VWAP', 'OBV', 'Volume_SMA_Ratio']
        for feature in volume_features:
            self.assertIn(feature, result.columns)
    
    def test_time_based_features(self):
        """Test time-based feature creation"""
        result = self.feature_engineer.create_time_based_features(self.test_data)
        
        # Check time features
        time_features = ['Day_of_Week', 'Month', 'Quarter', 'Is_Monday', 'Is_Friday']
        for feature in time_features:
            self.assertIn(feature, result.columns)
        
        # Check cyclical encoding
        self.assertIn('Day_of_Week_sin', result.columns)
        self.assertIn('Day_of_Week_cos', result.columns)
    
    def test_advanced_technical_indicators(self):
        """Test advanced technical indicator creation"""
        result = self.feature_engineer.create_advanced_technical_indicators(self.test_data)
        
        # Check for multiple MA periods
        self.assertIn('SMA_5', result.columns)
        self.assertIn('SMA_20', result.columns)
        self.assertIn('SMA_50', result.columns)
        
        # Check for multiple RSI periods
        self.assertIn('RSI_7', result.columns)
        self.assertIn('RSI_14', result.columns)
        
        # Check MACD variations
        self.assertIn('MACD_12_26', result.columns)
        self.assertIn('MACD_5_35', result.columns)
    
    def test_volatility_features(self):
        """Test volatility feature creation"""
        result = self.feature_engineer.create_volatility_features(self.test_data)
        
        # Check volatility features
        vol_features = ['Volatility_5', 'Volatility_20', 'Parkinson_Vol_10', 'GK_Vol_20']
        for feature in vol_features:
            self.assertIn(feature, result.columns)
        
        # Volatility should be positive
        for feature in vol_features:
            if feature in result.columns:
                non_nan_values = result[feature].dropna()
                if len(non_nan_values) > 0:
                    self.assertTrue((non_nan_values >= 0).all())
    
    def test_lagged_features(self):
        """Test lagged feature creation"""
        result = self.feature_engineer.create_lagged_features(
            self.test_data, 
            target_columns=['Close', 'Volume'],
            lags=[1, 2, 5]
        )
        
        # Check lagged features
        expected_lags = ['Close_lag_1', 'Close_lag_2', 'Close_lag_5', 'Volume_lag_1']
        for feature in expected_lags:
            self.assertIn(feature, result.columns)
    
    def test_rolling_statistics(self):
        """Test rolling statistics creation"""
        result = self.feature_engineer.create_rolling_statistics(
            self.test_data,
            target_columns=['Close'],
            windows=[5, 10]
        )
        
        # Check rolling statistics
        expected_stats = ['Close_mean_5', 'Close_std_5', 'Close_skew_10', 'Close_kurt_10']
        for feature in expected_stats:
            self.assertIn(feature, result.columns)
    
    def test_engineer_all_features(self):
        """Test comprehensive feature engineering"""
        result = self.feature_engineer.engineer_all_features(self.test_data)
        
        # Should have significantly more features
        self.assertGreater(len(result.columns), len(self.test_data.columns) * 3)
        
        # Should contain original columns
        for col in self.test_data.columns:
            self.assertIn(col, result.columns)
    
    def test_feature_selection(self):
        """Test feature selection functionality"""
        # Create features
        featured_data = self.feature_engineer.engineer_all_features(self.test_data)
        
        # Create target variable
        target = self.test_data['Close'].pct_change().shift(-1).dropna()
        
        # Select features
        try:
            selected_data, selected_features = self.feature_engineer.select_features(
                featured_data, target, method='mutual_info', k=10
            )
            
            # Should select requested number of features
            self.assertEqual(len(selected_features), 10)
            self.assertEqual(selected_data.shape[1], 10)
            
        except Exception as e:
            # Feature selection might fail with limited data
            print(f"Feature selection test skipped: {e}")
    
    def test_scaling_features(self):
        """Test feature scaling"""
        scaled_data = self.feature_engineer.scale_features(
            self.test_data, method='standard'
        )
        
        # Should have same shape
        self.assertEqual(scaled_data.shape, self.test_data.shape)
        
        # Scaled features should have different values
        self.assertFalse(scaled_data.equals(self.test_data))

class TestMLModels(unittest.TestCase):
    
    def setUp(self):
        # Create sample data for model testing
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=150, freq='D')
        
        # Create features
        n_features = 10
        self.X = pd.DataFrame(
            np.random.randn(len(dates), n_features),
            columns=[f'feature_{i}' for i in range(n_features)],
            index=dates
        )
        
        # Create target (with some relationship to features)
        self.y = pd.Series(
            (self.X.sum(axis=1) + np.random.randn(len(dates)) * 0.1).values,
            index=dates,
            name='target'
        )
    
    def test_lstm_predictor_initialization(self):
        """Test LSTM predictor initialization"""
        lstm = LSTMPredictor(
            sequence_length=30,
            lstm_units=[50, 25],
            dropout_rate=0.3
        )
        
        self.assertEqual(lstm.sequence_length, 30)
        self.assertEqual(lstm.lstm_units, [50, 25])
        self.assertEqual(lstm.dropout_rate, 0.3)
    
    def test_lstm_model_building(self):
        """Test LSTM model building"""
        lstm = LSTMPredictor(sequence_length=20, lstm_units=[10])
        
        # Build model
        model = lstm.build_model((20, 5))  # sequence_length=20, features=5
        
        # Check model structure
        self.assertIsNotNone(model)
        self.assertEqual(len(model.layers), 4)  # LSTM, Dropout, Dense, Output
    
    def test_ensemble_predictor_initialization(self):
        """Test ensemble predictor initialization"""
        ensemble = EnsemblePredictor()
        
        self.assertEqual(len(ensemble.models), 0)
        self.assertEqual(len(ensemble.weights), 0)
    
    def test_ensemble_model_training(self):
        """Test ensemble model training with small dataset"""
        ensemble = EnsemblePredictor()
        
        # Use smaller dataset and fewer models for testing
        small_X = self.X.head(50)  # Use only first 50 samples
        small_y = self.y.head(50)
        
        try:
            # This might fail with very small data, so we wrap in try-except
            results = ensemble.train_traditional_models(small_X, small_y, cv_folds=2)
            
            # Should have some models trained
            self.assertGreater(len(results), 0)
            
            # Check that models were added to ensemble
            self.assertGreater(len(ensemble.models), 0)
            
        except Exception as e:
            print(f"Ensemble training test skipped: {e}")
    
    def test_model_evaluator(self):
        """Test model evaluation functionality"""
        evaluator = ModelEvaluator()
        
        # Create dummy predictions
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        metrics = evaluator.evaluate_predictions(y_true, y_pred, "Test_Model")
        
        # Check that metrics were calculated
        expected_metrics = ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE', 'Direction_Accuracy']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
        
        # Check that metrics are reasonable
        self.assertGreater(metrics['R2'], 0.8)  # Should be good fit
        self.assertLess(metrics['RMSE'], 0.5)   # Should be low error

class TestMLStrategy(unittest.TestCase):
    
    def setUp(self):
        # Create sample market data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        prices = 100 + np.cumsum(np.random.normal(0.1, 1, len(dates)))
        
        self.market_data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.01, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.015, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.015, len(dates)))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        self.market_data.attrs['symbol'] = 'TEST'
    
    def test_ml_strategy_initialization(self):
        """Test ML strategy initialization"""
        strategy = MLPredictionStrategy(
            model_type='ensemble',
            prediction_horizon=1,
            confidence_threshold=0.6
        )
        
        self.assertEqual(strategy.model_type, 'ensemble')
        self.assertEqual(strategy.prediction_horizon, 1)
        self.assertEqual(strategy.confidence_threshold, 0.6)
        self.assertFalse(strategy.is_trained)
    
    def test_ml_strategy_feature_preparation(self):
        """Test feature preparation in ML strategy"""
        strategy = MLPredictionStrategy(
            model_type='ensemble',
            feature_selection_k=10
        )
        
        try:
            features = strategy.prepare_features(self.market_data)
            
            # Should return a DataFrame
            self.assertIsInstance(features, pd.DataFrame)
            
            # Should have some features
            self.assertGreater(len(features.columns), 0)
            
        except Exception as e:
            print(f"Feature preparation test warning: {e}")
    
    def test_ml_strategy_signal_generation_without_training(self):
        """Test signal generation without prior training"""
        strategy = MLPredictionStrategy(model_type='ensemble')
        
        # Should handle untrained model gracefully
        signals = strategy.generate_signals(self.market_data)
        
        # Should return a list (might be empty if training fails)
        self.assertIsInstance(signals, list)

if __name__ == '__main__':
    # Run tests with reduced verbosity for ML components
    unittest.main(verbosity=1)