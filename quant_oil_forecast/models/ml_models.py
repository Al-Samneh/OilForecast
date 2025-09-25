"""
Machine Learning Models Module
==============================

Implementation of various ML models for oil price forecasting.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from typing import Dict, List, Tuple, Optional, Any
import warnings

from ..config.settings import TIME_SERIES_SPLIT_N_SPLITS, RANDOM_STATE


class MLModelSuite:
    """
    Suite of machine learning models for oil price forecasting.
    """
    
    def __init__(self, models_config: Optional[Dict] = None):
        """
        Initialize ML model suite.
        
        Args:
            models_config: Dictionary of model configurations
        """
        self.models_config = models_config or self._get_default_config()
        self.fitted_models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def _get_default_config(self) -> Dict:
        """Get default model configurations."""
        return {
            'xgboost': {
                'model': xgb.XGBRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': RANDOM_STATE,
                    'n_jobs': -1
                },
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            },
            'random_forest': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': RANDOM_STATE,
                    'n_jobs': -1
                },
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'ridge': {
                'model': Ridge,
                'params': {
                    'random_state': RANDOM_STATE
                },
                'param_grid': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            }
        }
    
    def prepare_data(self, df: pd.DataFrame, target_col: str, 
                    feature_cols: Optional[List[str]] = None,
                    scale_features: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for model training.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            feature_cols: List of feature columns (if None, use all except target)
            scale_features: Whether to scale features
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Select features
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]
        
        # Extract features and target
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Remove rows with missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Scale features if requested
        if scale_features:
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                index=X.index,
                columns=X.columns
            )
            self.scalers['features'] = scaler
            return X_scaled, y
        
        return X, y
    
    def fit_models(self, X: pd.DataFrame, y: pd.Series, 
                  use_grid_search: bool = False) -> Dict[str, Any]:
        """
        Fit all models in the suite.
        
        Args:
            X: Feature matrix
            y: Target vector
            use_grid_search: Whether to use grid search for hyperparameter tuning
            
        Returns:
            Dictionary with fitting results
        """
        results = {}
        
        for model_name, config in self.models_config.items():
            print(f"Fitting {model_name}...")
            
            try:
                if use_grid_search and 'param_grid' in config:
                    # Use grid search with time series split
                    tscv = TimeSeriesSplit(n_splits=TIME_SERIES_SPLIT_N_SPLITS)
                    model = config['model']()
                    
                    grid_search = GridSearchCV(
                        model, 
                        config['param_grid'],
                        cv=tscv,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1
                    )
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        grid_search.fit(X, y)
                    
                    self.fitted_models[model_name] = grid_search.best_estimator_
                    results[model_name] = {
                        'best_params': grid_search.best_params_,
                        'best_score': grid_search.best_score_,
                        'cv_results': grid_search.cv_results_
                    }
                else:
                    # Use default parameters
                    model = config['model'](**config['params'])
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(X, y)
                    
                    self.fitted_models[model_name] = model
                    results[model_name] = {'status': 'fitted'}
                
                # Extract feature importance if available
                if hasattr(self.fitted_models[model_name], 'feature_importances_'):
                    self.feature_importance[model_name] = pd.Series(
                        self.fitted_models[model_name].feature_importances_,
                        index=X.columns,
                        name=f'{model_name}_importance'
                    ).sort_values(ascending=False)
                
                print(f"Successfully fitted {model_name}")
                
            except Exception as e:
                print(f"Error fitting {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Generate predictions from fitted models.
        
        Args:
            X: Feature matrix for prediction
            model_name: Specific model to use (if None, use all fitted models)
            
        Returns:
            Dictionary of predictions by model
        """
        if not self.fitted_models:
            raise ValueError("No models have been fitted")
        
        # Scale features if scaler was used during training
        if 'features' in self.scalers:
            X_scaled = pd.DataFrame(
                self.scalers['features'].transform(X),
                index=X.index,
                columns=X.columns
            )
        else:
            X_scaled = X
        
        predictions = {}
        
        models_to_use = [model_name] if model_name else list(self.fitted_models.keys())
        
        for name in models_to_use:
            if name in self.fitted_models:
                try:
                    pred = self.fitted_models[name].predict(X_scaled)
                    predictions[name] = pd.Series(pred, index=X.index, name=f'{name}_pred')
                except Exception as e:
                    print(f"Error predicting with {name}: {e}")
        
        return predictions
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Evaluate all fitted models on test data.
        
        Args:
            X_test: Test feature matrix
            y_test: Test target vector
            
        Returns:
            DataFrame with evaluation metrics
        """
        predictions = self.predict(X_test)
        
        metrics = []
        for model_name, pred in predictions.items():
            # Align predictions with test data
            common_idx = y_test.index.intersection(pred.index)
            y_true = y_test.loc[common_idx]
            y_pred = pred.loc[common_idx]
            
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            metrics.append({
                'model': model_name,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'n_samples': len(y_true)
            })
        
        return pd.DataFrame(metrics).set_index('model')
    
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """
        Get summary of feature importance across all models.
        
        Returns:
            DataFrame with feature importance by model
        """
        if not self.feature_importance:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame(self.feature_importance)
        importance_df = importance_df.fillna(0)
        
        # Add mean importance across models
        importance_df['mean_importance'] = importance_df.mean(axis=1)
        
        return importance_df.sort_values('mean_importance', ascending=False)
    
    def create_ensemble_prediction(self, X: pd.DataFrame, 
                                 weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """
        Create ensemble prediction from multiple models.
        
        Args:
            X: Feature matrix
            weights: Dictionary of model weights (if None, use equal weights)
            
        Returns:
            Ensemble prediction series
        """
        predictions = self.predict(X)
        
        if not predictions:
            raise ValueError("No predictions available")
        
        # Default to equal weights
        if weights is None:
            weights = {name: 1.0/len(predictions) for name in predictions.keys()}
        
        # Create weighted ensemble
        ensemble = None
        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 0)
            if ensemble is None:
                ensemble = weight * pred
            else:
                ensemble += weight * pred
        
        ensemble.name = 'ensemble_prediction'
        return ensemble


def perform_time_series_validation(model_suite: MLModelSuite, X: pd.DataFrame, y: pd.Series,
                                  n_splits: int = 5) -> pd.DataFrame:
    """
    Perform time series cross-validation for model evaluation.
    
    Args:
        model_suite: Fitted ML model suite
        X: Feature matrix
        y: Target vector
        n_splits: Number of time series splits
        
    Returns:
        DataFrame with cross-validation results
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Create new model suite for this fold
        fold_suite = MLModelSuite(model_suite.models_config)
        fold_suite.fit_models(X_train, y_train)
        
        # Evaluate on test set
        fold_metrics = fold_suite.evaluate_models(X_test, y_test)
        fold_metrics['fold'] = fold
        
        cv_results.append(fold_metrics.reset_index())
    
    return pd.concat(cv_results, ignore_index=True)
