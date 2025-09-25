"""
GARCH Models Module
==================

Implementation of GARCH and related volatility models for oil price forecasting.
"""

import pandas as pd
import numpy as np
from arch import arch_model
from typing import Dict, Tuple, Optional, List
import warnings


class GARCHModel:
    """
    GARCH model implementation for volatility forecasting.
    """
    
    def __init__(self, vol_model: str = 'GARCH', p: int = 1, q: int = 1, 
                 dist: str = 'normal', mean_model: str = 'ARX'):
        """
        Initialize GARCH model.
        
        Args:
            vol_model: Volatility model type ('GARCH', 'EGARCH', 'GJR-GARCH')
            p: Number of GARCH lags
            q: Number of ARCH lags  
            dist: Error distribution ('normal', 't', 'skewt')
            mean_model: Mean model type ('Constant', 'Zero', 'ARX')
        """
        self.vol_model = vol_model
        self.p = p
        self.q = q
        self.dist = dist
        self.mean_model = mean_model
        self.model = None
        self.results = None
        
    def fit(self, returns: pd.Series, exog: Optional[pd.DataFrame] = None) -> 'GARCHModel':
        """
        Fit the GARCH model to return series.
        
        Args:
            returns: Return series to fit
            exog: Exogenous variables for mean equation
            
        Returns:
            Self for method chaining
        """
        # Clean the data
        returns_clean = returns.dropna()
        if exog is not None:
            exog_clean = exog.loc[returns_clean.index]
        else:
            exog_clean = None
        
        # Create and fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if exog_clean is not None:
                self.model = arch_model(
                    returns_clean,
                    x=exog_clean,
                    vol=self.vol_model,
                    p=self.p,
                    q=self.q,
                    dist=self.dist,
                    mean=self.mean_model
                )
            else:
                self.model = arch_model(
                    returns_clean,
                    vol=self.vol_model,
                    p=self.p,
                    q=self.q,
                    dist=self.dist,
                    mean=self.mean_model
                )
            
            self.results = self.model.fit(disp='off')
        
        return self
    
    def forecast(self, horizon: int = 1, method: str = 'simulation', 
                 simulations: int = 1000) -> Dict:
        """
        Generate forecasts from fitted model.
        
        Args:
            horizon: Forecast horizon
            method: Forecasting method ('simulation' or 'analytic')
            simulations: Number of simulations for simulation method
            
        Returns:
            Dictionary containing forecasts
        """
        if self.results is None:
            raise ValueError("Model must be fitted before forecasting")
        
        if method == 'simulation':
            forecast = self.results.forecast(horizon=horizon, method='simulation',
                                           simulations=simulations)
        else:
            forecast = self.results.forecast(horizon=horizon)
        
        return {
            'mean': forecast.mean.iloc[-1].values,
            'variance': forecast.variance.iloc[-1].values,
            'residual_variance': forecast.residual_variance.iloc[-1].values if hasattr(forecast, 'residual_variance') else None
        }
    
    def get_conditional_volatility(self) -> pd.Series:
        """
        Get conditional volatility from fitted model.
        
        Returns:
            Series of conditional volatility
        """
        if self.results is None:
            raise ValueError("Model must be fitted before extracting volatility")
        
        return pd.Series(
            np.sqrt(self.results.conditional_volatility),
            index=self.results.resid.index,
            name='conditional_volatility'
        )
    
    def get_model_summary(self) -> str:
        """
        Get model summary statistics.
        
        Returns:
            String representation of model summary
        """
        if self.results is None:
            return "Model not fitted"
        
        return str(self.results.summary())
    
    def get_information_criteria(self) -> Dict[str, float]:
        """
        Get information criteria for model selection.
        
        Returns:
            Dictionary with AIC, BIC, etc.
        """
        if self.results is None:
            raise ValueError("Model must be fitted")
        
        return {
            'aic': self.results.aic,
            'bic': self.results.bic,
            'loglikelihood': self.results.loglikelihood
        }


def fit_garch_models(returns: pd.Series, exog: Optional[pd.DataFrame] = None,
                    model_specs: List[Dict] = None) -> Dict[str, GARCHModel]:
    """
    Fit multiple GARCH model specifications and compare.
    
    Args:
        returns: Return series
        exog: Exogenous variables
        model_specs: List of model specification dictionaries
        
    Returns:
        Dictionary of fitted models
    """
    if model_specs is None:
        model_specs = [
            {'vol_model': 'GARCH', 'p': 1, 'q': 1},
            {'vol_model': 'EGARCH', 'p': 1, 'q': 1},
            {'vol_model': 'GJR-GARCH', 'p': 1, 'q': 1},
            {'vol_model': 'GARCH', 'p': 2, 'q': 1},
            {'vol_model': 'GARCH', 'p': 1, 'q': 2}
        ]
    
    fitted_models = {}
    
    for i, spec in enumerate(model_specs):
        model_name = f"{spec['vol_model']}({spec['p']},{spec['q']})"
        
        try:
            model = GARCHModel(**spec)
            model.fit(returns, exog)
            fitted_models[model_name] = model
            print(f"Successfully fitted {model_name}")
        except Exception as e:
            print(f"Failed to fit {model_name}: {e}")
            continue
    
    return fitted_models


def select_best_garch_model(fitted_models: Dict[str, GARCHModel], 
                           criterion: str = 'aic') -> Tuple[str, GARCHModel]:
    """
    Select best GARCH model based on information criteria.
    
    Args:
        fitted_models: Dictionary of fitted models
        criterion: Selection criterion ('aic', 'bic')
        
    Returns:
        Tuple of (best_model_name, best_model)
    """
    if not fitted_models:
        raise ValueError("No fitted models provided")
    
    best_score = float('inf')
    best_name = None
    best_model = None
    
    for name, model in fitted_models.items():
        try:
            criteria = model.get_information_criteria()
            score = criteria[criterion]
            
            if score < best_score:
                best_score = score
                best_name = name
                best_model = model
        except Exception as e:
            print(f"Error evaluating model {name}: {e}")
            continue
    
    if best_model is None:
        raise ValueError("No valid models found")
    
    return best_name, best_model
