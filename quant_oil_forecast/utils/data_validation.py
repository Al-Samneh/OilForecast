"""
Data Validation Module
======================

This module provides functions to validate that no future information leaks into the forecasting models.
Critical for maintaining proper temporal constraints in quantitative finance applications.

Author: Professional Quant Engineering Team
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings


class TemporalValidationError(Exception):
    """Raised when temporal validation fails (future information detected)."""
    pass


def validate_no_lookahead_bias(df: pd.DataFrame, 
                              prediction_date: pd.Timestamp,
                              feature_metadata: Dict[str, Dict[str, Any]] = None,
                              strict_mode: bool = True,
                              enforce_no_future_rows: bool = True) -> Dict[str, Any]:
    """
    Validate that no features use information from after the prediction date.
    
    This is CRITICAL for realistic forecasting - using future information 
    will give artificially good performance that cannot be replicated in production.
    
    Args:
        df: DataFrame with features for prediction
        prediction_date: Date for which we're making predictions
        feature_metadata: Dictionary mapping feature names to metadata (publication_lag, etc.)
        strict_mode: If True, raise exception on violations. If False, return warnings.
        
    Returns:
        Dictionary with validation results
        
    Raises:
        TemporalValidationError: If lookahead bias detected and strict_mode=True
    """
    validation_results = {
        'prediction_date': prediction_date,
        'violations': [],
        'warnings': [],
        'passed': True,
        'total_features': len(df.columns)
    }
    
    # Check 1: No data points after prediction date (optional for test sets)
    if enforce_no_future_rows:
        future_data_mask = df.index > prediction_date
        if future_data_mask.any():
            violation = f"Found {future_data_mask.sum()} data points after prediction date {prediction_date}"
            validation_results['violations'].append(violation)
            validation_results['passed'] = False
            
            if strict_mode:
                raise TemporalValidationError(violation)
    
    # Check 2: Publication lag compliance for known features
    if feature_metadata:
        for feature_name, metadata in feature_metadata.items():
            if feature_name not in df.columns:
                continue
                
            publication_lag = metadata.get('publication_lag', '0D')
            # Support month-based metadata as well
            if publication_lag == '0D' and 'publication_lag_months' in metadata:
                try:
                    months = int(metadata.get('publication_lag_months', 0))
                    publication_lag = f"{months * 30}D"
                except Exception:
                    pass
            
            # Parse publication lag
            if isinstance(publication_lag, str):
                if 'D' in publication_lag:
                    lag_days = int(publication_lag.replace('D', ''))
                else:
                    lag_days = 0
            else:
                lag_days = int(publication_lag)
            
            # Check if data is available considering publication lag
            effective_cutoff = prediction_date - timedelta(days=lag_days)
            
            # Only consider data available up to prediction_date to avoid false positives
            feature_data = df.loc[df.index <= prediction_date, feature_name].dropna()
            if not feature_data.empty:
                latest_available = feature_data.index.max()
                
                # Skip strict enforcement for published series that are already delayed
                if latest_available > effective_cutoff and not metadata.get('is_published_series', False):
                    violation = (f"Feature '{feature_name}' has data ({latest_available.date()}) "
                               f"that would not be available by prediction date considering "
                               f"{lag_days}-day publication lag (cutoff: {effective_cutoff.date()})")
                    validation_results['violations'].append(violation)
                    validation_results['passed'] = False
                    
                    if strict_mode:
                        raise TemporalValidationError(violation)
    
    # Check 3: Common problematic feature patterns
    problematic_patterns = {
        'next_day': ['next', 'future', 'forward'],
        'same_day': ['close', 'adj_close'] if prediction_date.hour < 16 else [],  # Market closes at 4 PM
        'intraday': ['high', 'low', 'volume'] if prediction_date.hour < 20 else []  # Data available after market close
    }
    
    for pattern_type, keywords in problematic_patterns.items():
        for keyword in keywords:
            matching_cols = [col for col in df.columns if keyword.lower() in col.lower()]
            if matching_cols:
                warning = f"Found potentially problematic features with '{keyword}' pattern: {matching_cols}"
                validation_results['warnings'].append(warning)
    
    # Check 4: Verify time series ordering
    if not df.index.is_monotonic_increasing:
        violation = "DataFrame index is not monotonically increasing - this can cause temporal leakage"
        validation_results['violations'].append(violation)
        validation_results['passed'] = False
        
        if strict_mode:
            raise TemporalValidationError(violation)
    
    return validation_results


def validate_train_test_split(train_df: pd.DataFrame, 
                             test_df: pd.DataFrame,
                             min_gap_days: int = 0) -> Dict[str, Any]:
    """
    Validate that train/test split maintains proper temporal ordering.
    
    Args:
        train_df: Training dataset
        test_df: Test dataset  
        min_gap_days: Minimum gap between train and test (to avoid leakage from autocorrelation)
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'passed': True,
        'violations': [],
        'train_end': train_df.index.max(),
        'test_start': test_df.index.min(),
        'gap_days': None
    }
    
    # Check temporal ordering
    train_end = train_df.index.max()
    test_start = test_df.index.min()
    
    if train_end >= test_start:
        violation = f"Train data ends ({train_end.date()}) after test data starts ({test_start.date()})"
        validation_results['violations'].append(violation)
        validation_results['passed'] = False
        
    # Check minimum gap
    gap_days = (test_start - train_end).days
    validation_results['gap_days'] = gap_days
    
    if gap_days < min_gap_days:
        violation = f"Gap between train and test ({gap_days} days) is less than minimum required ({min_gap_days} days)"
        validation_results['violations'].append(violation)
        validation_results['passed'] = False
    
    return validation_results


def check_gpr_publication_lag(df: pd.DataFrame, 
                             expected_lag_days: int = 7) -> Dict[str, Any]:
    """
    Specifically validate GPR data publication lag compliance.
    
    GPR data is published weekly (typically Mondays) so daily GPR values
    should not be available for at least 7 days.
    
    Args:
        df: DataFrame containing GPR features
        expected_lag_days: Expected publication lag in days
        
    Returns:
        Dictionary with GPR-specific validation results
    """
    # Consider only raw GPR columns (exclude engineered columns like *_diff, *_ma*, *_std*, *_zscore*)
    def _is_raw_gpr(col: str) -> bool:
        col_lower = col.lower()
        if not col_lower.startswith('gpr'):
            return False
        # Exclude engineered variants
        engineered_tokens = ['_diff', '_ma', '_std', '_zscore']
        if any(tok in col_lower for tok in engineered_tokens):
            return False
        # Exclude known metadata columns
        if col_lower in ['gpr_confidence', 'gpr_days_stale', 'gpr_quality_score']:
            return False
        return True

    gpr_cols = [col for col in df.columns if _is_raw_gpr(col)]
    
    validation_results = {
        'passed': True,
        'violations': [],
        'gpr_features_found': gpr_cols,
        'expected_lag_days': expected_lag_days
    }
    
    if not gpr_cols:
        validation_results['violations'].append("No GPR features found for validation")
        return validation_results
    
    # Check each GPR feature for proper lag
    for col in gpr_cols:
        if col in ['gpr_confidence', 'gpr_days_stale', 'gpr_quality_score']:
            continue  # These are metadata columns, not raw GPR data
            
        feature_data = df[col].dropna()
        if feature_data.empty:
            continue
            
        # Check if the latest GPR data is properly lagged
        latest_date = feature_data.index.max()
        # Evaluate relative to dataset end, not wall-clock "now"
        dataset_end = df.index.max().normalize() if not df.index.empty else pd.Timestamp.now().normalize()
        actual_lag = (dataset_end - latest_date.normalize()).days
        
        if actual_lag < expected_lag_days:
            violation = (f"GPR feature '{col}' latest date {latest_date.date()} is <{expected_lag_days} days before "
                        f"dataset end {dataset_end.date()} (lag={actual_lag}d)")
            validation_results['violations'].append(violation)
            validation_results['passed'] = False
    
    return validation_results


def check_feature_staleness(df: pd.DataFrame, 
                           max_staleness_days: Dict[str, int] = None) -> Dict[str, Any]:
    """
    Check if features are too stale (old) to be useful for prediction.
    
    Args:
        df: DataFrame with features
        max_staleness_days: Dict mapping feature patterns to max allowed staleness
        
    Returns:
        Dictionary with staleness validation results
    """
    if max_staleness_days is None:
        max_staleness_days = {
            'market': 1,      # Market data should be very recent
            'gpr': 14,        # GPR can be up to 2 weeks old
            'epu': 3,         # EPU should be fairly recent
            'weather': 1,     # Weather should be recent
            'conflict': 365   # Conflict data can be older
        }
    
    current_date = pd.Timestamp.now().normalize()
    
    validation_results = {
        'passed': True,
        'warnings': [],
        'staleness_check': {}
    }
    
    for feature_pattern, max_days in max_staleness_days.items():
        # Find matching columns
        if feature_pattern == 'market':
            matching_cols = [col for col in df.columns 
                           if any(term in col.lower() for term in ['price', 'yield', 'dxy', 'vix', 'sp500', 'nasdaq'])]
        elif feature_pattern == 'gpr':
            matching_cols = [col for col in df.columns if 'gpr' in col.lower()]
        elif feature_pattern == 'epu':
            matching_cols = [col for col in df.columns if 'epu' in col.lower()]
        elif feature_pattern == 'weather':
            matching_cols = [col for col in df.columns 
                           if any(term in col.lower() for term in ['temp', 'precip', 'wind'])]
        elif feature_pattern == 'conflict':
            matching_cols = [col for col in df.columns 
                           if any(term in col.lower() for term in ['conflict', 'flag', 'best'])]
        else:
            continue
        
        for col in matching_cols:
            feature_data = df[col].dropna()
            if feature_data.empty:
                continue
                
            latest_date = feature_data.index.max()
            staleness_days = (current_date - latest_date).days
            
            validation_results['staleness_check'][col] = {
                'latest_date': latest_date,
                'staleness_days': staleness_days,
                'max_allowed': max_days,
                'status': 'ok' if staleness_days <= max_days else 'stale'
            }
            
            if staleness_days > max_days:
                warning = (f"Feature '{col}' is stale: {staleness_days} days old "
                          f"(max allowed: {max_days} days)")
                validation_results['warnings'].append(warning)
    
    return validation_results


def generate_validation_report(validation_results: List[Dict]) -> str:
    """
    Generate a comprehensive validation report.
    
    Args:
        validation_results: List of validation result dictionaries
        
    Returns:
        Formatted validation report string
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("TEMPORAL DATA VALIDATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    total_violations = 0
    total_warnings = 0
    
    for i, result in enumerate(validation_results, 1):
        report_lines.append(f"Check {i}: {result.get('check_name', 'Unknown')}")
        report_lines.append("-" * 40)
        
        if result.get('passed', False):
            report_lines.append("✅ PASSED")
        else:
            report_lines.append("❌ FAILED")
            
        violations = result.get('violations', [])
        warnings = result.get('warnings', [])
        
        total_violations += len(violations)
        total_warnings += len(warnings)
        
        if violations:
            report_lines.append("\nVIOLATIONS:")
            for violation in violations:
                report_lines.append(f"  • {violation}")
        
        if warnings:
            report_lines.append("\nWARNINGS:")
            for warning in warnings:
                report_lines.append(f"  • {warning}")
        
        report_lines.append("")
    
    # Summary
    report_lines.append("=" * 80)
    report_lines.append("SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"Total Checks: {len(validation_results)}")
    report_lines.append(f"Total Violations: {total_violations}")
    report_lines.append(f"Total Warnings: {total_warnings}")
    
    if total_violations == 0:
        report_lines.append("\n🎉 ALL TEMPORAL VALIDATION CHECKS PASSED!")
        report_lines.append("No future information leakage detected.")
    else:
        report_lines.append("\n⚠️  TEMPORAL VALIDATION FAILURES DETECTED!")
        report_lines.append("Please fix violations before using the model for trading.")
    
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


def run_comprehensive_validation(train_df: pd.DataFrame,
                                test_df: pd.DataFrame,
                                feature_metadata: Dict[str, Dict] = None,
                                prediction_date: pd.Timestamp = None) -> str:
    """
    Run comprehensive temporal validation on the entire pipeline.
    
    Args:
        train_df: Training dataset
        test_df: Test dataset
        feature_metadata: Metadata about features
        prediction_date: Date for prediction (defaults to test start date)
        
    Returns:
        Comprehensive validation report
    """
    if prediction_date is None:
        prediction_date = test_df.index.min()
    
    validation_results = []
    
    # 1. Train/Test split validation
    split_result = validate_train_test_split(train_df, test_df, min_gap_days=1)
    split_result['check_name'] = 'Train/Test Split Temporal Ordering'
    validation_results.append(split_result)
    
    # 2. Training data lookahead validation
    train_lookahead = validate_no_lookahead_bias(
        train_df, prediction_date, feature_metadata, strict_mode=False
    )
    train_lookahead['check_name'] = 'Training Data Lookahead Bias'
    validation_results.append(train_lookahead)
    
    # 3. Test data lookahead validation  
    test_lookahead = validate_no_lookahead_bias(
        test_df, prediction_date, feature_metadata, strict_mode=False
    )
    test_lookahead['check_name'] = 'Test Data Lookahead Bias'
    validation_results.append(test_lookahead)
    
    # 4. GPR publication lag validation
    gpr_validation = check_gpr_publication_lag(pd.concat([train_df, test_df]))
    gpr_validation['check_name'] = 'GPR Publication Lag Compliance'
    validation_results.append(gpr_validation)
    
    # 5. Feature staleness check
    staleness_check = check_feature_staleness(pd.concat([train_df, test_df]))
    staleness_check['check_name'] = 'Feature Staleness Assessment'
    staleness_check['passed'] = len(staleness_check.get('warnings', [])) == 0
    validation_results.append(staleness_check)
    
    return generate_validation_report(validation_results)
