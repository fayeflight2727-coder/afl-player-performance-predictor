"""
Drift monitoring for model features.
To be implemented in Week 3-4.
"""
import pandas as pd
import numpy as np
from typing import Dict

def calculate_psi(expected: pd.Series, actual: pd.Series, bins=10) -> float:
    """Calculate Population Stability Index"""
    expected_percents = expected.value_counts(normalize=True, bins=bins).sort_index()
    actual_percents = actual.value_counts(normalize=True, bins=bins).sort_index()
    
    psi = 0
    for i in expected_percents.index:
        psi += (actual_percents[i] - expected_percents[i]) * np.log(actual_percents[i] / expected_percents[i])
    return psi

def check_data_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict:
    """Check drift for key features"""
    drift_results = {}
    
    drift_features = ['Height', 'Weight', 'BMI', 'Age', 'Disposals', 'Clearances']
    
    for feature in drift_features:
        if feature in reference_df.columns:
            psi = calculate_psi(reference_df[feature], current_df[feature])
            drift_results[feature] = {
                'psi': psi,
                'drift_detected': psi > 0.1
            }
    
    return drift_results
