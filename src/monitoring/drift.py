"""Drift detection for model features (Phase 6)"""
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return obj

def calculate_psi(expected, actual, bins=10):
    """
    Calculate Population Stability Index (PSI)
    Measures how much a distribution has shifted over time
    PSI < 0.1: No significant drift
    PSI 0.1-0.2: Moderate drift
    PSI > 0.2: Significant drift
    """
    # Create bins based on expected distribution
    expected_bins = np.percentile(expected, np.linspace(0, 100, bins+1))
    
    # Bin the data
    expected_counts, _ = np.histogram(expected, bins=expected_bins)
    actual_counts, _ = np.histogram(actual, bins=expected_bins)
    
    # Convert to percentages
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    # Add small epsilon to avoid division by zero
    expected_pct = np.clip(expected_pct, 1e-10, 1)
    actual_pct = np.clip(actual_pct, 1e-10, 1)
    
    # Calculate PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    return float(psi)  # Convert to Python float

def check_data_drift(reference_df, current_df, features):
    """Check drift for key features"""
    drift_results = {}
    
    for feature in features:
        if feature in reference_df.columns and feature in current_df.columns:
            # Remove NaN values
            ref_vals = reference_df[feature].dropna().values
            cur_vals = current_df[feature].dropna().values
            
            if len(ref_vals) > 0 and len(cur_vals) > 0:
                psi = calculate_psi(ref_vals, cur_vals)
                
                # Determine severity
                if psi < 0.1:
                    severity = "low"
                elif psi < 0.2:
                    severity = "moderate"
                else:
                    severity = "high"
                
                drift_results[feature] = {
                    "psi": psi,
                    "drift_detected": bool(psi > 0.1),
                    "severity": severity,
                    "reference_mean": float(ref_vals.mean()),
                    "current_mean": float(cur_vals.mean()),
                    "reference_std": float(ref_vals.std()),
                    "current_std": float(cur_vals.std()),
                    "reference_min": float(ref_vals.min()),
                    "current_min": float(cur_vals.min()),
                    "reference_max": float(ref_vals.max()),
                    "current_max": float(cur_vals.max())
                }
    
    return drift_results

def generate_drift_report(data_path="data/processed/afl_features_latest.csv", 
                          output_path="reports/drift/drift_report.json"):
    """Generate complete drift report"""
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Split by year (train: 2012-2022, current: 2023-2025)
    train_df = df[df['Year'] <= 2022]
    current_df = df[df['Year'] >= 2023]
    
    # Features to monitor
    drift_features = ['Height', 'Weight', 'BMI', 'Age', 'Disposals', 
                      'Clearances', 'Marks', 'Tackles', 'Inside50s']
    
    # Calculate drift
    drift_results = check_data_drift(train_df, current_df, drift_features)
    
    # Create report with native Python types
    report = {
        "timestamp": datetime.now().isoformat(),
        "train_period": {
            "start": int(train_df['Year'].min()),
            "end": int(train_df['Year'].max()),
            "rows": int(len(train_df))
        },
        "current_period": {
            "start": int(current_df['Year'].min()),
            "end": int(current_df['Year'].max()),
            "rows": int(len(current_df))
        },
        "drift_results": drift_results,
        "summary": {
            "total_features_checked": len(drift_features),
            "features_with_drift": int(sum(1 for v in drift_results.values() if v['drift_detected'])),
            "high_severity": int(sum(1 for v in drift_results.values() if v.get('severity') == 'high')),
            "moderate_severity": int(sum(1 for v in drift_results.values() if v.get('severity') == 'moderate')),
            "low_severity": int(sum(1 for v in drift_results.values() if v.get('severity') == 'low'))
        }
    }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("="*50)
    print("DRIFT DETECTION REPORT")
    print("="*50)
    print(f"Train period: {report['train_period']['start']}-{report['train_period']['end']} ({report['train_period']['rows']} rows)")
    print(f"Current period: {report['current_period']['start']}-{report['current_period']['end']} ({report['current_period']['rows']} rows)")
    print("\nDrift Results:")
    for feature, result in drift_results.items():
        status = "⚠️ DRIFT" if result['drift_detected'] else "✅ Stable"
        print(f"  {feature}: PSI={result['psi']:.4f} - {status} ({result['severity']})")
    
    print(f"\nSummary: {report['summary']['features_with_drift']}/{report['summary']['total_features_checked']} features show drift")
    print(f"Report saved to: {output_path}")
    
    return report

if __name__ == "__main__":
    generate_drift_report()