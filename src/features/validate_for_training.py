"""Data validation script for ML Engineer (Member A)"""
import pandas as pd
import numpy as np

def validate_training_data(filepath="data/processed/afl_features_latest.csv"):
    """Run validation checks before model training"""
    
    df = pd.read_csv(filepath)
    
    print("="*50)
    print("DATA VALIDATION FOR MODEL TRAINING")
    print("="*50)
    
    # Basic checks
    print(f"\n✓ Dataset shape: {df.shape}")
    print(f"✓ No missing values: {df.isnull().sum().sum() == 0}")
    print(f"✓ Years: {df['Year'].min()} - {df['Year'].max()}")
    
    # Target variable distribution
    print(f"\nTarget variable (Total_Score):")
    print(f"  - Mean: {df['Total_Score'].mean():.2f}")
    print(f"  - Median: {df['Total_Score'].median():.2f}")
    print(f"  - Zero values: {(df['Total_Score'] == 0).mean()*100:.1f}%")
    
    # Position-specific recommendations
    print(f"\nPosition-specific recommendations:")
    print(f"  - Forward: predict Total_Score")
    print(f"  - Midfield: predict Clearances")
    print(f"  - Ruck: predict HitOuts")
    print(f"  - Defender: predict Rebounds")
    
    # Feature types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nFeature types:")
    print(f"  - Numeric features: {len(numeric_cols)}")
    print(f"  - Categorical features: {len(categorical_cols)}")
    print(f"  - Categorical: {categorical_cols}")
    
    # Sample data
    print(f"\nSample data (first 3 rows):")
    print(df[['PlayerId', 'Year', 'PrimaryPosition', 'Total_Score']].head(3))
    
    return df

if __name__ == "__main__":
    df = validate_training_data()
