"""
Feature engineering pipeline for AFL player performance prediction.
Converts raw CSV files to engineered features for model training.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


class AFLDataPreprocessor:
    """
    Complete data preprocessing pipeline for AFL data.
    Handles player data, game data, and stats data integration.
    """
    
    def __init__(self, raw_data_path: str = "data/raw", processed_data_path: str = "data/processed"):
        self.raw_path = raw_data_path
        self.processed_path = processed_data_path
        self.df_final = None
        
        # Ensure directories exist
        os.makedirs(self.raw_path, exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)
    
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load raw CSV files"""
        print("Loading raw data...")
        
        df_game = pd.read_csv(os.path.join(self.raw_path, "games.csv"))
        df_player = pd.read_csv(os.path.join(self.raw_path, "players.csv"))
        df_stats = pd.read_csv(os.path.join(self.raw_path, "stats.csv"))
        
        print(f"  - games.csv: {len(df_game)} rows")
        print(f"  - players.csv: {len(df_player)} rows")
        print(f"  - stats.csv: {len(df_stats)} rows")
        
        return df_game, df_player, df_stats
    
    # ========== PLAYER DATA PROCESSING ==========
    
    def extract_birth_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract birth year from Dob column"""
        df = df.copy()
        if "Dob" in df.columns:
            df["Dob"] = pd.to_datetime(df["Dob"], errors="coerce")
            df["BirthYear"] = df["Dob"].dt.year.astype("Int64")
        return df
    
    def finalize_player_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize positions using Ruck-Priority rule.
        Any hybrid with 'Ruck' becomes 'Ruck', otherwise take first listed position.
        """
        df_clean = df[df['Position'] != 'no position'].copy()
        
        def map_to_single(pos):
            if pd.isna(pos):
                return 'Unknown'
            if 'Ruck' in pos:
                return 'Ruck'
            return pos.split(',')[0].strip()
        
        df_clean['PrimaryPosition'] = df_clean['Position'].apply(map_to_single)
        return df_clean
    
    def add_players_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate BMI from Height and Weight"""
        df = df.copy()
        if "Height" in df.columns and "Weight" in df.columns:
            valid_mask = (df["Height"] > 0) & (df["Height"].notna())
            h_m = df.loc[valid_mask, "Height"] / 100.0
            df.loc[valid_mask, "BMI"] = df.loc[valid_mask, "Weight"] / (h_m ** 2)
        return df
    
    def process_player_data(self, df_player: pd.DataFrame) -> pd.DataFrame:
        """Run full player data pipeline"""
        print("\nProcessing player data...")
        
        df_processed = (df_player
                        .pipe(self.extract_birth_year)
                        .pipe(self.finalize_player_positions)
                        .pipe(self.add_players_derived_features))
        
        # Drop raw columns
        cols_to_drop = [c for c in ["Dob", "Position"] if c in df_processed.columns]
        df_processed = df_processed.drop(columns=cols_to_drop)
        
        print(f"  - Players processed: {len(df_processed)}")
        return df_processed
    
    # ========== GAME DATA PROCESSING ==========
    
    def clean_game_data(self, df_game: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with missing weather data"""
        print("\nCleaning game data...")
        original_count = len(df_game)
        df_cleaned = df_game.dropna(subset=['MaxTemp', 'MinTemp', 'Rainfall']).copy()
        print(f"  - Removed {original_count - len(df_cleaned)} rows with missing weather data")
        return df_cleaned
    
    def refine_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create weather-derived features: AvgTemp, TempRange, IsRainy"""
        df = df.copy()
        df['AvgTemp'] = (df['MaxTemp'] + df['MinTemp']) / 2
        df['TempRange'] = df['MaxTemp'] - df['MinTemp']
        df['IsRainy'] = (df['Rainfall'] > 0).astype(int)
        return df
    
    def process_game_data(self, df_game: pd.DataFrame) -> pd.DataFrame:
        """Run full game data pipeline"""
        print("\nProcessing game data...")
        df_processed = (df_game
                        .pipe(self.clean_game_data)
                        .pipe(self.refine_weather_features))
        print(f"  - Games processed: {len(df_processed)}")
        return df_processed
    
    # ========== STATS DATA PROCESSING ==========
    
    def process_stats_data(self, df_stats: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare stats data"""
        print("\nProcessing stats data...")
        
        # Remove duplicates
        original_count = len(df_stats)
        df_processed = df_stats.drop_duplicates()
        print(f"  - Removed {original_count - len(df_processed)} duplicate rows")
        
        # Drop Subs column (mostly missing, not informative)
        if 'Subs' in df_processed.columns:
            df_processed = df_processed.drop(columns=['Subs'])
        
        # Drop Round and GameNumber for model simplicity
        if 'Round' in df_processed.columns:
            df_processed = df_processed.drop(columns=['Round'])
        if 'GameNumber' in df_processed.columns:
            df_processed = df_processed.drop(columns=['GameNumber'])
        
        # Remove redundant/leaky/sparse features
        columns_to_remove = [
            'Kicks', 'Handballs',
            'ContestedPossessions', 'UncontestedPossessions',
            'PlayerName',
            'BrownlowVotes',
            'Bounces'
        ]
        existing_cols = [col for col in columns_to_remove if col in df_processed.columns]
        df_processed = df_processed.drop(columns=existing_cols)
        
        print(f"  - Stats processed: {len(df_processed)} rows, {len(df_processed.columns)} columns")
        return df_processed
    
    # ========== DATA INTEGRATION ==========
    
    def merge_datasets(self, df_stats: pd.DataFrame, df_player: pd.DataFrame, df_game: pd.DataFrame) -> pd.DataFrame:
        """Merge all three datasets"""
        print("\nMerging datasets...")
        
        # Merge stats with player data
        df_merged = df_stats.merge(df_player, on='PlayerId', how='inner')
        print(f"  - After player merge: {len(df_merged)} rows")
        
        # Merge with game data (only needed columns)
        game_cols = ['GameId', 'AwayTeam', 'AvgTemp', 'TempRange', 'IsRainy']
        df_merged = df_merged.merge(df_game[game_cols], on='GameId', how='inner')
        print(f"  - After game merge: {len(df_merged)} rows")
        
        # Calculate Age
        if 'BirthYear' in df_merged.columns:
            df_merged['Age'] = df_merged['Year'] - df_merged['BirthYear']
            df_merged = df_merged.drop(columns=['BirthYear'])
        
        # Create Total Score
        df_merged['Total_Score'] = (df_merged['Goals'] * 6) + (df_merged['Behinds'] * 1)
        
        # Remove weight=0 outliers
        original_count = len(df_merged)
        df_merged = df_merged[df_merged['Weight'] != 0]
        if len(df_merged) < original_count:
            print(f"  - Removed {original_count - len(df_merged)} rows with Weight=0")
        
        self.df_final = df_merged
        print(f"  - Final dataset: {len(df_merged)} rows, {len(df_merged.columns)} columns")
        
        return df_merged
    
    # ========== POSITION-SPECIFIC SCALING PIPELINE ==========
    
    def get_position_scaling_pipeline(self, df: pd.DataFrame, position_filter: str, target_col: str, 
                                       test_size: float = 0.2, random_state: int = 42):
        """
        Returns train/test split with position-specific scaling.
        """
        df_pos = df[df['PrimaryPosition'] == position_filter].copy()
        if df_pos.empty:
            raise ValueError(f"No data found for position: {position_filter}")
        
        # Define features
        numeric_features = [
            'Height', 'Weight', 'BMI', 'Age', 'AvgTemp', 'TempRange',
            'Disposals', 'Marks', 'Goals', 'Behinds', 'HitOuts', 'Tackles',
            'Rebounds', 'Inside50s', 'Clearances', 'Clangers', 'Frees',
            'FreesAgainst', 'ContestedMarks', 'MarksInside50', 'OnePercenters',
            'GoalAssists', '%Played'
        ]
        # Filter to available columns
        numeric_features = [f for f in numeric_features if f in df_pos.columns]
        passthrough_features = ['IsRainy'] if 'IsRainy' in df_pos.columns else []
        all_cols = numeric_features + passthrough_features
        
        X = df_pos[all_cols]
        y = df_pos[target_col]
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scaling pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', RobustScaler(), numeric_features),
                ('pass', 'passthrough', passthrough_features)
            ]
        )
        
        scaling_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        X_train_scaled = scaling_pipeline.fit_transform(X_train)
        X_test_scaled = scaling_pipeline.transform(X_test)
        
        # Convert back to DataFrame
        X_train_final = pd.DataFrame(X_train_scaled, columns=all_cols)
        X_test_final = pd.DataFrame(X_test_scaled, columns=all_cols)
        
        print(f"--- Pipeline Ready: Data processed for {position_filter} ---")
        return X_train_final, X_test_final, y_train, y_test, scaling_pipeline
    
    # ========== MAIN EXECUTION ==========
    
    def run_full_pipeline(self, save_output: bool = True) -> pd.DataFrame:
        print("="*60)
        print("AFL DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # Load
        df_game, df_player, df_stats = self.load_raw_data()
        
        # Process each dataset
        df_player_processed = self.process_player_data(df_player)
        df_game_processed = self.process_game_data(df_game)
        df_stats_processed = self.process_stats_data(df_stats)
        
        # Merge
        df_final = self.merge_datasets(df_stats_processed, df_player_processed, df_game_processed)
        
        if save_output:
            # 1. Save main feature file
            output_file = os.path.join(self.processed_path, "afl_features_latest.csv")
            df_final.to_csv(output_file, index=False)
            print(f"\n✓ Saved feature dataset to: {output_file}")
            
            # 2. Create reports directory if it doesn't exist
            os.makedirs("reports/data_quality", exist_ok=True)
            
            # 3. Save data quality report (JSON)
            import json
            quality_report = {
                'total_rows': len(df_final),
                'total_columns': len(df_final.columns),
                'years_covered': f"{df_final['Year'].min()} - {df_final['Year'].max()}",
                'unique_players': df_final['PlayerId'].nunique(),
                'unique_games': df_final['GameId'].nunique(),
                'missing_values': df_final.isnull().sum().to_dict(),
                'duplicate_rows': int(df_final.duplicated().sum()),
                'position_distribution': df_final['PrimaryPosition'].value_counts().to_dict()
            }
            
            with open("reports/data_quality/data_quality_report.json", 'w') as f:
                json.dump(quality_report, f, indent=2)
            print(f"✓ Saved data quality report to: reports/data_quality/data_quality_report.json")
            
            # 4. Save column summary (CSV)
            column_summary = pd.DataFrame({
                'column': df_final.columns,
                'dtype': df_final.dtypes.values,
                'null_count': df_final.isnull().sum().values,
                'null_percentage': (df_final.isnull().sum() / len(df_final) * 100).values,
                'unique_values': df_final.nunique().values
            })
            column_summary.to_csv("reports/data_quality/column_summary.csv", index=False)
            print(f"✓ Saved column summary to: reports/data_quality/column_summary.csv")
            
            # 5. Save basic dataset summary (TXT)
            with open("reports/data_quality/dataset_summary.txt", 'w') as f:
                f.write("="*60 + "\n")
                f.write("AFL DATASET SUMMARY\n")
                f.write("="*60 + "\n\n")
                f.write(f"Total rows: {len(df_final):,}\n")
                f.write(f"Total columns: {len(df_final.columns)}\n")
                f.write(f"Years: {df_final['Year'].min()} - {df_final['Year'].max()}\n\n")
                f.write("Position distribution:\n")
                for pos, count in df_final['PrimaryPosition'].value_counts().items():
                    f.write(f"  - {pos}: {count:,} ({count/len(df_final)*100:.1f}%)\n")
        
        return df_final

# ========== ENTRY POINTS ==========

def build_features():
    """Command-line entry point for feature engineering"""
    preprocessor = AFLDataPreprocessor()
    df = preprocessor.run_full_pipeline(save_output=True)
    print("\n✓ Feature engineering complete!")
    return df


if __name__ == "__main__":
    build_features()