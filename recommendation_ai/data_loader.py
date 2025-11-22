"""
CSV Data Loader and Utilities for Crop Recommendation Dataset

Handles:
- CSV loading and validation
- Data preprocessing and cleaning
- Feature scaling
- Train/test splitting
- Data augmentation
- Statistics and visualization helpers
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class CropDataLoader:
    """Load and preprocess crop recommendation CSV data"""

    EXPECTED_COLUMNS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
    FEATURE_COLUMNS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    LABEL_COLUMN = 'label'

    # Valid ranges for features
    VALID_RANGES = {
        'N': (0, 150),
        'P': (0, 150),
        'K': (0, 250),
        'temperature': (-10, 50),
        'humidity': (0, 100),
        'ph': (0, 14),
        'rainfall': (0, 500),
    }

    def __init__(self, verbose=True):
        """
        Initialize data loader

        Args:
            verbose: Print progress information
        """
        self.verbose = verbose
        self.df = None
        self.scaler = StandardScaler()

    @staticmethod
    def validate_csv_format(csv_path):
        """
        Validate CSV has correct format before loading

        Args:
            csv_path: Path to CSV file

        Returns:
            Boolean indicating if format is valid
        """
        try:
            df_sample = pd.read_csv(csv_path, nrows=1)
            expected = set(CropDataLoader.EXPECTED_COLUMNS)
            actual = set(df_sample.columns)

            if expected != actual:
                missing = expected - actual
                extra = actual - expected
                if missing:
                    print(f"❌ Missing columns: {missing}")
                if extra:
                    print(f"⚠️  Extra columns: {extra}")
                return False

            return True
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return False

    def load(self, csv_path, remove_outliers=True):
        """
        Load CSV data with validation

        Args:
            csv_path: Path to CSV file
            remove_outliers: Remove data points outside valid ranges

        Returns:
            Loaded DataFrame
        """
        if self.verbose:
            print(f"📂 Loading CSV: {csv_path}")

        # Validate format
        if not self.validate_csv_format(csv_path):
            raise ValueError(f"Invalid CSV format in {csv_path}")

        # Load data
        self.df = pd.read_csv(csv_path)

        if self.verbose:
            print(f"✅ Loaded {len(self.df)} rows")

        # Clean data
        self._clean_data()

        # Remove outliers
        if remove_outliers:
            initial_count = len(self.df)
            self._remove_outliers()
            if self.verbose:
                print(f"🔍 Removed {initial_count - len(self.df)} outliers")

        if self.verbose:
            print(f"✅ Final dataset: {len(self.df)} rows")

        return self.df

    def _clean_data(self):
        """Remove null values and duplicates"""
        # Remove null values
        null_count = self.df.isnull().sum().sum()
        if null_count > 0:
            if self.verbose:
                print(f"🧹 Removing {null_count} null values...")
            self.df = self.df.dropna()

        # Remove duplicates
        dup_count = len(self.df) - len(self.df.drop_duplicates())
        if dup_count > 0:
            if self.verbose:
                print(f"🧹 Removing {dup_count} duplicate rows...")
            self.df = self.df.drop_duplicates()

        # Reset index
        self.df = self.df.reset_index(drop=True)

    def _remove_outliers(self):
        """Remove data points outside valid ranges"""
        initial_len = len(self.df)

        for col, (min_val, max_val) in self.VALID_RANGES.items():
            self.df = self.df[(self.df[col] >= min_val) & (self.df[col] <= max_val)]

        return initial_len - len(self.df)

    def get_statistics(self):
        """Get dataset statistics"""
        stats = {
            'total_samples': len(self.df),
            'unique_crops': self.df[self.LABEL_COLUMN].nunique(),
            'crop_distribution': self.df[self.LABEL_COLUMN].value_counts().to_dict(),
        }

        # Feature statistics
        for col in self.FEATURE_COLUMNS:
            stats[f'{col}_mean'] = self.df[col].mean()
            stats[f'{col}_std'] = self.df[col].std()
            stats[f'{col}_min'] = self.df[col].min()
            stats[f'{col}_max'] = self.df[col].max()

        return stats

    def print_statistics(self):
        """Print formatted statistics"""
        stats = self.get_statistics()

        print(f"\n📊 Dataset Statistics:")
        print(f"   Total Samples: {stats['total_samples']}")
        print(f"   Unique Crops: {stats['unique_crops']}")
        print(f"\n🌾 Crop Distribution:")
        for crop, count in sorted(stats['crop_distribution'].items(), key=lambda x: x[1], reverse=True):
            pct = (count / stats['total_samples']) * 100
            print(f"   {crop:20s}: {count:4d} samples ({pct:5.1f}%)")

        print(f"\n📈 Feature Statistics:")
        for col in self.FEATURE_COLUMNS:
            mean = stats[f'{col}_mean']
            std = stats[f'{col}_std']
            min_v = stats[f'{col}_min']
            max_v = stats[f'{col}_max']
            print(f"   {col:15s}: mean={mean:8.2f}, std={std:6.2f}, min={min_v:8.2f}, max={max_v:8.2f}")

    def get_features_and_labels(self):
        """Get features and labels for training"""
        X = self.df[self.FEATURE_COLUMNS].values
        y = self.df[self.LABEL_COLUMN].values
        return X, y

    def split_by_crop(self, crop_name):
        """Get all samples for specific crop"""
        return self.df[self.df[self.LABEL_COLUMN] == crop_name]

    def balance_dataset(self, target_samples=None):
        """
        Balance dataset by resampling crops to equal representation

        Args:
            target_samples: Target samples per crop (default: median)
        """
        if target_samples is None:
            # Use median as target
            counts = self.df[self.LABEL_COLUMN].value_counts()
            target_samples = int(counts.median())

        if self.verbose:
            print(f"⚖️  Balancing dataset to {target_samples} samples per crop...")

        balanced_dfs = []
        for crop in self.df[self.LABEL_COLUMN].unique():
            crop_df = self.df[self.df[self.LABEL_COLUMN] == crop]

            if len(crop_df) < target_samples:
                # Upsample
                crop_df = crop_df.sample(n=target_samples, replace=True, random_state=42)
            else:
                # Downsample
                crop_df = crop_df.sample(n=target_samples, replace=False, random_state=42)

            balanced_dfs.append(crop_df)

        self.df = pd.concat(balanced_dfs, ignore_index=True)

        if self.verbose:
            print(f"✅ Balanced dataset: {len(self.df)} rows")

        return self.df

    def export_statistics_csv(self, output_path):
        """Export statistics to CSV"""
        stats = self.get_statistics()

        rows = []
        for crop, count in stats['crop_distribution'].items():
            rows.append({
                'crop': crop,
                'count': count,
                'percentage': (count / stats['total_samples']) * 100
            })

        stats_df = pd.DataFrame(rows)
        stats_df.to_csv(output_path, index=False)

        if self.verbose:
            print(f"📊 Statistics exported to {output_path}")


class DataPreprocessor:
    """Handle feature scaling and normalization"""

    def __init__(self, scaling_method='standard'):
        """
        Initialize preprocessor

        Args:
            scaling_method: 'standard' or 'minmax'
        """
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")

    def fit(self, X):
        """Fit scaler on training data"""
        self.scaler.fit(X)
        return self

    def transform(self, X):
        """Transform data using fitted scaler"""
        return self.scaler.transform(X)

    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.scaler.fit_transform(X)

    def get_scaler(self):
        """Get the underlying scaler object"""
        return self.scaler


if __name__ == '__main__':
    # Example usage
    print("🌾 Crop Data Loader - Example Usage\n")

    # Create sample loader
    loader = CropDataLoader(verbose=True)

    # Example: Load CSV
    # df = loader.load('data/crop_data.csv')
    # loader.print_statistics()

    # Example: Balance dataset
    # df_balanced = loader.balance_dataset(target_samples=100)
    # loader.print_statistics()

    # Example: Export statistics
    # loader.export_statistics_csv('data/crop_statistics.csv')

    print("✅ Data loader ready. Use in your training script:")
    print("   from data_loader import CropDataLoader")
    print("   loader = CropDataLoader()")
    print("   df = loader.load('data/crop_data.csv')")
