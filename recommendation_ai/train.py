"""
Training script for Crop Recommendation AI Model

This script trains a Random Forest classifier on crop recommendation data.
Supports custom CSV datasets with sensor readings and crop labels.

CSV Format:
    N,P,K,temperature,humidity,ph,rainfall,label
    90,42,43,20.88,81,6.02,202.9,rice
    ...

Usage:
    python train.py --csv data.csv --model models/crop_recommender.pkl
    python train.py --csv data.csv --test-size 0.2 --n-estimators 200
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


class CropRecommendationTrainer:
    """Trains Random Forest model for crop recommendation"""

    FEATURE_COLUMNS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    LABEL_COLUMN = 'label'

    def __init__(self, random_state=42, test_size=0.2, n_estimators=100, verbose=True):
        """
        Initialize trainer

        Args:
            random_state: Random seed for reproducibility
            test_size: Fraction of data for testing
            n_estimators: Number of trees in RandomForest
            verbose: Print progress information
        """
        self.random_state = random_state
        self.test_size = test_size
        self.n_estimators = n_estimators
        self.verbose = verbose

        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.metrics = {}

    def load_data(self, csv_path):
        """
        Load and validate CSV data

        Args:
            csv_path: Path to CSV file

        Returns:
            DataFrame with validated data
        """
        if self.verbose:
            print(f"📂 Loading data from {csv_path}...")

        df = pd.read_csv(csv_path)

        # Validate columns
        missing_cols = set(self.FEATURE_COLUMNS + [self.LABEL_COLUMN]) - set(df.columns)
        if missing_cols:
            raise ValueError(f"❌ Missing columns: {missing_cols}")

        # Check for missing values
        if df.isnull().sum().sum() > 0:
            print(f"⚠️  Found missing values:\n{df.isnull().sum()}")
            print("Removing rows with missing values...")
            df = df.dropna()

        if self.verbose:
            print(f"✅ Loaded {len(df)} samples")
            print(f"📊 Unique crops: {df[self.LABEL_COLUMN].nunique()}")
            print(f"🌾 Crops: {', '.join(sorted(df[self.LABEL_COLUMN].unique()))}")

        return df

    def preprocess_data(self, df):
        """
        Preprocess and split data

        Args:
            df: Input DataFrame

        Returns:
            Preprocessed data with train/test split
        """
        if self.verbose:
            print(f"\n🔄 Preprocessing data...")

        # Extract features and labels
        X = df[self.FEATURE_COLUMNS].values
        y = df[self.LABEL_COLUMN].values

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.crop_labels = self.label_encoder.classes_

        if self.verbose:
            print(f"✅ Features: {self.FEATURE_COLUMNS}")
            print(f"✅ Features shape: {X.shape}")
            print(f"✅ Labels shape: {y_encoded.shape}")

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_encoded
        )

        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        if self.verbose:
            print(f"✅ Train set: {len(self.X_train)} samples")
            print(f"✅ Test set: {len(self.X_test)} samples")
            print(f"✅ Feature scaling: StandardScaler")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self):
        """Train Random Forest model"""
        if self.X_train is None:
            raise ValueError("❌ Data not preprocessed. Call preprocess_data() first")

        if self.verbose:
            print(f"\n🤖 Training Random Forest (n_estimators={self.n_estimators})...")

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPUs
            verbose=1 if self.verbose else 0
        )

        self.model.fit(self.X_train, self.y_train)

        if self.verbose:
            print(f"✅ Model trained successfully")

    def evaluate(self):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("❌ Model not trained. Call train() first")

        if self.verbose:
            print(f"\n📊 Evaluating model...")

        # Predictions
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)

        # Metrics
        self.metrics = {
            'train_accuracy': accuracy_score(self.y_train, y_pred_train),
            'test_accuracy': accuracy_score(self.y_test, y_pred_test),
            'train_precision': precision_score(self.y_train, y_pred_train, average='weighted', zero_division=0),
            'test_precision': precision_score(self.y_test, y_pred_test, average='weighted', zero_division=0),
            'train_recall': recall_score(self.y_train, y_pred_train, average='weighted', zero_division=0),
            'test_recall': recall_score(self.y_test, y_pred_test, average='weighted', zero_division=0),
            'train_f1': f1_score(self.y_train, y_pred_train, average='weighted', zero_division=0),
            'test_f1': f1_score(self.y_test, y_pred_test, average='weighted', zero_division=0),
        }

        # Cross-validation
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
        self.metrics['cv_mean'] = cv_scores.mean()
        self.metrics['cv_std'] = cv_scores.std()

        if self.verbose:
            print(f"\n📈 Performance Metrics:")
            print(f"   Train Accuracy: {self.metrics['train_accuracy']:.4f}")
            print(f"   Test Accuracy:  {self.metrics['test_accuracy']:.4f}")
            print(f"   Test F1-Score:  {self.metrics['test_f1']:.4f}")
            print(f"   Test Precision: {self.metrics['test_precision']:.4f}")
            print(f"   Test Recall:    {self.metrics['test_recall']:.4f}")
            print(f"   CV Score:       {self.metrics['cv_mean']:.4f} (+/- {self.metrics['cv_std']:.4f})")

            # Classification report
            print(f"\n📋 Classification Report:")
            report = classification_report(self.y_test, y_pred_test, target_names=self.crop_labels)
            print(report)

            # Feature importance
            print(f"\n🎯 Feature Importance:")
            importances = self.model.feature_importances_
            for feat, imp in sorted(zip(self.FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True):
                print(f"   {feat:15s}: {imp:.4f}")

    def save_model(self, model_path):
        """
        Save trained model and scalers

        Args:
            model_path: Path to save model
        """
        if self.model is None:
            raise ValueError("❌ No model to save. Train model first")

        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        # Save scaler
        scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        # Save label encoder
        encoder_path = model_path.replace('.pkl', '_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)

        # Save metadata
        metadata = {
            'feature_columns': self.FEATURE_COLUMNS,
            'label_column': self.LABEL_COLUMN,
            'crop_labels': self.crop_labels.tolist(),
            'n_estimators': self.n_estimators,
            'metrics': self.metrics,
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test),
        }
        meta_path = model_path.replace('.pkl', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        if self.verbose:
            print(f"\n💾 Model saved to: {model_path}")
            print(f"   Scaler: {scaler_path}")
            print(f"   Encoder: {encoder_path}")
            print(f"   Metadata: {meta_path}")

    def run_training_pipeline(self, csv_path, model_path='models/crop_recommender.pkl'):
        """
        Complete training pipeline: load → preprocess → train → evaluate → save

        Args:
            csv_path: Path to training CSV
            model_path: Path to save trained model
        """
        print("=" * 60)
        print("🌾 CROP RECOMMENDATION MODEL TRAINING")
        print("=" * 60)

        try:
            # Load and preprocess
            df = self.load_data(csv_path)
            self.preprocess_data(df)

            # Train and evaluate
            self.train()
            self.evaluate()

            # Save
            self.save_model(model_path)

            print("\n" + "=" * 60)
            print("✅ TRAINING COMPLETED SUCCESSFULLY")
            print("=" * 60)

        except Exception as e:
            print(f"\n❌ Error during training: {str(e)}")
            raise


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Train Crop Recommendation AI Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --csv data/crop_data.csv
  python train.py --csv data/crop_data.csv --model models/my_model.pkl
  python train.py --csv data/crop_data.csv --n-estimators 200 --test-size 0.15
        """
    )

    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file with training data')
    parser.add_argument('--model', type=str, default='models/crop_recommender.pkl',
                        help='Path to save trained model')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of trees in RandomForest')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data for testing (0-1)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no-verbose', action='store_true',
                        help='Disable verbose output')

    args = parser.parse_args()

    # Validate CSV exists
    if not os.path.exists(args.csv):
        print(f"❌ CSV file not found: {args.csv}")
        sys.exit(1)

    # Validate test size
    if not 0 < args.test_size < 1:
        print(f"❌ test-size must be between 0 and 1")
        sys.exit(1)

    # Run training
    trainer = CropRecommendationTrainer(
        random_state=args.random_state,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        verbose=not args.no_verbose
    )

    trainer.run_training_pipeline(args.csv, args.model)


if __name__ == '__main__':
    main()
