# -*- coding: utf-8 -*-
"""
Split Dataset into Train/Test/Validation Sets
Creates separate CSV files for reproducible ML experiments
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

print("=" * 60)
print("DATASET SPLITTING UTILITY")
print("=" * 60)

# Load the complete dataset
data_file = '../data/processed/real_data.csv'
print(f"\n[Step 1/4] Loading dataset from: {data_file}")
df = pd.read_csv(data_file)
print(f"   Total records: {len(df)}")
print(f"   Columns: {list(df.columns)}")

# Define features and targets
feature_columns = ['sqrtA', 'a', 'e', 'i', 'RAAN', 'omega', 'M']
target_columns = ['X_Error', 'Y_Error', 'Z_Error', 'Clock_Error']

print(f"\n   Features (7): {feature_columns}")
print(f"   Targets (4): {target_columns}")

# Extract features and targets
X = df[feature_columns]
y = df[target_columns]

# Split ratios: 80% train, 15% test, 5% validation
print("\n[Step 2/4] Splitting dataset...")
print("   Train: 80% | Test: 15% | Validation: 5%")

# First split: 80% train, 20% temp (test + validation)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Second split: 15% test (75% of temp), 5% validation (25% of temp)
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.20 = 0.05
)

print(f"\n   Training samples:   {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
print(f"   Test samples:       {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")
print(f"   Validation samples: {len(X_val)} ({len(X_val)/len(df)*100:.1f}%)")
print(f"   Total:              {len(X_train) + len(X_test) + len(X_val)}")

# Create output directory
output_dir = '../data/splits'
os.makedirs(output_dir, exist_ok=True)

# Save train set
print("\n[Step 3/4] Saving split datasets...")
train_df = pd.concat([X_train, y_train], axis=1)
train_df.to_csv(f'{output_dir}/train.csv', index=False)
print(f"   [OK] Saved: {output_dir}/train.csv ({len(train_df)} records)")

# Save test set
test_df = pd.concat([X_test, y_test], axis=1)
test_df.to_csv(f'{output_dir}/test.csv', index=False)
print(f"   [OK] Saved: {output_dir}/test.csv ({len(test_df)} records)")

# Save validation set
val_df = pd.concat([X_val, y_val], axis=1)
val_df.to_csv(f'{output_dir}/validation.csv', index=False)
print(f"   [OK] Saved: {output_dir}/validation.csv ({len(val_df)} records)")

# Generate statistics
print("\n[Step 4/4] Dataset Statistics:")
print("\n" + "=" * 60)
print("TRAINING SET STATISTICS")
print("=" * 60)
print(train_df[feature_columns].describe())

print("\n" + "=" * 60)
print("TRAINING SET TARGET STATISTICS")
print("=" * 60)
print(train_df[target_columns].describe())

print("\n" + "=" * 60)
print("TEST SET STATISTICS")
print("=" * 60)
print(test_df[feature_columns].describe())

print("\n" + "=" * 60)
print("VALIDATION SET STATISTICS")
print("=" * 60)
print(val_df[feature_columns].describe())

print("\n" + "=" * 60)
print("SPLIT COMPLETE!")
print("=" * 60)
print("\nDatasets saved to:")
print(f"  - {output_dir}/train.csv")
print(f"  - {output_dir}/test.csv")
print(f"  - {output_dir}/validation.csv")
print("\nYou can now use these split datasets for training!")
print("=" * 60)
