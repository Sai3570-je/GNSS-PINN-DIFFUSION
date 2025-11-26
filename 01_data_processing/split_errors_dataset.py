# -*- coding: utf-8 -*-
"""
Split gnss_error_data.csv into Train/Test/Validation Sets
This uses the error-only dataset (3,510 records)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

print("=" * 60)
print("SPLITTING gnss_error_data.csv FOR TRAINING")
print("=" * 60)

# Load the error dataset
data_file = '../04_results/gnss_error_data.csv'
print(f"\n[Step 1/4] Loading dataset: {data_file}")
df = pd.read_csv(data_file)
print(f"   Total records: {len(df)}")
print(f"   Columns: {list(df.columns)}")

# All columns are targets (errors)
target_columns = ['X_Error', 'Y_Error', 'Z_Error', 'Clock_Error']
print(f"   Target columns (4): {target_columns}")

# Split ratios: 80% train, 15% test, 5% validation
print("\n[Step 2/4] Splitting dataset...")
print("   Train: 80% | Test: 15% | Validation: 5%")

# First split: 80% train, 20% temp (test + validation)
df_train, df_temp = train_test_split(df, test_size=0.20, random_state=42)

# Second split: 15% test (75% of temp), 5% validation (25% of temp)
df_test, df_val = train_test_split(df_temp, test_size=0.25, random_state=42)

print(f"\n   Training samples:   {len(df_train)} ({len(df_train)/len(df)*100:.1f}%)")
print(f"   Test samples:       {len(df_test)} ({len(df_test)/len(df)*100:.1f}%)")
print(f"   Validation samples: {len(df_val)} ({len(df_val)/len(df)*100:.1f}%)")
print(f"   Total:              {len(df_train) + len(df_test) + len(df_val)}")

# Create output directory
output_dir = '../data/splits'
os.makedirs(output_dir, exist_ok=True)

# Save datasets
print("\n[Step 3/4] Saving split datasets...")

df_train.to_csv(f'{output_dir}/train_errors.csv', index=False)
print(f"   [OK] Saved: {output_dir}/train_errors.csv ({len(df_train)} records)")

df_test.to_csv(f'{output_dir}/test_errors.csv', index=False)
print(f"   [OK] Saved: {output_dir}/test_errors.csv ({len(df_test)} records)")

df_val.to_csv(f'{output_dir}/validation_errors.csv', index=False)
print(f"   [OK] Saved: {output_dir}/validation_errors.csv ({len(df_val)} records)")

# Generate statistics
print("\n[Step 4/4] Dataset Statistics:")
print("\n" + "=" * 60)
print("TRAINING SET STATISTICS")
print("=" * 60)
print(df_train.describe())

print("\n" + "=" * 60)
print("TEST SET STATISTICS")
print("=" * 60)
print(df_test.describe())

print("\n" + "=" * 60)
print("VALIDATION SET STATISTICS")
print("=" * 60)
print(df_val.describe())

print("\n" + "=" * 60)
print("SPLIT COMPLETE!")
print("=" * 60)
print("\nError datasets saved to:")
print(f"  - {output_dir}/train_errors.csv")
print(f"  - {output_dir}/test_errors.csv")
print(f"  - {output_dir}/validation_errors.csv")
print("\nNote: This dataset contains only error values (no features).")
print("Use 'data/splits/train.csv' for full dataset with features.")
print("=" * 60)
