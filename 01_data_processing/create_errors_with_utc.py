# -*- coding: utf-8 -*-
"""
Create Error Datasets with UTC Time and Split for Training
Extracts errors + timestamp from real_data.csv and splits properly
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

print("=" * 70)
print("CREATING ERROR DATASETS WITH UTC TIME")
print("=" * 70)

# Load full dataset with timestamp
print("\n[Step 1/4] Loading real_data.csv...")
df = pd.read_csv('../data/processed/real_data.csv')
print(f"   Total records: {len(df)}")
print(f"   Columns: {list(df.columns)}")

# Select only timestamp and error columns
print("\n[Step 2/4] Extracting timestamp and errors...")
error_data = df[['timestamp', 'X_Error', 'Y_Error', 'Z_Error', 'Clock_Error']].copy()
print(f"   Extracted columns: {list(error_data.columns)}")
print(f"   Total records: {len(error_data)}")

# Convert timestamp to datetime
error_data['timestamp'] = pd.to_datetime(error_data['timestamp'])

# Display sample
print(f"\n   Sample data:")
print(error_data.head(3))

# Split: 80% train, 15% test, 5% validation
print("\n[Step 3/4] Splitting data (80% train, 15% test, 5% validation)...")

# First split: 80% train, 20% temp
df_train, df_temp = train_test_split(error_data, test_size=0.20, random_state=42, shuffle=True)

# Second split: 15% test (75% of temp), 5% validation (25% of temp)
df_test, df_val = train_test_split(df_temp, test_size=0.25, random_state=42, shuffle=True)

print(f"   Training:   {len(df_train)} samples ({len(df_train)/len(error_data)*100:.1f}%)")
print(f"   Test:       {len(df_test)} samples ({len(df_test)/len(error_data)*100:.1f}%)")
print(f"   Validation: {len(df_val)} samples ({len(df_val)/len(error_data)*100:.1f}%)")

# Save datasets
print("\n[Step 4/4] Saving datasets...")
output_dir = '../data/splits'
os.makedirs(output_dir, exist_ok=True)

df_train.to_csv(f'{output_dir}/train_errors_utc.csv', index=False)
df_test.to_csv(f'{output_dir}/test_errors_utc.csv', index=False)
df_val.to_csv(f'{output_dir}/validation_errors_utc.csv', index=False)

print(f"   Saved: {output_dir}/train_errors_utc.csv")
print(f"   Saved: {output_dir}/test_errors_utc.csv")
print(f"   Saved: {output_dir}/validation_errors_utc.csv")

# Display statistics
print("\n" + "=" * 70)
print("DATASET STATISTICS")
print("=" * 70)

for name, dataset in [('TRAIN', df_train), ('TEST', df_test), ('VALIDATION', df_val)]:
    print(f"\n{name} SET:")
    print(f"  Samples: {len(dataset)}")
    print(f"  Time range: {dataset['timestamp'].min()} to {dataset['timestamp'].max()}")
    print(f"  X_Error:     mean={dataset['X_Error'].mean():.2f}, std={dataset['X_Error'].std():.2f}")
    print(f"  Y_Error:     mean={dataset['Y_Error'].mean():.2f}, std={dataset['Y_Error'].std():.2f}")
    print(f"  Z_Error:     mean={dataset['Z_Error'].mean():.2f}, std={dataset['Z_Error'].std():.2f}")
    print(f"  Clock_Error: mean={dataset['Clock_Error'].mean():.2f}, std={dataset['Clock_Error'].std():.2f}")

print("\n" + "=" * 70)
print("SUCCESS! Error datasets with UTC time created.")
print("=" * 70)
