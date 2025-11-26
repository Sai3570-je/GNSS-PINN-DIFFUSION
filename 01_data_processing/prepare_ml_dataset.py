"""
Prepare comprehensive ML dataset for GNSS error prediction
Combines Keplerian elements (features) with error values (targets)
"""

import pandas as pd
import numpy as np

print("🔄 Preparing ML-ready dataset...")

# Load Keplerian elements (features)
print("\n1️⃣ Loading Keplerian elements...")
kepler_df = pd.read_csv('GNSS_kepler_elements_clean.csv')
print(f"   ✓ Loaded {len(kepler_df)} Keplerian records")
print(f"   Columns: {list(kepler_df.columns)}")

# Load error data (targets)
print("\n2️⃣ Loading error data...")
error_df = pd.read_csv('gnss_error_output.csv')
print(f"   ✓ Loaded {len(error_df)} error records")

# Load real clock errors
print("\n3️⃣ Loading real clock errors...")
clock_df = pd.read_csv('real_gnss_clock_errors.csv')
print(f"   ✓ Loaded {len(clock_df)} clock records")

# Prepare clock data
clock_df = clock_df[['timestamp', 'sat_id', 'total_clock_error_meters']].copy()
clock_df.rename(columns={'total_clock_error_meters': 'Real_Clock_Error'}, inplace=True)

# Convert timestamps
kepler_df['timestamp'] = pd.to_datetime(kepler_df['timestamp'])
error_df['timestamp'] = pd.to_datetime(error_df['timestamp'])
clock_df['timestamp'] = pd.to_datetime(clock_df['timestamp'])

print("\n4️⃣ Merging datasets...")
# First merge: Keplerian + position errors
merged = pd.merge(
    kepler_df,
    error_df[['timestamp', 'sat_id', 'X_Error', 'Y_Error', 'Z_Error']],
    on=['timestamp', 'sat_id'],
    how='inner'
)
print(f"   After Keplerian + position errors: {len(merged)} records")

# Second merge: Add real clock errors
merged = pd.merge(
    merged,
    clock_df,
    on=['timestamp', 'sat_id'],
    how='inner'
)
print(f"   After adding real clock errors: {len(merged)} records")

# Rename clock error
merged.rename(columns={'Real_Clock_Error': 'Clock_Error'}, inplace=True)

# Drop NaN values
print("\n5️⃣ Cleaning data...")
print(f"   Before: {len(merged)} records")
merged = merged.dropna()
print(f"   After: {len(merged)} records")

# Save the comprehensive dataset
output_file = 'real_data.csv'
merged.to_csv(output_file, index=False)

print(f"\n✅ SUCCESS!")
print(f"💾 Saved to: {output_file}")
print(f"📊 Total records: {len(merged)}")
print(f"\n📋 Columns ({len(merged.columns)}):")
print(f"   Features: {[col for col in merged.columns if col not in ['X_Error', 'Y_Error', 'Z_Error', 'Clock_Error']]}")
print(f"   Targets: ['X_Error', 'Y_Error', 'Z_Error', 'Clock_Error']")

# Display sample
print(f"\n🔍 Sample of first 3 records:")
print(merged.head(3))

print(f"\n📈 Dataset Statistics:")
print(merged.describe())
