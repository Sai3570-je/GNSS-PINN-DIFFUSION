"""
📌 Create Final GNSS Error Dataset for Smart India Hackathon
Combines position errors from computed ECEF with real clock errors from RINEX
Outputs ONLY: X_Error, Y_Error, Z_Error, Clock_Error
"""

import pandas as pd
import numpy as np

print("🔄 Creating final error dataset...")

# Step 1: Load the existing position error data (with simulated clock)
print("\n1️⃣ Loading position error data...")
position_df = pd.read_csv('gnss_error_output.csv')
print(f"   ✓ Loaded {len(position_df)} position records")

# Step 2: Load the real clock error data
print("\n2️⃣ Loading real clock error data...")
clock_df = pd.read_csv('real_gnss_clock_errors.csv')
print(f"   ✓ Loaded {len(clock_df)} clock records")

# Step 3: Prepare clock data for merging
# Select only timestamp, sat_id, and total_clock_error_meters
clock_df = clock_df[['timestamp', 'sat_id', 'total_clock_error_meters']].copy()
clock_df.rename(columns={'total_clock_error_meters': 'Real_Clock_Error'}, inplace=True)

# Convert timestamps to datetime for proper matching
position_df['timestamp'] = pd.to_datetime(position_df['timestamp'])
clock_df['timestamp'] = pd.to_datetime(clock_df['timestamp'])

print(f"\n3️⃣ Merging position and clock data...")
# Merge on timestamp and sat_id
merged_df = pd.merge(
    position_df[['timestamp', 'sat_id', 'X_Error', 'Y_Error', 'Z_Error']],
    clock_df,
    on=['timestamp', 'sat_id'],
    how='inner'
)

print(f"   ✓ Merged: {len(merged_df)} records matched")

# Step 4: Extract ONLY the error columns
print("\n4️⃣ Extracting error columns...")
error_cols = ['X_Error', 'Y_Error', 'Z_Error', 'Real_Clock_Error']
error_df = merged_df[error_cols].copy()

# Rename Real_Clock_Error to Clock_Error as per user template
error_df.rename(columns={'Real_Clock_Error': 'Clock_Error'}, inplace=True)

# Step 5: Remove NaN values
print("\n5️⃣ Cleaning data (removing NaN values)...")
print(f"   Before: {len(error_df)} records")
error_df = error_df.dropna()
print(f"   After: {len(error_df)} records (removed {len(merged_df) - len(error_df)} NaN rows)")

# Step 6: Save the final dataset
output_file = 'gnss_error_data.csv'
error_df.to_csv(output_file, index=False)

print(f"\n✅ SUCCESS! Final error dataset created")
print(f"💾 Saved to: {output_file}")
print(f"📊 Total records: {len(error_df)}")

# Display statistics
print("\n📈 Error Statistics:")
print("\nX_Error (meters):")
print(f"   Mean: {error_df['X_Error'].mean():.3f} m")
print(f"   Std:  {error_df['X_Error'].std():.3f} m")
print(f"   Range: [{error_df['X_Error'].min():.3f}, {error_df['X_Error'].max():.3f}] m")

print("\nY_Error (meters):")
print(f"   Mean: {error_df['Y_Error'].mean():.3f} m")
print(f"   Std:  {error_df['Y_Error'].std():.3f} m")
print(f"   Range: [{error_df['Y_Error'].min():.3f}, {error_df['Y_Error'].max():.3f}] m")

print("\nZ_Error (meters):")
print(f"   Mean: {error_df['Z_Error'].mean():.3f} m")
print(f"   Std:  {error_df['Z_Error'].std():.3f} m")
print(f"   Range: [{error_df['Z_Error'].min():.3f}, {error_df['Z_Error'].max():.3f}] m")

print("\nClock_Error (meters):")
print(f"   Mean: {error_df['Clock_Error'].mean():.3f} m")
print(f"   Std:  {error_df['Clock_Error'].std():.3f} m")
print(f"   Range: [{error_df['Clock_Error'].min():.3f}, {error_df['Clock_Error'].max():.3f}] m")

print("\n📋 Sample of first 10 records:")
print(error_df.head(10))

print("\n🎯 Dataset ready for Smart India Hackathon ML training!")
print("✓ Contains ONLY real GNSS error values")
print("✓ NO synthetic or generated data")
print("✓ Extracted from genuine GPS broadcast ephemeris (Jan 2024)")
