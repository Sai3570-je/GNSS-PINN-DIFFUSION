# -*- coding: utf-8 -*-
"""
Example: How to use the trained GNSS error prediction model
This script shows how to load the model and make predictions on new satellite data
"""

import tensorflow as tf
import pickle
import numpy as np
import pandas as pd

print("="*70)
print("GNSS ERROR PREDICTION - EXAMPLE USAGE")
print("="*70)

# ============================================================================
# STEP 1: LOAD THE TRAINED MODEL AND SCALER
# ============================================================================
print("\n[STEP 1] Loading trained model and scaler...")

model_path = '../03_models/gnss_error_model.keras'
scaler_path = '../03_models/scaler.pkl'

model = tf.keras.models.load_model(model_path)
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

print("[OK] Model and scaler loaded successfully!")

# ============================================================================
# STEP 2: PREPARE SAMPLE DATA
# ============================================================================
print("\n[STEP 2] Preparing sample satellite data...")

# Example Keplerian elements for GPS satellite
# Format: [sqrtA, a, e, i, RAAN, omega, M]
sample_data = np.array([
    [5154.024, 26563970, 0.0089, 55.13, 180.5, 90.3, 45.2],  # Satellite 1
    [5153.998, 26562640, 0.0092, 55.20, 175.2, 88.5, 50.1],  # Satellite 2
    [5154.015, 26563500, 0.0085, 55.05, 182.3, 91.2, 43.8],  # Satellite 3
])

# Create DataFrame for better visualization
sample_df = pd.DataFrame(sample_data, columns=[
    'sqrtA', 'a', 'e', 'i', 'RAAN', 'omega', 'M'
])

print("\nSample Input Data (Keplerian Elements):")
print(sample_df)

# ============================================================================
# STEP 3: NORMALIZE FEATURES
# ============================================================================
print("\n[STEP 3] Normalizing features...")

sample_data_scaled = scaler.transform(sample_data)
print("[OK] Features normalized")

# ============================================================================
# STEP 4: MAKE PREDICTIONS
# ============================================================================
print("\n[STEP 4] Making predictions...")

predictions = model.predict(sample_data_scaled, verbose=0)

# Create results DataFrame
results_df = pd.DataFrame(predictions, columns=[
    'X_Error', 'Y_Error', 'Z_Error', 'Clock_Error'
])

print("\n" + "="*70)
print("PREDICTION RESULTS")
print("="*70)

for i in range(len(predictions)):
    print(f"\nSatellite {i+1}:")
    print(f"  X_Error:     {predictions[i][0]:>10.2f} meters")
    print(f"  Y_Error:     {predictions[i][1]:>10.2f} meters")
    print(f"  Z_Error:     {predictions[i][2]:>10.2f} meters")
    print(f"  Clock_Error: {predictions[i][3]:>10.2f} meters")
    
    # Calculate total position error
    pos_error = np.sqrt(predictions[i][0]**2 + predictions[i][1]**2 + predictions[i][2]**2)
    print(f"  Total Position Error: {pos_error:>10.2f} meters")

# ============================================================================
# STEP 5: SAVE PREDICTIONS (OPTIONAL)
# ============================================================================
print("\n[STEP 5] Saving predictions...")

# Combine input and predictions
output_df = pd.concat([sample_df, results_df], axis=1)
output_file = '../04_results/sample_predictions.csv'
output_df.to_csv(output_file, index=False)

print(f"[OK] Predictions saved to: {output_file}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"  Samples processed: {len(sample_data)}")
print(f"  Model used: {model_path}")
print(f"  Average X_Error: {predictions[:, 0].mean():.2f} m")
print(f"  Average Y_Error: {predictions[:, 1].mean():.2f} m")
print(f"  Average Z_Error: {predictions[:, 2].mean():.2f} m")
print(f"  Average Clock_Error: {predictions[:, 3].mean():.2f} m")
print("="*70)
print("\nTo use this model with your own data:")
print("1. Prepare Keplerian elements: [sqrtA, a, e, i, RAAN, omega, M]")
print("2. Normalize using the scaler: scaler.transform(data)")
print("3. Predict: model.predict(normalized_data)")
print("4. Output: [X_Error, Y_Error, Z_Error, Clock_Error] in meters")
print("="*70)
