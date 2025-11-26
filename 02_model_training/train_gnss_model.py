# -*- coding: utf-8 -*-
"""
GNSS Satellite Error Prediction - Dense Neural Network
Trains a regression model to predict X_Error, Y_Error, Z_Error, Clock_Error
Dataset: Real GPS broadcast ephemeris data (Jan 2024)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import sys
import io

# Set UTF-8 encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 70)
print("  GNSS SATELLITE ERROR PREDICTION MODEL")
print("=" * 70)

# ============================================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================================
print("\n[STEP 1] Loading dataset...")
# Load the comprehensive dataset with features and targets
df = pd.read_csv('real_data.csv')
print(f"[OK] Dataset loaded successfully!")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# ============================================================================
# STEP 2: DATA CLEANING
# ============================================================================
print("\n[STEP 2] Cleaning data...")
print(f"  Before cleaning: {len(df)} rows")

# Check for missing values
missing_counts = df.isnull().sum()
if missing_counts.sum() > 0:
    print(f"  Missing values found:")
    for col, count in missing_counts[missing_counts > 0].items():
        print(f"    - {col}: {count}")
    
# Drop rows with missing values
df = df.dropna()
print(f"  After cleaning: {len(df)} rows")
print(f"  [OK] Removed {missing_counts.sum()} rows with missing values")

# ============================================================================
# STEP 3: PREPARE FEATURES AND TARGETS
# ============================================================================
print("\n[STEP 3] Preparing features and targets...")

# Define target columns
target_columns = ['X_Error', 'Y_Error', 'Z_Error', 'Clock_Error']

# Define columns to exclude from features (non-numeric)
exclude_columns = ['timestamp', 'sat_id'] + target_columns

# Check if target columns exist
missing_targets = [col for col in target_columns if col not in df.columns]
if missing_targets:
    print(f"[ERROR] Target columns missing: {missing_targets}")
    print(f"   Available columns: {list(df.columns)}")
    raise ValueError("Required target columns not found in dataset")

# Separate features and targets
feature_columns = [col for col in df.columns if col not in exclude_columns]
X = df[feature_columns].values
y = df[target_columns].values

print(f"  Features shape: {X.shape}")
print(f"  Targets shape: {y.shape}")
print(f"  Feature columns: {feature_columns}")
print(f"  Target columns: {target_columns}")

# ============================================================================
# STEP 4: NORMALIZE FEATURES
# ============================================================================
print("\n[STEP 4] Normalizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"  [OK] Features normalized using StandardScaler")
print(f"    Mean: {X_scaled.mean():.6f}")
print(f"    Std: {X_scaled.std():.6f}")

# ============================================================================
# STEP 5: SPLIT DATA (80% train, 15% test, 5% validation)
# ============================================================================
print("\n[STEP 5] Splitting data...")

# First split: 80% train+val, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42
)

# Second split: From remaining 85%, take ~5.88% for validation (which is 5% of total)
# 5 / 85 ≈ 0.0588
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.0588, random_state=42
)

print(f"  Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"  Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
print(f"  Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(df)*100:.1f}%)")

# ============================================================================
# STEP 6: BUILD DENSE NEURAL NETWORK
# ============================================================================
print("\n[STEP 6] Building neural network architecture...")

input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

model = keras.Sequential([
    # Input layer
    layers.Input(shape=(input_dim,)),
    
    # Hidden layer 1
    layers.Dense(128, activation='relu', name='hidden_1'),
    layers.Dropout(0.3, name='dropout_1'),
    
    # Hidden layer 2
    layers.Dense(64, activation='relu', name='hidden_2'),
    layers.Dropout(0.2, name='dropout_2'),
    
    # Hidden layer 3 (additional for better learning)
    layers.Dense(32, activation='relu', name='hidden_3'),
    layers.Dropout(0.2, name='dropout_3'),
    
    # Output layer (4 neurons for 4 error values)
    layers.Dense(output_dim, activation='linear', name='output')
])

print(f"  [OK] Model architecture created")
print(f"    Input dimension: {input_dim}")
print(f"    Output dimension: {output_dim}")
print("\nModel Summary:")
model.summary()

# ============================================================================
# STEP 7: COMPILE MODEL
# ============================================================================
print("\n[STEP 7] Compiling model...")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',  # Mean Squared Error
    metrics=['mae']  # Mean Absolute Error
)

print("  [OK] Model compiled")
print("    Optimizer: Adam (lr=0.001)")
print("    Loss: Mean Squared Error (MSE)")
print("    Metrics: Mean Absolute Error (MAE)")

# ============================================================================
# STEP 8: SETUP CALLBACKS
# ============================================================================
print("\n[STEP 8] Setting up callbacks...")

# Early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Model checkpoint to save best model
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_gnss_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=0
)

print("  [OK] Early stopping configured (patience=10)")
print("  [OK] Model checkpoint configured")

# ============================================================================
# STEP 9: TRAIN MODEL
# ============================================================================
print("\n[STEP 9] Training model...")
print("  Epochs: 100 (max)")
print("  Batch size: 64")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

print("\n[OK] Training completed!")
print(f"  Trained for {len(history.history['loss'])} epochs")

# ============================================================================
# STEP 10: EVALUATE ON TEST SET
# ============================================================================
print("\n[STEP 10] Evaluating model on test set...")

# Make predictions
y_pred = model.predict(X_test, verbose=0)

# Calculate metrics
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

print(f"\n*** TEST SET PERFORMANCE ***")
print(f"{'='*70}")
print(f"  Overall MSE: {test_loss:.4f}")
print(f"  Overall MAE: {test_mae:.4f} meters")
print(f"{'='*70}")

# Calculate MAE for each error type
print(f"\nMAE by Error Type:")
for i, target in enumerate(target_columns):
    mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
    print(f"  {target:12s}: MAE = {mae:8.4f} m  |  RMSE = {rmse:8.4f} m")

# ============================================================================
# STEP 11: SAMPLE PREDICTIONS
# ============================================================================
print(f"\n[STEP 11] Sample predictions (first 5 test samples):")
print(f"{'='*70}")

sample_indices = np.arange(5)
print(f"\n{'Target':<15} {'Actual':<20} {'Predicted':<20} {'Error':<15}")
print(f"{'-'*70}")

for i in sample_indices:
    print(f"\nSample {i+1}:")
    for j, target in enumerate(target_columns):
        actual = y_test[i, j]
        predicted = y_pred[i, j]
        error = abs(actual - predicted)
        print(f"  {target:<13} {actual:>12.4f} m     {predicted:>12.4f} m     {error:>10.4f} m")

# ============================================================================
# STEP 12: PLOT TRAINING HISTORY
# ============================================================================
print(f"\n[STEP 12] Generating training history plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot loss
axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss (MSE)', fontsize=12)
axes[0].set_title('Model Loss During Training', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot MAE
axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('MAE (meters)', fontsize=12)
axes[1].set_title('Model MAE During Training', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Training history saved to 'training_history.png'")

# ============================================================================
# STEP 13: SAVE MODEL AND SCALER
# ============================================================================
print(f"\n[STEP 13] Saving model and scaler...")

# Save the final model
model.save('gnss_error_model.keras')
print(f"  [OK] Model saved to 'gnss_error_model.keras'")

# Save the scaler for future predictions
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"  [OK] Scaler saved to 'scaler.pkl'")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print(f"\n{'='*70}")
print(f"*** MODEL TRAINING COMPLETE! ***")
print(f"{'='*70}")
print(f"\nFinal Results:")
print(f"  Dataset size: {len(df)} samples")
print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")
print(f"  Validation samples: {len(X_val)}")
print(f"  Features: {input_dim}")
print(f"  Targets: {output_dim}")
print(f"  Final test MAE: {test_mae:.4f} meters")
print(f"\nGenerated Files:")
print(f"  - gnss_error_model.keras (trained model)")
print(f"  - best_gnss_model.keras (best model checkpoint)")
print(f"  - scaler.pkl (feature scaler)")
print(f"  - training_history.png (training plots)")
print(f"\n*** Model is ready for GNSS error prediction! ***")
print(f"{'='*70}\n")
