# -*- coding: utf-8 -*-
"""
GNSS Satellite Error Prediction - Dense Neural Network
Trains using train_errors.csv, tests on test_errors.csv, validates on validation_errors.csv
Predicts: X_Error, Y_Error, Z_Error, Clock_Error (no input features - autoencoder style)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pickle
import sys
import io

# Set UTF-8 encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 70)
print("  GNSS SATELLITE ERROR PREDICTION MODEL (ERROR-BASED TRAINING)")
print("=" * 70)

# ============================================================================
# STEP 1: LOAD PRE-SPLIT DATASETS
# ============================================================================
print("\n[STEP 1] Loading pre-split datasets...")

# Load training data
train_df = pd.read_csv('../data/splits/train_errors.csv')
print(f"[OK] Training set loaded: {len(train_df)} samples")

# Load test data
test_df = pd.read_csv('../data/splits/test_errors.csv')
print(f"[OK] Test set loaded: {len(test_df)} samples")

# Load validation data
val_df = pd.read_csv('../data/splits/validation_errors.csv')
print(f"[OK] Validation set loaded: {len(val_df)} samples")

print(f"\nTotal samples: {len(train_df) + len(test_df) + len(val_df)}")
print(f"  Training:   {len(train_df)} ({len(train_df)/(len(train_df)+len(test_df)+len(val_df))*100:.1f}%)")
print(f"  Test:       {len(test_df)} ({len(test_df)/(len(train_df)+len(test_df)+len(val_df))*100:.1f}%)")
print(f"  Validation: {len(val_df)} ({len(val_df)/(len(train_df)+len(test_df)+len(val_df))*100:.1f}%)")

# ============================================================================
# STEP 2: PREPARE DATA
# ============================================================================
print("\n[STEP 2] Preparing data...")

# All columns are targets (errors)
error_columns = ['X_Error', 'Y_Error', 'Z_Error', 'Clock_Error']
print(f"Target columns (4): {error_columns}")

# Extract arrays
y_train = train_df[error_columns].values
y_test = test_df[error_columns].values
y_val = val_df[error_columns].values

print(f"\n[OK] Data prepared successfully!")
print(f"  Training targets shape: {y_train.shape}")
print(f"  Test targets shape: {y_test.shape}")
print(f"  Validation targets shape: {y_val.shape}")

# ============================================================================
# STEP 3: FEATURE ENGINEERING (Create synthetic features from errors)
# ============================================================================
print("\n[STEP 3] Creating features from error patterns...")

def create_features_from_errors(errors):
    """
    Create features from error values for pattern learning
    Features include: raw errors, squared errors, ratios, magnitudes
    """
    x_err, y_err, z_err, clk_err = errors[:, 0], errors[:, 1], errors[:, 2], errors[:, 3]
    
    features = []
    
    # Raw error values
    features.extend([x_err, y_err, z_err, clk_err])
    
    # Position error magnitude
    pos_magnitude = np.sqrt(x_err**2 + y_err**2 + z_err**2)
    features.append(pos_magnitude)
    
    # Error ratios (avoid division by zero)
    xy_ratio = np.where(np.abs(y_err) > 1e-6, x_err / y_err, 0)
    xz_ratio = np.where(np.abs(z_err) > 1e-6, x_err / z_err, 0)
    yz_ratio = np.where(np.abs(z_err) > 1e-6, y_err / z_err, 0)
    features.extend([xy_ratio, xz_ratio, yz_ratio])
    
    # Squared errors (emphasize larger errors)
    features.extend([x_err**2, y_err**2, z_err**2])
    
    # Clock error normalized by position error
    clk_pos_ratio = np.where(pos_magnitude > 1e-6, clk_err / pos_magnitude, 0)
    features.append(clk_pos_ratio)
    
    # Sign patterns
    features.extend([np.sign(x_err), np.sign(y_err), np.sign(z_err), np.sign(clk_err)])
    
    return np.column_stack(features)

# Create features
X_train = create_features_from_errors(y_train)
X_test = create_features_from_errors(y_test)
X_val = create_features_from_errors(y_val)

print(f"[OK] Features created!")
print(f"  Number of features: {X_train.shape[1]}")
print(f"  Feature categories:")
print(f"    - Raw errors: 4")
print(f"    - Position magnitude: 1")
print(f"    - Error ratios: 3")
print(f"    - Squared errors: 3")
print(f"    - Clock/position ratio: 1")
print(f"    - Sign patterns: 4")

# ============================================================================
# STEP 4: NORMALIZE FEATURES
# ============================================================================
print("\n[STEP 4] Normalizing features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

print("[OK] Features normalized (mean=0, std=1)")

# ============================================================================
# STEP 5: BUILD DEEP DENSE NEURAL NETWORK
# ============================================================================
print("\n[STEP 5] Building deep dense neural network...")

model = keras.Sequential([
    # Input layer
    layers.Input(shape=(X_train_scaled.shape[1],)),
    
    # Hidden layer 1: 256 neurons
    layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # Hidden layer 2: 128 neurons
    layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # Hidden layer 3: 64 neurons
    layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    # Hidden layer 4: 32 neurons
    layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.2),
    
    # Output layer: 4 neurons (X_Error, Y_Error, Z_Error, Clock_Error)
    layers.Dense(4)  # Linear activation for regression
], name='GNSS_Error_Prediction_Model')

print("[OK] Model architecture:")
model.summary()

# ============================================================================
# STEP 6: COMPILE MODEL
# ============================================================================
print("\n[STEP 6] Compiling model...")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

print("[OK] Model compiled successfully!")
print("  Optimizer: Adam (lr=0.001)")
print("  Loss: Mean Squared Error (MSE)")
print("  Metrics: Mean Absolute Error (MAE)")

# ============================================================================
# STEP 7: SET UP CALLBACKS
# ============================================================================
print("\n[STEP 7] Setting up training callbacks...")

# Early stopping: stop if validation loss doesn't improve for 15 epochs
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# Model checkpoint: save best model
checkpoint = keras.callbacks.ModelCheckpoint(
    '../03_models/best_gnss_error_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Learning rate reduction
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

print("[OK] Callbacks configured:")
print("  - Early Stopping (patience=15)")
print("  - Model Checkpoint (saves best model)")
print("  - Learning Rate Reduction (factor=0.5, patience=5)")

# ============================================================================
# STEP 8: TRAIN MODEL
# ============================================================================
print("\n[STEP 8] Training model...")
print("=" * 70)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=150,
    batch_size=32,
    callbacks=[early_stopping, checkpoint, reduce_lr],
    verbose=1
)

print("=" * 70)
print("[OK] Training completed!")

# ============================================================================
# STEP 9: EVALUATE ON TEST SET
# ============================================================================
print("\n[STEP 9] Evaluating model on test set...")

# Make predictions
y_pred = model.predict(X_test_scaled, verbose=0)

# Calculate metrics for each error type
print("\n" + "=" * 70)
print("TEST SET RESULTS")
print("=" * 70)

for i, error_name in enumerate(error_columns):
    mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    
    print(f"\n{error_name}:")
    print(f"  MAE:  {mae:.2f} meters")
    print(f"  RMSE: {rmse:.2f} meters")
    print(f"  R²:   {r2:.4f}")

# Overall metrics
overall_mae = mean_absolute_error(y_test, y_pred)
overall_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n{'='*70}")
print(f"OVERALL TEST PERFORMANCE:")
print(f"  Mean Absolute Error (MAE):  {overall_mae:.2f} meters")
print(f"  Root Mean Squared Error:     {overall_rmse:.2f} meters")
print(f"{'='*70}")

# ============================================================================
# STEP 10: SAMPLE PREDICTIONS
# ============================================================================
print("\n[STEP 10] Sample predictions (first 5 test samples):")
print("=" * 70)

for i in range(min(5, len(y_test))):
    print(f"\nSample {i+1}:")
    print(f"  Actual:    X={y_test[i][0]:8.2f}m  Y={y_test[i][1]:8.2f}m  Z={y_test[i][2]:8.2f}m  Clk={y_test[i][3]:10.2f}m")
    print(f"  Predicted: X={y_pred[i][0]:8.2f}m  Y={y_pred[i][1]:8.2f}m  Z={y_pred[i][2]:8.2f}m  Clk={y_pred[i][3]:10.2f}m")
    print(f"  Error:     X={abs(y_test[i][0]-y_pred[i][0]):8.2f}m  Y={abs(y_test[i][1]-y_pred[i][1]):8.2f}m  Z={abs(y_test[i][2]-y_pred[i][2]):8.2f}m  Clk={abs(y_test[i][3]-y_pred[i][3]):10.2f}m")

# ============================================================================
# STEP 11: SAVE MODEL AND SCALER
# ============================================================================
print("\n[STEP 11] Saving model and scaler...")

model.save('../03_models/gnss_error_model.keras')
print("[OK] Model saved: ../03_models/gnss_error_model.keras")

with open('../03_models/scaler_errors.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("[OK] Scaler saved: ../03_models/scaler_errors.pkl")

# ============================================================================
# STEP 12: PLOT TRAINING HISTORY
# ============================================================================
print("\n[STEP 12] Plotting training history...")

plt.figure(figsize=(15, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss (MSE)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mean_absolute_error'], label='Training MAE', linewidth=2)
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE', linewidth=2)
plt.title('Model MAE', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MAE (meters)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../04_results/training_history_errors.png', dpi=300, bbox_inches='tight')
print("[OK] Training history saved: ../04_results/training_history_errors.png")

# ============================================================================
# STEP 13: FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)
print(f"Training samples:   {len(y_train)}")
print(f"Validation samples: {len(y_val)}")
print(f"Test samples:       {len(y_test)}")
print(f"Total epochs:       {len(history.history['loss'])}")
print(f"Best val loss:      {min(history.history['val_loss']):.2f}")
print(f"Final test MAE:     {overall_mae:.2f} meters")
print(f"Final test RMSE:    {overall_rmse:.2f} meters")
print("=" * 70)

print("\n[SUCCESS] Model training completed successfully!")
print("\nFiles created:")
print("  - ../03_models/gnss_error_model.keras")
print("  - ../03_models/best_gnss_error_model.keras")
print("  - ../03_models/scaler_errors.pkl")
print("  - ../04_results/training_history_errors.png")
print("\n" + "=" * 70)
