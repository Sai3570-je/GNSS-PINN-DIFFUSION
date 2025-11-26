# -*- coding: utf-8 -*-
"""
Optimized GNSS Error Prediction Model with UTC Time
Clean, minimal training script using errors + timestamp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pickle
import os

print("=" * 70)
print("  GNSS ERROR PREDICTION - OPTIMIZED MODEL")
print("  Features: Time-based features from UTC timestamp only")
print("=" * 70)

# ============================================================================
# STEP 1: Load Pre-Split Data with UTC Time
# ============================================================================
print("\n[STEP 1] Loading pre-split datasets with UTC time...")

df_train = pd.read_csv('../data/splits/train_errors_utc.csv')
df_test = pd.read_csv('../data/splits/test_errors_utc.csv')
df_val = pd.read_csv('../data/splits/validation_errors_utc.csv')

print(f"[OK] Training:   {len(df_train)} samples")
print(f"     Test:       {len(df_test)} samples")
print(f"     Validation: {len(df_val)} samples")

# ============================================================================
# STEP 2: Feature Engineering from UTC Time
# ============================================================================
print("\n[STEP 2] Engineering features from UTC timestamp...")

def engineer_time_features(df):
    """Extract temporal features from UTC timestamp"""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Time components
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    
    # Cyclical encoding for periodic patterns
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Unix timestamp (seconds since epoch) - normalized
    df['unix_time'] = df['timestamp'].astype(np.int64) // 10**9
    df['unix_time_norm'] = (df['unix_time'] - df['unix_time'].min()) / (df['unix_time'].max() - df['unix_time'].min())
    
    return df

df_train = engineer_time_features(df_train)
df_test = engineer_time_features(df_test)
df_val = engineer_time_features(df_val)

print(f"[OK] Features engineered!")

# Define features and targets
time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'doy_sin', 'doy_cos', 'unix_time_norm']
target_cols = ['X_Error', 'Y_Error', 'Z_Error', 'Clock_Error']

print(f"     Features: {len(time_features)} (time-based)")
print(f"     Targets:  {len(target_cols)} error values")

# ============================================================================
# STEP 3: Prepare Data
# ============================================================================
print("\n[STEP 3] Preparing data...")

X_train = df_train[time_features].values
X_test = df_test[time_features].values
X_val = df_val[time_features].values

y_train = df_train[target_cols].values
y_test = df_test[target_cols].values
y_val = df_val[target_cols].values

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

print(f"[OK] Data prepared and normalized")
print(f"     X_train shape: {X_train_scaled.shape}")
print(f"     y_train shape: {y_train.shape}")

# ============================================================================
# STEP 4: Build Optimized Model
# ============================================================================
print("\n[STEP 4] Building optimized neural network...")

model = keras.Sequential([
    layers.Input(shape=(len(time_features),)),
    
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.2),
    
    layers.Dense(4, activation='linear')  # X, Y, Z, Clock errors
], name='GNSS_Error_UTC_Model')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("[OK] Model created!")
model.summary()

# ============================================================================
# STEP 5: Train Model
# ============================================================================
print("\n[STEP 5] Training model...")
print("=" * 70)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='../03_models/best_utc_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=150,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print("\n[OK] Training completed!")

# ============================================================================
# STEP 6: Evaluate on Test Set
# ============================================================================
print("\n[STEP 6] Evaluating on test set...")
print("=" * 70)

y_pred = model.predict(X_test_scaled, verbose=0)

print("\nTEST SET RESULTS:")
print("=" * 70)

error_names = ['X_Error', 'Y_Error', 'Z_Error', 'Clock_Error']
for i, name in enumerate(error_names):
    mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    
    print(f"\n{name}:")
    print(f"  MAE:  {mae:.2f} meters")
    print(f"  RMSE: {rmse:.2f} meters")
    print(f"  R²:   {r2:.4f}")

# Overall metrics
mae_overall = mean_absolute_error(y_test, y_pred)
rmse_overall = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nOVERALL:")
print(f"  MAE:  {mae_overall:.2f} meters")
print(f"  RMSE: {rmse_overall:.2f} meters")

# ============================================================================
# STEP 7: Save Model and Scaler
# ============================================================================
print("\n[STEP 7] Saving model and scaler...")

model.save('../03_models/gnss_utc_model.keras')

with open('../03_models/scaler_utc.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('../03_models/features_utc.pkl', 'wb') as f:
    pickle.dump(time_features, f)

print("[OK] Saved:")
print("     - gnss_utc_model.keras")
print("     - best_utc_model.keras")
print("     - scaler_utc.pkl")
print("     - features_utc.pkl")

# ============================================================================
# STEP 8: Visualization
# ============================================================================
print("\n[STEP 8] Creating visualizations...")

os.makedirs('../04_results/utc_model', exist_ok=True)

# Training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
axes[1].set_title('Model MAE', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE (meters)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../04_results/utc_model/training_history.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: training_history.png")
plt.close()

# Predictions vs Actual
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, (name, ax) in enumerate(zip(error_names, axes.flat)):
    ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(y_test[:, i].min(), y_pred[:, i].min())
    max_val = max(y_test[:, i].max(), y_pred[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    
    mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    
    ax.set_xlabel(f'Actual {name} (m)', fontsize=11)
    ax.set_ylabel(f'Predicted {name} (m)', fontsize=11)
    ax.set_title(f'{name}\nMAE: {mae:.2f}m | R²: {r2:.4f}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../04_results/utc_model/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: predictions_vs_actual.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)
print(f"\nDataset:")
print(f"  Training:   {len(df_train)} samples")
print(f"  Test:       {len(df_test)} samples")
print(f"  Validation: {len(df_val)} samples")

print(f"\nFeatures:")
print(f"  Time-based: {len(time_features)}")

print(f"\nModel:")
print(f"  Architecture: 128 → 64 → 32 → 4")
print(f"  Epochs:       {len(history.history['loss'])}")
print(f"  Best val MAE: {min(history.history['val_mae']):.2f} meters")

print(f"\nTest Performance:")
print(f"  Overall MAE:  {mae_overall:.2f} meters")
print(f"  Overall RMSE: {rmse_overall:.2f} meters")

print("\n" + "=" * 70)
print("[SUCCESS] UTC-based model training completed!")
print("=" * 70)
