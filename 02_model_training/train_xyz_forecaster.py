# -*- coding: utf-8 -*-
"""
GNSS Error Forecasting - Accurate Deep Learning Regressor
Dataset: Real GNSS error dataset with UTC timestamps
Targets: X_Error, Y_Error, Z_Error (3D position error model)
Features: timestamp (cyclical), satellite ID (one-hot), day of week, hour of day
Clock_Error: Separate model (not in this script)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import joblib
import os

print("=" * 80)
print("GNSS 3D ERROR FORECASTING - DEEP LEARNING REGRESSOR")
print("Targets: X_Error, Y_Error, Z_Error (Position Errors Only)")
print("=" * 80)

# ============================================================================
# STEP 1: Load UTC-Stamped Error Dataset
# ============================================================================
print("\n[STEP 1] Loading UTC-stamped error dataset...")

# Load the master dataset with all features
data = pd.read_csv("../data/processed/real_data.csv")
print(f"✓ Loaded {len(data)} records from real_data.csv")
print(f"  Columns: {list(data.columns)}")

# ============================================================================
# STEP 2: Feature Engineering
# ============================================================================
print("\n[STEP 2] Engineering features from timestamp and satellite ID...")

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Extract time-based features
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek
data['day_of_year'] = data['timestamp'].dt.dayofyear

# Cyclical encoding for hour (0-23)
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

# Cyclical encoding for day of week (0-6)
data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

# Cyclical encoding for day of year (1-365)
data['doy_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 365)
data['doy_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365)

# One-hot encode satellite ID
print(f"  Unique satellites: {data['sat_id'].nunique()}")
data_encoded = pd.get_dummies(data, columns=['sat_id'], prefix='sat')

print(f"✓ Features engineered")
print(f"  Time features: hour_sin, hour_cos, day_sin, day_cos, doy_sin, doy_cos")
print(f"  Satellite features: {data['sat_id'].nunique()} one-hot encoded columns")

# ============================================================================
# STEP 3: Prepare Features and Targets
# ============================================================================
print("\n[STEP 3] Preparing features and targets...")

# Define feature columns (time + satellite one-hot)
time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'doy_sin', 'doy_cos']
sat_features = [col for col in data_encoded.columns if col.startswith('sat_')]
feature_cols = time_features + sat_features

# Target columns (X, Y, Z errors only - NO Clock_Error)
target_cols = ['X_Error', 'Y_Error', 'Z_Error']

X = data_encoded[feature_cols].values
y = data_encoded[target_cols].values

print(f"✓ Features prepared")
print(f"  Feature shape: {X.shape}")
print(f"  Target shape:  {y.shape}")
print(f"  Total features: {len(feature_cols)} ({len(time_features)} time + {len(sat_features)} satellite)")

# ============================================================================
# STEP 4: Scale Features and Targets
# ============================================================================
print("\n[STEP 4] Scaling features and targets...")

# MinMaxScaler for features (scales to 0-1)
x_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X)

# StandardScaler for targets (zero mean, unit variance)
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)

print(f"✓ Scaling complete")
print(f"  Feature scaler: MinMaxScaler (range 0-1)")
print(f"  Target scaler:  StandardScaler (mean=0, std=1)")

# ============================================================================
# STEP 5: Split Data (80% Train, 15% Test, 5% Val)
# ============================================================================
print("\n[STEP 5] Splitting data (80% train, 15% test, 5% val)...")

# First split: 80% train, 20% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y_scaled, test_size=0.20, random_state=42, shuffle=True
)

# Second split: 15% test (75% of temp), 5% val (25% of temp)
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, shuffle=True
)

print(f"✓ Data split complete")
print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
print(f"  Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")

# ============================================================================
# STEP 6: Build Deep Neural Network Model
# ============================================================================
print("\n[STEP 6] Building Deep Neural Network...")

model = Sequential([
    # Input layer
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    
    # Hidden layer 1
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    # Hidden layer 2
    Dense(32, activation='relu'),
    Dropout(0.2),
    
    # Output layer (3 outputs: X, Y, Z errors)
    Dense(3, activation='linear')
], name='GNSS_XYZ_Error_Model')

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print("✓ Model built")
model.summary()

# ============================================================================
# STEP 7: Train Model with Early Stopping
# ============================================================================
print("\n[STEP 7] Training model...")
print("=" * 80)

# Create models directory if not exists
os.makedirs('../03_models', exist_ok=True)
os.makedirs('../04_results', exist_ok=True)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='../03_models/best_xyz_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print("\n✓ Training complete!")

# ============================================================================
# STEP 8: Evaluate on Test Set
# ============================================================================
print("\n[STEP 8] Evaluating on test set...")
print("=" * 80)

# Predict on test set
y_pred_scaled = model.predict(X_test, verbose=0)

# Inverse transform to original scale
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_test_original = y_scaler.inverse_transform(y_test)

# Calculate metrics for each dimension
print("\nTEST SET RESULTS:")
print("=" * 80)

error_names = ['X_Error', 'Y_Error', 'Z_Error']
for i, name in enumerate(error_names):
    mae = mean_absolute_error(y_test_original[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test_original[:, i], y_pred[:, i]))
    r2 = r2_score(y_test_original[:, i], y_pred[:, i])
    
    print(f"\n{name}:")
    print(f"  MAE:  {mae:.2f} meters")
    print(f"  RMSE: {rmse:.2f} meters")
    print(f"  R²:   {r2:.4f}")

# Overall metrics
mae_overall = mean_absolute_error(y_test_original, y_pred)
rmse_overall = np.sqrt(mean_squared_error(y_test_original, y_pred))

print(f"\nOVERALL (X, Y, Z combined):")
print(f"  MAE:  {mae_overall:.2f} meters")
print(f"  RMSE: {rmse_overall:.2f} meters")

# ============================================================================
# STEP 9: Save Model and Scalers
# ============================================================================
print("\n[STEP 9] Saving model and scalers...")

# Save final model
model.save('../03_models/gnss_xyz_model.keras')

# Save scalers
joblib.dump(x_scaler, '../03_models/x_scaler.pkl')
joblib.dump(y_scaler, '../03_models/y_scaler.pkl')

# Save feature names for future reference
joblib.dump(feature_cols, '../03_models/feature_columns.pkl')

print("✓ Saved files:")
print("  - gnss_xyz_model.keras (final model)")
print("  - best_xyz_model.keras (best checkpoint)")
print("  - x_scaler.pkl (feature scaler)")
print("  - y_scaler.pkl (target scaler)")
print("  - feature_columns.pkl (feature order)")

# ============================================================================
# STEP 10: Visualize Training History
# ============================================================================
print("\n[STEP 10] Creating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_title('Model Loss (MSE)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Mean Squared Error')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MAE plot
axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
axes[1].set_title('Model MAE', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Mean Absolute Error (meters)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../04_results/xyz_training_history.png', dpi=300, bbox_inches='tight')
print("✓ Saved: xyz_training_history.png")
plt.close()

# Prediction vs Actual scatter plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, name in enumerate(error_names):
    ax = axes[i]
    
    # Scatter plot
    ax.scatter(y_test_original[:, i], y_pred[:, i], alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(y_test_original[:, i].min(), y_pred[:, i].min())
    max_val = max(y_test_original[:, i].max(), y_pred[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    
    # Metrics
    mae = mean_absolute_error(y_test_original[:, i], y_pred[:, i])
    r2 = r2_score(y_test_original[:, i], y_pred[:, i])
    
    ax.set_xlabel(f'Actual {name} (m)', fontsize=11)
    ax.set_ylabel(f'Predicted {name} (m)', fontsize=11)
    ax.set_title(f'{name}\nMAE: {mae:.2f}m | R²: {r2:.4f}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../04_results/xyz_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
print("✓ Saved: xyz_predictions_vs_actual.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING SUMMARY")
print("=" * 80)

print(f"\nDataset:")
print(f"  Total samples: {len(data)}")
print(f"  Train: {len(X_train)} | Test: {len(X_test)} | Val: {len(X_val)}")

print(f"\nFeatures:")
print(f"  Time-based: {len(time_features)}")
print(f"  Satellite one-hot: {len(sat_features)}")
print(f"  Total: {len(feature_cols)}")

print(f"\nTargets:")
print(f"  X_Error, Y_Error, Z_Error (3 position errors)")
print(f"  Clock_Error: NOT included (train separately)")

print(f"\nModel Architecture:")
print(f"  Input: {X_train.shape[1]} features")
print(f"  Hidden: 128 → 64 → 32")
print(f"  Output: 3 (X, Y, Z errors)")
print(f"  Total params: {model.count_params():,}")

print(f"\nTraining:")
print(f"  Epochs completed: {len(history.history['loss'])}")
print(f"  Best val MAE: {min(history.history['val_mae']):.2f} meters")

print(f"\nTest Performance:")
print(f"  Overall MAE:  {mae_overall:.2f} meters")
print(f"  Overall RMSE: {rmse_overall:.2f} meters")

print("\n" + "=" * 80)
print("SUCCESS! Model trained and saved.")
print("=" * 80)

print("\nNext Steps:")
print("1. Review visualizations in 04_results/")
print("2. Test model on Day 8 data (if available)")
print("3. Train separate Clock_Error model if needed")
print("4. Deploy model for production forecasting")
