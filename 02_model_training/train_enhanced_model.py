# -*- coding: utf-8 -*-
"""
Enhanced GNSS Error Prediction with Separate Position & Clock Models
Features:
- Two specialized models (Position: X/Y/Z, Clock: Clock_Error)
- Advanced feature engineering (time features, satellite encoding)
- Comprehensive visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pickle
import os
from datetime import datetime

print("=" * 70)
print("  ENHANCED GNSS SATELLITE ERROR PREDICTION")
print("  Two-Model Approach: Position (X/Y/Z) + Clock (Clock_Error)")
print("=" * 70)

# ============================================================================
# STEP 1: Load Full Dataset with Features
# ============================================================================
print("\n[STEP 1] Loading full dataset with features...")
data_file = '../data/processed/real_data.csv'
df = pd.read_csv(data_file)
print(f"[OK] Dataset loaded: {len(df)} samples")
print(f"     Columns: {list(df.columns)}")

# ============================================================================
# STEP 2: Feature Engineering
# ============================================================================
print("\n[STEP 2] Engineering advanced features...")

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Time-based features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['day_of_year'] = df['timestamp'].dt.dayofyear

# Cyclical encoding for time features
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

# Orbital phase from Mean Anomaly (already cyclical)
df['M_sin'] = np.sin(df['M'])
df['M_cos'] = np.cos(df['M'])

# RAAN and omega cyclical encoding
df['RAAN_sin'] = np.sin(df['RAAN'])
df['RAAN_cos'] = np.cos(df['RAAN'])
df['omega_sin'] = np.sin(df['omega'])
df['omega_cos'] = np.cos(df['omega'])

# Inclination features
df['i_sin'] = np.sin(df['i'])
df['i_cos'] = np.cos(df['i'])

# Satellite ID encoding
le_sat = LabelEncoder()
df['sat_id_encoded'] = le_sat.fit_transform(df['sat_id'])

# Additional orbital features
df['e_squared'] = df['e'] ** 2
df['semi_major_axis_km'] = df['a'] / 1000  # Convert to km

print(f"[OK] Features engineered!")
print(f"     Satellite IDs: {len(le_sat.classes_)} unique satellites")
print(f"     Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# ============================================================================
# STEP 3: Define Feature Sets
# ============================================================================
print("\n[STEP 3] Defining feature sets...")

# Base orbital features
orbital_features = ['sqrtA', 'a', 'e', 'i', 'RAAN', 'omega', 'M']

# Engineered features
time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'doy_sin', 'doy_cos']
cyclical_features = ['M_sin', 'M_cos', 'RAAN_sin', 'RAAN_cos', 'omega_sin', 'omega_cos', 'i_sin', 'i_cos']
derived_features = ['e_squared', 'semi_major_axis_km', 'sat_id_encoded']

# Combined feature set
all_features = orbital_features + time_features + cyclical_features + derived_features

print(f"[OK] Total features: {len(all_features)}")
print(f"     Orbital: {len(orbital_features)}")
print(f"     Time: {len(time_features)}")
print(f"     Cyclical: {len(cyclical_features)}")
print(f"     Derived: {len(derived_features)}")

# Target columns
position_targets = ['X_Error', 'Y_Error', 'Z_Error']
clock_target = ['Clock_Error']

# ============================================================================
# STEP 4: Split Data (80/15/5)
# ============================================================================
print("\n[STEP 4] Splitting dataset (80% train, 15% test, 5% validation)...")

from sklearn.model_selection import train_test_split

# First split: 80% train, 20% temp
df_train, df_temp = train_test_split(df, test_size=0.20, random_state=42, shuffle=True)

# Second split: 15% test, 5% validation
df_test, df_val = train_test_split(df_temp, test_size=0.25, random_state=42, shuffle=True)

print(f"[OK] Data split complete!")
print(f"     Training:   {len(df_train)} samples ({len(df_train)/len(df)*100:.1f}%)")
print(f"     Test:       {len(df_test)} samples ({len(df_test)/len(df)*100:.1f}%)")
print(f"     Validation: {len(df_val)} samples ({len(df_val)/len(df)*100:.1f}%)")

# ============================================================================
# STEP 5: Prepare Features and Targets
# ============================================================================
print("\n[STEP 5] Preparing features and targets...")

X_train = df_train[all_features].values
X_test = df_test[all_features].values
X_val = df_val[all_features].values

# Position targets (X, Y, Z)
y_train_pos = df_train[position_targets].values
y_test_pos = df_test[position_targets].values
y_val_pos = df_val[position_targets].values

# Clock target
y_train_clk = df_train[clock_target].values
y_test_clk = df_test[clock_target].values
y_val_clk = df_val[clock_target].values

print(f"[OK] Data prepared!")
print(f"     Feature shape: {X_train.shape}")
print(f"     Position target shape: {y_train_pos.shape}")
print(f"     Clock target shape: {y_train_clk.shape}")

# ============================================================================
# STEP 6: Normalize Features
# ============================================================================
print("\n[STEP 6] Normalizing features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

print(f"[OK] Features normalized (mean=0, std=1)")

# ============================================================================
# STEP 7: Build Position Model (X/Y/Z Errors)
# ============================================================================
print("\n[STEP 7] Building Position Error Model (X/Y/Z)...")

def build_position_model(input_dim, output_dim=3):
    """Deep neural network for position errors"""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.2),
        
        layers.Dense(output_dim, activation='linear')  # 3 outputs: X, Y, Z
    ], name='Position_Error_Model')
    
    return model

model_position = build_position_model(input_dim=len(all_features), output_dim=3)
model_position.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("[OK] Position model created!")
model_position.summary()

# ============================================================================
# STEP 8: Build Clock Model (Clock_Error)
# ============================================================================
print("\n[STEP 8] Building Clock Error Model...")

def build_clock_model(input_dim, output_dim=1):
    """Deep neural network for clock error"""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.2),
        
        layers.Dense(output_dim, activation='linear')  # 1 output: Clock_Error
    ], name='Clock_Error_Model')
    
    return model

model_clock = build_clock_model(input_dim=len(all_features), output_dim=1)
model_clock.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("[OK] Clock model created!")
model_clock.summary()

# ============================================================================
# STEP 9: Train Position Model
# ============================================================================
print("\n[STEP 9] Training Position Error Model...")
print("=" * 70)

callbacks_pos = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='../03_models/best_position_model.keras',
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

history_pos = model_position.fit(
    X_train_scaled, y_train_pos,
    validation_data=(X_val_scaled, y_val_pos),
    epochs=150,
    batch_size=32,
    callbacks=callbacks_pos,
    verbose=1
)

print("\n[OK] Position model training completed!")

# ============================================================================
# STEP 10: Train Clock Model
# ============================================================================
print("\n[STEP 10] Training Clock Error Model...")
print("=" * 70)

callbacks_clk = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='../03_models/best_clock_model.keras',
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

history_clk = model_clock.fit(
    X_train_scaled, y_train_clk,
    validation_data=(X_val_scaled, y_val_clk),
    epochs=150,
    batch_size=32,
    callbacks=callbacks_clk,
    verbose=1
)

print("\n[OK] Clock model training completed!")

# ============================================================================
# STEP 11: Evaluate Position Model
# ============================================================================
print("\n[STEP 11] Evaluating Position Error Model on Test Set...")
print("=" * 70)

y_pred_pos = model_position.predict(X_test_scaled, verbose=0)

print("\nPOSITION ERROR RESULTS (X/Y/Z):")
print("=" * 70)

position_names = ['X_Error', 'Y_Error', 'Z_Error']
for i, name in enumerate(position_names):
    mae = mean_absolute_error(y_test_pos[:, i], y_pred_pos[:, i])
    rmse = np.sqrt(mean_squared_error(y_test_pos[:, i], y_pred_pos[:, i]))
    r2 = r2_score(y_test_pos[:, i], y_pred_pos[:, i])
    
    print(f"\n{name}:")
    print(f"  MAE:  {mae:.2f} meters")
    print(f"  RMSE: {rmse:.2f} meters")
    print(f"  R²:   {r2:.4f}")

# Overall position metrics
mae_pos_overall = mean_absolute_error(y_test_pos, y_pred_pos)
rmse_pos_overall = np.sqrt(mean_squared_error(y_test_pos, y_pred_pos))
print(f"\nOVERALL POSITION:")
print(f"  MAE:  {mae_pos_overall:.2f} meters")
print(f"  RMSE: {rmse_pos_overall:.2f} meters")

# ============================================================================
# STEP 12: Evaluate Clock Model
# ============================================================================
print("\n[STEP 12] Evaluating Clock Error Model on Test Set...")
print("=" * 70)

y_pred_clk = model_clock.predict(X_test_scaled, verbose=0)

mae_clk = mean_absolute_error(y_test_clk, y_pred_clk)
rmse_clk = np.sqrt(mean_squared_error(y_test_clk, y_pred_clk))
r2_clk = r2_score(y_test_clk, y_pred_clk)

print("\nCLOCK ERROR RESULTS:")
print("=" * 70)
print(f"  MAE:  {mae_clk:.2f} meters")
print(f"  RMSE: {rmse_clk:.2f} meters")
print(f"  R²:   {r2_clk:.4f}")

# ============================================================================
# STEP 13: Save Models and Scaler
# ============================================================================
print("\n[STEP 13] Saving models and scaler...")

model_position.save('../03_models/position_model.keras')
model_clock.save('../03_models/clock_model.keras')

with open('../03_models/scaler_enhanced.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('../03_models/label_encoder_sat.pkl', 'wb') as f:
    pickle.dump(le_sat, f)

# Save feature names
with open('../03_models/feature_names_enhanced.pkl', 'wb') as f:
    pickle.dump(all_features, f)

print("[OK] Models and preprocessing objects saved!")
print("     - position_model.keras")
print("     - clock_model.keras")
print("     - scaler_enhanced.pkl")
print("     - label_encoder_sat.pkl")
print("     - feature_names_enhanced.pkl")

# ============================================================================
# STEP 14: Visualizations
# ============================================================================
print("\n[STEP 14] Creating visualizations...")

# Create results directory if it doesn't exist
os.makedirs('../04_results/enhanced_model', exist_ok=True)

# 14.1: Training History
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Position model - Loss
axes[0, 0].plot(history_pos.history['loss'], label='Train Loss', linewidth=2)
axes[0, 0].plot(history_pos.history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 0].set_title('Position Model - Loss', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('MSE Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Position model - MAE
axes[0, 1].plot(history_pos.history['mae'], label='Train MAE', linewidth=2)
axes[0, 1].plot(history_pos.history['val_mae'], label='Val MAE', linewidth=2)
axes[0, 1].set_title('Position Model - MAE', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MAE (meters)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Clock model - Loss
axes[1, 0].plot(history_clk.history['loss'], label='Train Loss', linewidth=2)
axes[1, 0].plot(history_clk.history['val_loss'], label='Val Loss', linewidth=2)
axes[1, 0].set_title('Clock Model - Loss', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('MSE Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Clock model - MAE
axes[1, 1].plot(history_clk.history['mae'], label='Train MAE', linewidth=2)
axes[1, 1].plot(history_clk.history['val_mae'], label='Val MAE', linewidth=2)
axes[1, 1].set_title('Clock Model - MAE', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('MAE (meters)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../04_results/enhanced_model/training_history.png', dpi=300, bbox_inches='tight')
print("[OK] Training history saved: training_history.png")
plt.close()

# 14.2: Prediction vs Actual Scatter Plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# X_Error
axes[0, 0].scatter(y_test_pos[:, 0], y_pred_pos[:, 0], alpha=0.5, s=20)
axes[0, 0].plot([-10, 10], [-10, 10], 'r--', linewidth=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual X_Error (m)', fontsize=12)
axes[0, 0].set_ylabel('Predicted X_Error (m)', fontsize=12)
axes[0, 0].set_title('X_Error Prediction', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Y_Error
axes[0, 1].scatter(y_test_pos[:, 1], y_pred_pos[:, 1], alpha=0.5, s=20, color='orange')
axes[0, 1].plot([-10, 10], [-10, 10], 'r--', linewidth=2, label='Perfect Prediction')
axes[0, 1].set_xlabel('Actual Y_Error (m)', fontsize=12)
axes[0, 1].set_ylabel('Predicted Y_Error (m)', fontsize=12)
axes[0, 1].set_title('Y_Error Prediction', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Z_Error
axes[1, 0].scatter(y_test_pos[:, 2], y_pred_pos[:, 2], alpha=0.5, s=20, color='green')
axes[1, 0].plot([-10, 10], [-10, 10], 'r--', linewidth=2, label='Perfect Prediction')
axes[1, 0].set_xlabel('Actual Z_Error (m)', fontsize=12)
axes[1, 0].set_ylabel('Predicted Z_Error (m)', fontsize=12)
axes[1, 0].set_title('Z_Error Prediction', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Clock_Error
axes[1, 1].scatter(y_test_clk, y_pred_clk, alpha=0.5, s=20, color='purple')
min_clk = min(y_test_clk.min(), y_pred_clk.min())
max_clk = max(y_test_clk.max(), y_pred_clk.max())
axes[1, 1].plot([min_clk, max_clk], [min_clk, max_clk], 'r--', linewidth=2, label='Perfect Prediction')
axes[1, 1].set_xlabel('Actual Clock_Error (m)', fontsize=12)
axes[1, 1].set_ylabel('Predicted Clock_Error (m)', fontsize=12)
axes[1, 1].set_title('Clock_Error Prediction', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../04_results/enhanced_model/prediction_vs_actual.png', dpi=300, bbox_inches='tight')
print("[OK] Prediction scatter plots saved: prediction_vs_actual.png")
plt.close()

# 14.3: Residual Plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# X_Error residuals
residuals_x = y_test_pos[:, 0] - y_pred_pos[:, 0]
axes[0, 0].scatter(y_pred_pos[:, 0], residuals_x, alpha=0.5, s=20)
axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Predicted X_Error (m)', fontsize=12)
axes[0, 0].set_ylabel('Residual (m)', fontsize=12)
axes[0, 0].set_title(f'X_Error Residuals (MAE: {mean_absolute_error(y_test_pos[:, 0], y_pred_pos[:, 0]):.2f}m)', 
                      fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Y_Error residuals
residuals_y = y_test_pos[:, 1] - y_pred_pos[:, 1]
axes[0, 1].scatter(y_pred_pos[:, 1], residuals_y, alpha=0.5, s=20, color='orange')
axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicted Y_Error (m)', fontsize=12)
axes[0, 1].set_ylabel('Residual (m)', fontsize=12)
axes[0, 1].set_title(f'Y_Error Residuals (MAE: {mean_absolute_error(y_test_pos[:, 1], y_pred_pos[:, 1]):.2f}m)', 
                      fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Z_Error residuals
residuals_z = y_test_pos[:, 2] - y_pred_pos[:, 2]
axes[1, 0].scatter(y_pred_pos[:, 2], residuals_z, alpha=0.5, s=20, color='green')
axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Predicted Z_Error (m)', fontsize=12)
axes[1, 0].set_ylabel('Residual (m)', fontsize=12)
axes[1, 0].set_title(f'Z_Error Residuals (MAE: {mean_absolute_error(y_test_pos[:, 2], y_pred_pos[:, 2]):.2f}m)', 
                      fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Clock_Error residuals
residuals_clk = y_test_clk.flatten() - y_pred_clk.flatten()
axes[1, 1].scatter(y_pred_clk, residuals_clk, alpha=0.5, s=20, color='purple')
axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Predicted Clock_Error (m)', fontsize=12)
axes[1, 1].set_ylabel('Residual (m)', fontsize=12)
axes[1, 1].set_title(f'Clock_Error Residuals (MAE: {mae_clk:.2f}m)', 
                      fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../04_results/enhanced_model/residual_plots.png', dpi=300, bbox_inches='tight')
print("[OK] Residual plots saved: residual_plots.png")
plt.close()

# ============================================================================
# STEP 15: Summary Report
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)
print(f"\nDataset:")
print(f"  Total samples:      {len(df)}")
print(f"  Training samples:   {len(df_train)}")
print(f"  Test samples:       {len(df_test)}")
print(f"  Validation samples: {len(df_val)}")

print(f"\nFeatures:")
print(f"  Total features:     {len(all_features)}")
print(f"  Satellites:         {len(le_sat.classes_)}")

print(f"\nPosition Model (X/Y/Z):")
print(f"  Epochs trained:     {len(history_pos.history['loss'])}")
print(f"  Final MAE:          {mae_pos_overall:.2f} meters")
print(f"  Final RMSE:         {rmse_pos_overall:.2f} meters")

print(f"\nClock Model:")
print(f"  Epochs trained:     {len(history_clk.history['loss'])}")
print(f"  Final MAE:          {mae_clk:.2f} meters")
print(f"  Final RMSE:         {rmse_clk:.2f} meters")
print(f"  R²:                 {r2_clk:.4f}")

print("\n" + "=" * 70)
print("[SUCCESS] Enhanced model training completed!")
print("=" * 70)

print("\nFiles created:")
print("  - 03_models/position_model.keras")
print("  - 03_models/clock_model.keras")
print("  - 03_models/best_position_model.keras")
print("  - 03_models/best_clock_model.keras")
print("  - 03_models/scaler_enhanced.pkl")
print("  - 03_models/label_encoder_sat.pkl")
print("  - 03_models/feature_names_enhanced.pkl")
print("  - 04_results/enhanced_model/training_history.png")
print("  - 04_results/enhanced_model/prediction_vs_actual.png")
print("  - 04_results/enhanced_model/residual_plots.png")
print("=" * 70)
