"""
GNSS SATELLITE ERROR PREDICTION - COMPREHENSIVE MODEL
======================================================
Predicts: X_Error, Y_Error, Z_Error, Clock_Error (4 outputs)

Features:
- 6 cyclical time features (hour_sin/cos, day_sin/cos, doy_sin/cos)
- 32 satellite one-hot encoded features (G01-G32)
- 7 Keplerian orbital elements (sqrtA, a, e, i, RAAN, omega, M)
Total: 45 features

Architecture:
- Input(45) → Dense(128) → Dropout(0.2) → Dense(64) → Dropout(0.2) → Dense(32) → Dense(4)
- L2 regularization on dense layers
- MAE loss, Adam optimizer (lr=0.001)
- Early stopping (patience=5, monitor=val_loss)

Target metrics:
- X, Y, Z MAE < 2.0m, R² > 0.5
- Clock MAE < 5000m, R² > 0.5
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = '../data/processed/real_data.csv'
MODELS_DIR = '../03_models'
RESULTS_DIR = '../04_results'

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5
L2_LAMBDA = 0.001

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 80)
print("GNSS COMPREHENSIVE ERROR PREDICTION - DEEP LEARNING MODEL")
print("Targets: X_Error, Y_Error, Z_Error, Clock_Error (All 4 Outputs)")
print("=" * 80)
print()

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("[STEP 1] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded {len(df)} records from {DATA_PATH}")
print(f"  Columns: {list(df.columns)}")
print()

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
print("[STEP 2] Engineering features...")

# Parse timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Cyclical time features (6 features)
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['day_of_year'] = df['timestamp'].dt.dayofyear

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

# Satellite one-hot encoding (32 features)
sat_dummies = pd.get_dummies(df['sat_id'], prefix='sat')
# Ensure all 32 satellites are present
all_sats = [f'sat_G{i:02d}' for i in range(1, 33)]
for sat in all_sats:
    if sat not in sat_dummies.columns:
        sat_dummies[sat] = 0
sat_dummies = sat_dummies[all_sats]  # Ensure correct order

print(f"  Unique satellites: {df['sat_id'].nunique()}")
print("✓ Features engineered")
print(f"  Time features: hour_sin, hour_cos, day_sin, day_cos, doy_sin, doy_cos")
print(f"  Satellite features: 32 one-hot encoded columns")
print(f"  Keplerian elements: 7 columns (sqrtA, a, e, i, RAAN, omega, M)")
print()

# ============================================================================
# STEP 3: PREPARE FEATURES AND TARGETS
# ============================================================================
print("[STEP 3] Preparing features and targets...")

# Time features (6)
time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'doy_sin', 'doy_cos']

# Keplerian orbital elements (7)
keplerian_features = ['sqrtA', 'a', 'e', 'i', 'RAAN', 'omega', 'M']

# Combine all features
X_time = df[time_features].values
X_sat = sat_dummies.values
X_kep = df[keplerian_features].values
X = np.hstack([X_time, X_sat, X_kep])

# All 4 error targets
y = df[['X_Error', 'Y_Error', 'Z_Error', 'Clock_Error']].values

print("✓ Features prepared")
print(f"  Feature shape: {X.shape}")
print(f"  Target shape:  {y.shape}")
print(f"  Total features: {X.shape[1]} (6 time + 32 satellite + 7 Keplerian)")
print(f"  Total targets: {y.shape[1]} (X, Y, Z, Clock errors)")
print()

# ============================================================================
# STEP 4: SCALE FEATURES AND TARGETS
# ============================================================================
print("[STEP 4] Scaling features and targets...")

# Scale features to [0, 1]
x_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X)

# Scale targets (standardize for better convergence)
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)

print("✓ Scaling complete")
print(f"  Feature scaler: MinMaxScaler (range 0-1)")
print(f"  Target scaler:  StandardScaler (mean=0, std=1)")
print()

# ============================================================================
# STEP 5: SPLIT DATA (80% train, 15% test, 5% val)
# ============================================================================
print("[STEP 5] Splitting data (80% train, 15% test, 5% val)...")

# First split: 80% train, 20% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y_scaled, test_size=0.20, random_state=RANDOM_SEED, shuffle=True
)

# Second split: 15% test, 5% val (from 20% temp)
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=RANDOM_SEED, shuffle=True
)

print("✓ Data split complete")
print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
print(f"  Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print()

# ============================================================================
# STEP 6: BUILD MODEL
# ============================================================================
print("[STEP 6] Building Deep Neural Network with L2 regularization...")

model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],), 
          kernel_regularizer=l2(L2_LAMBDA), name='dense_128'),
    Dropout(0.2, name='dropout_1'),
    
    Dense(64, activation='relu', 
          kernel_regularizer=l2(L2_LAMBDA), name='dense_64'),
    Dropout(0.2, name='dropout_2'),
    
    Dense(32, activation='relu', 
          kernel_regularizer=l2(L2_LAMBDA), name='dense_32'),
    
    Dense(4, activation='linear', name='output')  # 4 outputs: X, Y, Z, Clock
], name='GNSS_Comprehensive_Model')

# Compile with MAE loss
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='mae',
    metrics=['mae', 'mse']
)

print("✓ Model built")
model.summary()
print()

# ============================================================================
# STEP 7: TRAIN MODEL
# ============================================================================
print("[STEP 7] Training model...")
print("=" * 80)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, 'best_gnss_comprehensive.keras'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print()
print("✓ Training complete!")
print()

# ============================================================================
# STEP 8: EVALUATE ON TEST SET
# ============================================================================
print("[STEP 8] Evaluating on test set...")
print("=" * 80)
print()

# Predict on test set
y_pred_scaled = model.predict(X_test, verbose=0)

# Inverse transform to original scale
y_test_original = y_scaler.inverse_transform(y_test)
y_pred = y_scaler.inverse_transform(y_pred_scaled)

# Calculate metrics for each output
target_names = ['X_Error', 'Y_Error', 'Z_Error', 'Clock_Error']
print("TEST SET RESULTS:")
print("=" * 80)
print()

results = {}
for i, name in enumerate(target_names):
    mae = mean_absolute_error(y_test_original[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test_original[:, i], y_pred[:, i]))
    r2 = r2_score(y_test_original[:, i], y_pred[:, i])
    
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    
    print(f"{name}:")
    print(f"  MAE:  {mae:.2f} meters")
    print(f"  RMSE: {rmse:.2f} meters")
    print(f"  R²:   {r2:.4f}")
    
    # Check goal metrics
    if name == 'Clock_Error':
        goal_mae = 5000
    else:
        goal_mae = 2.0
    
    mae_status = "✓" if mae < goal_mae else "✗"
    r2_status = "✓" if r2 > 0.5 else "✗"
    print(f"  Goal Check: MAE {mae_status} (<{goal_mae}m), R² {r2_status} (>0.5)")
    print()

# Overall metrics
mae_overall = mean_absolute_error(y_test_original, y_pred)
rmse_overall = np.sqrt(mean_squared_error(y_test_original, y_pred))

print("OVERALL (all 4 outputs combined):")
print(f"  MAE:  {mae_overall:.2f} meters")
print(f"  RMSE: {rmse_overall:.2f} meters")
print()

# ============================================================================
# STEP 9: SAVE MODEL AND ARTIFACTS
# ============================================================================
print("[STEP 9] Saving model and artifacts...")

# Save final model
model.save(os.path.join(MODELS_DIR, 'gnss_comprehensive_model.keras'))

# Save scalers
with open(os.path.join(MODELS_DIR, 'x_scaler_comprehensive.pkl'), 'wb') as f:
    pickle.dump(x_scaler, f)

with open(os.path.join(MODELS_DIR, 'y_scaler_comprehensive.pkl'), 'wb') as f:
    pickle.dump(y_scaler, f)

# Save feature columns info
feature_info = {
    'time_features': time_features,
    'satellite_features': all_sats,
    'keplerian_features': keplerian_features,
    'total_features': X.shape[1]
}
with open(os.path.join(MODELS_DIR, 'feature_info_comprehensive.pkl'), 'wb') as f:
    pickle.dump(feature_info, f)

# Save results
results_df = pd.DataFrame(results).T
results_df.to_csv(os.path.join(RESULTS_DIR, 'comprehensive_test_results.csv'))

print("✓ Saved files:")
print(f"  - gnss_comprehensive_model.keras (final model)")
print(f"  - best_gnss_comprehensive.keras (best checkpoint)")
print(f"  - x_scaler_comprehensive.pkl (feature scaler)")
print(f"  - y_scaler_comprehensive.pkl (target scaler)")
print(f"  - feature_info_comprehensive.pkl (feature metadata)")
print(f"  - comprehensive_test_results.csv (metrics)")
print()

# ============================================================================
# STEP 10: CREATE VISUALIZATIONS
# ============================================================================
print("[STEP 10] Creating visualizations...")

# 1. Training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('MAE Loss', fontsize=12)
axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# MAE
axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('MAE (scaled)', fontsize=12)
axes[1].set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'comprehensive_training_history.png'), dpi=150)
plt.close()

# 2. Predictions vs Actual (50 samples)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

sample_size = min(50, len(y_test_original))
sample_indices = np.random.choice(len(y_test_original), sample_size, replace=False)

for i, name in enumerate(target_names):
    y_true_sample = y_test_original[sample_indices, i]
    y_pred_sample = y_pred[sample_indices, i]
    
    # Scatter plot
    axes[i].scatter(y_true_sample, y_pred_sample, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true_sample.min(), y_pred_sample.min())
    max_val = max(y_true_sample.max(), y_pred_sample.max())
    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    axes[i].set_xlabel('Actual (meters)', fontsize=11)
    axes[i].set_ylabel('Predicted (meters)', fontsize=11)
    axes[i].set_title(f'{name} - Actual vs Predicted\n(MAE: {results[name]["MAE"]:.2f}m, R²: {results[name]["R2"]:.3f})', 
                     fontsize=12, fontweight='bold')
    axes[i].legend(fontsize=10)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'comprehensive_predictions_vs_actual.png'), dpi=150)
plt.close()

print("✓ Saved visualizations:")
print(f"  - comprehensive_training_history.png")
print(f"  - comprehensive_predictions_vs_actual.png")
print()

# ============================================================================
# TRAINING SUMMARY
# ============================================================================
print("=" * 80)
print("TRAINING SUMMARY")
print("=" * 80)
print()

print("Dataset:")
print(f"  Total samples: {len(df)}")
print(f"  Train: {len(X_train)} | Test: {len(X_test)} | Val: {len(X_val)}")
print()

print("Features:")
print(f"  Time-based: 6 (cyclical hour, day, day-of-year)")
print(f"  Satellite one-hot: 32 (G01-G32)")
print(f"  Keplerian elements: 7 (sqrtA, a, e, i, RAAN, omega, M)")
print(f"  Total: 45")
print()

print("Targets:")
print(f"  X_Error, Y_Error, Z_Error, Clock_Error (4 outputs)")
print()

print("Model Architecture:")
print(f"  Input: 45 features")
print(f"  Hidden: 128 → 64 → 32")
print(f"  Output: 4 (X, Y, Z, Clock errors)")
print(f"  Total params: {model.count_params():,}")
print(f"  Regularization: L2 (lambda={L2_LAMBDA}), Dropout (0.2)")
print()

print("Training:")
print(f"  Epochs completed: {len(history.history['loss'])}")
print(f"  Best val MAE: {min(history.history['val_mae']):.4f}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Batch size: {BATCH_SIZE}")
print()

print("Test Performance:")
for name in target_names:
    print(f"  {name}: MAE {results[name]['MAE']:.2f}m, RMSE {results[name]['RMSE']:.2f}m, R² {results[name]['R2']:.3f}")
print(f"  Overall MAE: {mae_overall:.2f}m, RMSE {rmse_overall:.2f}m")
print()

# Check if all goals met
position_goals_met = all(results[name]['MAE'] < 2.0 and results[name]['R2'] > 0.5 
                         for name in ['X_Error', 'Y_Error', 'Z_Error'])
clock_goal_met = results['Clock_Error']['MAE'] < 5000 and results['Clock_Error']['R2'] > 0.5

if position_goals_met and clock_goal_met:
    print("🎯 ALL GOAL METRICS ACHIEVED!")
else:
    print("⚠️ Some goal metrics not yet met. Consider:")
    print("   - Increase model capacity (more layers/neurons)")
    print("   - Add more training data")
    print("   - Fine-tune hyperparameters")
    print("   - Try ensemble methods")
print()

print("=" * 80)
print("SUCCESS! Comprehensive model trained and saved.")
print("=" * 80)
print()

print("Next Steps:")
print("1. Review visualizations in 04_results/")
print("2. Test model on Day 8 data for forecasting")
print("3. Deploy model for production error prediction")
print("4. Monitor performance on new data")
