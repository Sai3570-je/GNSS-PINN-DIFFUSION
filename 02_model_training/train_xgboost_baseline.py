# -*- coding: utf-8 -*-
"""
XGBoost Baseline Model for GNSS Error Prediction
Gradient boosting baseline to compare with deep learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
import os

print("=" * 70)
print("  XGBOOST BASELINE - GNSS ERROR PREDICTION")
print("=" * 70)

# ============================================================================
# STEP 1: Load and Prepare Data
# ============================================================================
print("\n[STEP 1] Loading dataset...")
data_file = '../data/processed/real_data.csv'
df = pd.read_csv(data_file)
print(f"[OK] Dataset loaded: {len(df)} samples")

# ============================================================================
# STEP 2: Feature Engineering (same as enhanced model)
# ============================================================================
print("\n[STEP 2] Engineering features...")

df['timestamp'] = pd.to_datetime(df['timestamp'])

# Time features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['day_of_year'] = df['timestamp'].dt.dayofyear

# Cyclical encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

# Orbital features
df['M_sin'] = np.sin(df['M'])
df['M_cos'] = np.cos(df['M'])
df['RAAN_sin'] = np.sin(df['RAAN'])
df['RAAN_cos'] = np.cos(df['RAAN'])
df['omega_sin'] = np.sin(df['omega'])
df['omega_cos'] = np.cos(df['omega'])
df['i_sin'] = np.sin(df['i'])
df['i_cos'] = np.cos(df['i'])

# Satellite encoding
le_sat = LabelEncoder()
df['sat_id_encoded'] = le_sat.fit_transform(df['sat_id'])

# Derived features
df['e_squared'] = df['e'] ** 2
df['semi_major_axis_km'] = df['a'] / 1000

print(f"[OK] Features engineered!")

# ============================================================================
# STEP 3: Define Features
# ============================================================================
orbital_features = ['sqrtA', 'a', 'e', 'i', 'RAAN', 'omega', 'M']
time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'doy_sin', 'doy_cos']
cyclical_features = ['M_sin', 'M_cos', 'RAAN_sin', 'RAAN_cos', 'omega_sin', 'omega_cos', 'i_sin', 'i_cos']
derived_features = ['e_squared', 'semi_major_axis_km', 'sat_id_encoded']

all_features = orbital_features + time_features + cyclical_features + derived_features

position_targets = ['X_Error', 'Y_Error', 'Z_Error']
clock_target = ['Clock_Error']

print(f"[OK] Total features: {len(all_features)}")

# ============================================================================
# STEP 4: Split Data
# ============================================================================
print("\n[STEP 4] Splitting dataset...")

df_train, df_temp = train_test_split(df, test_size=0.20, random_state=42, shuffle=True)
df_test, df_val = train_test_split(df_temp, test_size=0.25, random_state=42, shuffle=True)

X_train = df_train[all_features].values
X_test = df_test[all_features].values
X_val = df_val[all_features].values

y_train_pos = df_train[position_targets].values
y_test_pos = df_test[position_targets].values
y_val_pos = df_val[position_targets].values

y_train_clk = df_train[clock_target].values.ravel()
y_test_clk = df_test[clock_target].values.ravel()
y_val_clk = df_val[clock_target].values.ravel()

print(f"[OK] Data split: {len(df_train)} train, {len(df_test)} test, {len(df_val)} val")

# ============================================================================
# STEP 5: Train Position Models (X, Y, Z separately)
# ============================================================================
print("\n[STEP 5] Training XGBoost models for position errors...")

# XGBoost parameters
xgb_params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

# Train separate model for each position dimension
models_pos = {}
predictions_pos = np.zeros((len(X_test), 3))

for i, name in enumerate(['X_Error', 'Y_Error', 'Z_Error']):
    print(f"\nTraining {name} model...")
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_train, y_train_pos[:, i],
        eval_set=[(X_val, y_val_pos[:, i])],
        early_stopping_rounds=20,
        verbose=False
    )
    
    models_pos[name] = model
    predictions_pos[:, i] = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test_pos[:, i], predictions_pos[:, i])
    rmse = np.sqrt(mean_squared_error(y_test_pos[:, i], predictions_pos[:, i]))
    r2 = r2_score(y_test_pos[:, i], predictions_pos[:, i])
    
    print(f"  MAE: {mae:.2f}m | RMSE: {rmse:.2f}m | R²: {r2:.4f}")

print("\n[OK] Position models trained!")

# ============================================================================
# STEP 6: Train Clock Model
# ============================================================================
print("\n[STEP 6] Training XGBoost model for clock error...")

model_clk = xgb.XGBRegressor(**xgb_params)
model_clk.fit(
    X_train, y_train_clk,
    eval_set=[(X_val, y_val_clk)],
    early_stopping_rounds=20,
    verbose=False
)

predictions_clk = model_clk.predict(X_test)

mae_clk = mean_absolute_error(y_test_clk, predictions_clk)
rmse_clk = np.sqrt(mean_squared_error(y_test_clk, predictions_clk))
r2_clk = r2_score(y_test_clk, predictions_clk)

print(f"  MAE: {mae_clk:.2f}m | RMSE: {rmse_clk:.2f}m | R²: {r2_clk:.4f}")
print("[OK] Clock model trained!")

# ============================================================================
# STEP 7: Detailed Evaluation
# ============================================================================
print("\n" + "=" * 70)
print("XGBOOST BASELINE RESULTS")
print("=" * 70)

print("\nPOSITION ERRORS:")
for i, name in enumerate(['X_Error', 'Y_Error', 'Z_Error']):
    mae = mean_absolute_error(y_test_pos[:, i], predictions_pos[:, i])
    rmse = np.sqrt(mean_squared_error(y_test_pos[:, i], predictions_pos[:, i]))
    r2 = r2_score(y_test_pos[:, i], predictions_pos[:, i])
    
    print(f"\n{name}:")
    print(f"  MAE:  {mae:.2f} meters")
    print(f"  RMSE: {rmse:.2f} meters")
    print(f"  R²:   {r2:.4f}")

print(f"\nCLOCK ERROR:")
print(f"  MAE:  {mae_clk:.2f} meters")
print(f"  RMSE: {rmse_clk:.2f} meters")
print(f"  R²:   {r2_clk:.4f}")

# Overall position metrics
mae_pos_overall = mean_absolute_error(y_test_pos, predictions_pos)
rmse_pos_overall = np.sqrt(mean_squared_error(y_test_pos, predictions_pos))

print(f"\nOVERALL POSITION:")
print(f"  MAE:  {mae_pos_overall:.2f} meters")
print(f"  RMSE: {rmse_pos_overall:.2f} meters")

# ============================================================================
# STEP 8: Save Models
# ============================================================================
print("\n[STEP 8] Saving models...")

for name, model in models_pos.items():
    with open(f'../03_models/xgb_{name.lower()}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

with open('../03_models/xgb_clock_model.pkl', 'wb') as f:
    pickle.dump(model_clk, f)

print("[OK] XGBoost models saved!")

# ============================================================================
# STEP 9: Visualizations
# ============================================================================
print("\n[STEP 9] Creating visualizations...")

os.makedirs('../04_results/xgboost_baseline', exist_ok=True)

# Prediction vs Actual
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# X_Error
axes[0, 0].scatter(y_test_pos[:, 0], predictions_pos[:, 0], alpha=0.5, s=20)
axes[0, 0].plot([-10, 10], [-10, 10], 'r--', linewidth=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual X_Error (m)', fontsize=12)
axes[0, 0].set_ylabel('Predicted X_Error (m)', fontsize=12)
mae_x = mean_absolute_error(y_test_pos[:, 0], predictions_pos[:, 0])
r2_x = r2_score(y_test_pos[:, 0], predictions_pos[:, 0])
axes[0, 0].set_title(f'X_Error (XGBoost)\nMAE: {mae_x:.2f}m | R²: {r2_x:.4f}', 
                      fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Y_Error
axes[0, 1].scatter(y_test_pos[:, 1], predictions_pos[:, 1], alpha=0.5, s=20, color='orange')
axes[0, 1].plot([-10, 10], [-10, 10], 'r--', linewidth=2, label='Perfect Prediction')
axes[0, 1].set_xlabel('Actual Y_Error (m)', fontsize=12)
axes[0, 1].set_ylabel('Predicted Y_Error (m)', fontsize=12)
mae_y = mean_absolute_error(y_test_pos[:, 1], predictions_pos[:, 1])
r2_y = r2_score(y_test_pos[:, 1], predictions_pos[:, 1])
axes[0, 1].set_title(f'Y_Error (XGBoost)\nMAE: {mae_y:.2f}m | R²: {r2_y:.4f}', 
                      fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Z_Error
axes[1, 0].scatter(y_test_pos[:, 2], predictions_pos[:, 2], alpha=0.5, s=20, color='green')
axes[1, 0].plot([-10, 10], [-10, 10], 'r--', linewidth=2, label='Perfect Prediction')
axes[1, 0].set_xlabel('Actual Z_Error (m)', fontsize=12)
axes[1, 0].set_ylabel('Predicted Z_Error (m)', fontsize=12)
mae_z = mean_absolute_error(y_test_pos[:, 2], predictions_pos[:, 2])
r2_z = r2_score(y_test_pos[:, 2], predictions_pos[:, 2])
axes[1, 0].set_title(f'Z_Error (XGBoost)\nMAE: {mae_z:.2f}m | R²: {r2_z:.4f}', 
                      fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Clock_Error
axes[1, 1].scatter(y_test_clk, predictions_clk, alpha=0.5, s=20, color='purple')
min_clk = min(y_test_clk.min(), predictions_clk.min())
max_clk = max(y_test_clk.max(), predictions_clk.max())
axes[1, 1].plot([min_clk, max_clk], [min_clk, max_clk], 'r--', linewidth=2, label='Perfect Prediction')
axes[1, 1].set_xlabel('Actual Clock_Error (m)', fontsize=12)
axes[1, 1].set_ylabel('Predicted Clock_Error (m)', fontsize=12)
axes[1, 1].set_title(f'Clock_Error (XGBoost)\nMAE: {mae_clk:.2f}m | R²: {r2_clk:.4f}', 
                      fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../04_results/xgboost_baseline/prediction_vs_actual.png', dpi=300, bbox_inches='tight')
print("[OK] Scatter plots saved!")
plt.close()

# Feature Importance
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for idx, (name, model) in enumerate(list(models_pos.items()) + [('Clock_Error', model_clk)]):
    ax = axes[idx // 2, idx % 2]
    
    importance = model.feature_importances_
    indices = np.argsort(importance)[-15:]  # Top 15 features
    
    ax.barh(range(len(indices)), importance[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([all_features[i] for i in indices])
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'{name} - Top 15 Features', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('../04_results/xgboost_baseline/feature_importance.png', dpi=300, bbox_inches='tight')
print("[OK] Feature importance plots saved!")
plt.close()

print("\n" + "=" * 70)
print("[SUCCESS] XGBoost baseline completed!")
print("=" * 70)
print("\nFiles created:")
print("  - 03_models/xgb_x_error_model.pkl")
print("  - 03_models/xgb_y_error_model.pkl")
print("  - 03_models/xgb_z_error_model.pkl")
print("  - 03_models/xgb_clock_model.pkl")
print("  - 04_results/xgboost_baseline/prediction_vs_actual.png")
print("  - 04_results/xgboost_baseline/feature_importance.png")
print("=" * 70)
