"""
Complete Day 8 Prediction Pipeline for SIH Submission
Generates predictions and comprehensive visualizations
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🛰️  GNSS DAY 8 ERROR PREDICTION - SIH SUBMISSION")
print("=" * 80)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# STEP 1: Load Model and Scalers
# ============================================================================
print("\n📥 STEP 1: Loading Enhanced Model...")

try:
    model = keras.models.load_model('03_models/best_gnss_enhanced.keras')
    print("✅ Model loaded successfully")
    print(f"   Architecture: {len(model.layers)} layers, {model.count_params():,} parameters")
except:
    model = keras.models.load_model('03_models/gnss_enhanced_model.keras')
    print("✅ Fallback model loaded")

with open('03_models/scalers_enhanced.pkl', 'rb') as f:
    scalers = pickle.load(f)
print("✅ Scalers loaded")

try:
    with open('03_models/feature_info_enhanced.pkl', 'rb') as f:
        feature_info = pickle.load(f)
    print("✅ Feature info loaded")
except:
    feature_info = None
    print("✅ Feature info not required")

# ============================================================================
# STEP 2: Prepare Day 8 Data
# ============================================================================
print("\n📊 STEP 2: Preparing Day 8 Feature Data...")

# Load full dataset
df = pd.read_csv('data/processed/real_data.csv')
print(f"✅ Loaded {len(df):,} total records")

# Filter Day 8 data
day8_data = df[df['timestamp'].str.startswith('2024-01-08')].copy()
print(f"✅ Extracted {len(day8_data)} Day 8 records")

if len(day8_data) == 0:
    print("❌ ERROR: No Day 8 data found!")
    print("   Using last available day for demonstration...")
    # Get last day in dataset
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    last_date = df['date'].max()
    day8_data = df[df['date'] == last_date].copy()
    print(f"   Using {last_date} with {len(day8_data)} records")

# Store actual errors for comparison
actual_errors = day8_data[['X_Error', 'Y_Error', 'Z_Error', 'Clock_Error']].copy()
print(f"✅ Stored actual errors for {len(actual_errors)} samples")

# ============================================================================
# STEP 3: Feature Engineering (same as training)
# ============================================================================
print("\n🔧 STEP 3: Engineering Features (52 total)...")

# Convert timestamp to datetime
day8_data['timestamp'] = pd.to_datetime(day8_data['timestamp'])

# 1. Time features (9 features)
day8_data['hour'] = day8_data['timestamp'].dt.hour
day8_data['minute'] = day8_data['timestamp'].dt.minute
day8_data['day'] = day8_data['timestamp'].dt.day
day8_data['day_of_year'] = day8_data['timestamp'].dt.dayofyear
day8_data['month'] = day8_data['timestamp'].dt.month

# Cyclical encoding
day8_data['hour_sin'] = np.sin(2 * np.pi * day8_data['hour'] / 24)
day8_data['hour_cos'] = np.cos(2 * np.pi * day8_data['hour'] / 24)
day8_data['minute_sin'] = np.sin(2 * np.pi * day8_data['minute'] / 60)
day8_data['minute_cos'] = np.cos(2 * np.pi * day8_data['minute'] / 60)
day8_data['day_sin'] = np.sin(2 * np.pi * day8_data['day'] / 31)
day8_data['day_cos'] = np.cos(2 * np.pi * day8_data['day'] / 31)
day8_data['doy_sin'] = np.sin(2 * np.pi * day8_data['day_of_year'] / 365)
day8_data['doy_cos'] = np.cos(2 * np.pi * day8_data['day_of_year'] / 365)
day8_data['month_sin'] = np.sin(2 * np.pi * day8_data['month'] / 12)

print("✅ Time features: 9")

# 2. Satellite one-hot encoding (32 features)
sat_dummies = pd.get_dummies(day8_data['sat_id'], prefix='sat')
# Ensure all 32 satellites are present
for i in range(1, 33):
    col = f'sat_G{i:02d}'
    if col not in sat_dummies.columns:
        sat_dummies[col] = 0
sat_dummies = sat_dummies[[f'sat_G{i:02d}' for i in range(1, 33)]]
day8_data = pd.concat([day8_data, sat_dummies], axis=1)
print("✅ Satellite features: 32")

# 3. Keplerian elements (7 features) - already in data
keplerian_cols = ['sqrtA', 'a', 'e', 'i', 'RAAN', 'omega', 'M']
print("✅ Keplerian features: 7")

# 4. Velocity proxy features (4 features)
day8_data['e_sin_M'] = day8_data['e'] * np.sin(np.radians(day8_data['M']))
day8_data['e_cos_M'] = day8_data['e'] * np.cos(np.radians(day8_data['M']))
day8_data['sqrt_a_e'] = np.sqrt(day8_data['a']) * day8_data['e']
day8_data['i_over_RAAN'] = day8_data['i'] / (day8_data['RAAN'] + 1e-6)
print("✅ Velocity features: 4")

# Prepare feature matrix
time_features = ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 
                 'day_sin', 'day_cos', 'doy_sin', 'doy_cos', 'month_sin']
sat_features = [f'sat_G{i:02d}' for i in range(1, 33)]
velocity_features = ['e_sin_M', 'e_cos_M', 'sqrt_a_e', 'i_over_RAAN']

all_features = time_features + sat_features + keplerian_cols + velocity_features
X_day8 = day8_data[all_features].values

print(f"\n✅ Total features prepared: {X_day8.shape[1]}")
print(f"✅ Feature matrix shape: {X_day8.shape}")

# ============================================================================
# STEP 4: Scale Features
# ============================================================================
print("\n⚖️  STEP 4: Scaling Features...")

# Apply same scaling as training
n_time = 9
n_sat = 32
n_kepler = 7

X_time = scalers['time_scaler'].transform(X_day8[:, :n_time])
X_sat = X_day8[:, n_time:n_time+n_sat]  # One-hot, no scaling
X_kepler = scalers['kep_scaler'].transform(X_day8[:, n_time+n_sat:n_time+n_sat+n_kepler])
X_velocity = scalers['vel_scaler'].transform(X_day8[:, -4:])

X_day8_scaled = np.hstack([X_time, X_sat, X_kepler, X_velocity])
print(f"✅ Features scaled: {X_day8_scaled.shape}")

# Ensure float32 dtype
X_day8_scaled = X_day8_scaled.astype(np.float32)

# ============================================================================
# STEP 5: Make Predictions
# ============================================================================
print("\n🤖 STEP 5: Predicting Errors for Day 8...")

predictions = model.predict(X_day8_scaled, verbose=0)
print(f"✅ Predictions generated: {predictions.shape}")

# Inverse transform predictions
predictions_unscaled = scalers['y_scaler'].inverse_transform(predictions)

# Create results DataFrame
results = pd.DataFrame({
    'timestamp': day8_data['timestamp'].values,
    'sat_id': day8_data['sat_id'].values,
    'X_Error': predictions_unscaled[:, 0],
    'Y_Error': predictions_unscaled[:, 1],
    'Z_Error': predictions_unscaled[:, 2],
    'Clock_Error': predictions_unscaled[:, 3],
    'X_Error_actual': actual_errors['X_Error'].values,
    'Y_Error_actual': actual_errors['Y_Error'].values,
    'Z_Error_actual': actual_errors['Z_Error'].values,
    'Clock_Error_actual': actual_errors['Clock_Error'].values
})

# ============================================================================
# STEP 6: Save Predictions
# ============================================================================
print("\n💾 STEP 6: Saving Predictions...")

output_file = 'outputs/predicted_errors_day8.csv'
results.to_csv(output_file, index=False)
print(f"✅ Saved to: {output_file}")

# Calculate metrics
print("\n📈 PREDICTION METRICS:")
for col in ['X_Error', 'Y_Error', 'Z_Error', 'Clock_Error']:
    actual = results[f'{col}_actual']
    pred = results[col]
    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred)**2))
    print(f"   {col:15s} MAE: {mae:10.2f}m   RMSE: {rmse:10.2f}m")

# ============================================================================
# STEP 7: Generate Visualizations
# ============================================================================
print("\n📊 STEP 7: Generating Visual Results...")

# Figure 1: Actual vs Predicted Line Charts
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Day 8 Predictions: Actual vs Predicted Errors', fontsize=16, fontweight='bold')

error_types = ['X_Error', 'Y_Error', 'Z_Error', 'Clock_Error']
titles = ['X-Axis Position Error', 'Y-Axis Position Error', 'Z-Axis Position Error', 'Clock Bias Error']

for idx, (err, title) in enumerate(zip(error_types, titles)):
    ax = axes[idx // 2, idx % 2]
    
    actual = results[f'{err}_actual']
    pred = results[err]
    
    ax.plot(range(len(actual)), actual, 'o-', label='Actual', linewidth=2, markersize=5, alpha=0.7)
    ax.plot(range(len(pred)), pred, 's-', label='Predicted', linewidth=2, markersize=4, alpha=0.7)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Sample Index', fontsize=10)
    ax.set_ylabel('Error (meters)', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add MAE annotation
    mae = np.mean(np.abs(actual - pred))
    ax.text(0.02, 0.98, f'MAE: {mae:.2f}m', transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('outputs/figures/day8_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("✅ Saved: outputs/figures/day8_actual_vs_predicted.png")
plt.close()

# Figure 2: Residuals Histograms
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Day 8 Prediction Residuals Distribution', fontsize=16, fontweight='bold')

for idx, (err, title) in enumerate(zip(error_types, titles)):
    ax = axes[idx // 2, idx % 2]
    
    actual = results[f'{err}_actual']
    pred = results[err]
    residuals = actual - pred
    
    ax.hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(np.mean(residuals), color='green', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(residuals):.2f}m')
    
    ax.set_title(f'{title} - Residuals', fontsize=12, fontweight='bold')
    ax.set_xlabel('Residual (Actual - Predicted) [meters]', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Stats
    stats_text = f'Mean: {np.mean(residuals):.2f}m\nStd: {np.std(residuals):.2f}m'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

plt.tight_layout()
plt.savefig('outputs/figures/day8_residuals_histogram.png', dpi=300, bbox_inches='tight')
print("✅ Saved: outputs/figures/day8_residuals_histogram.png")
plt.close()

# Figure 3: Clock Drift Over Time
fig, ax = plt.subplots(1, 1, figsize=(16, 6))

# Convert timestamps to hours
results['hour'] = pd.to_datetime(results['timestamp']).dt.hour + \
                  pd.to_datetime(results['timestamp']).dt.minute / 60

ax.plot(results['hour'], results['Clock_Error_actual'], 'o-', 
        label='Actual Clock Error', linewidth=2, markersize=6, alpha=0.7)
ax.plot(results['hour'], results['Clock_Error'], 's-', 
        label='Predicted Clock Error', linewidth=2, markersize=5, alpha=0.7)

ax.set_title('Day 8: GPS Satellite Clock Drift Over Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Time (hours)', fontsize=12)
ax.set_ylabel('Clock Error (meters)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Add shaded error band
mae = np.mean(np.abs(results['Clock_Error_actual'] - results['Clock_Error']))
ax.fill_between(results['hour'], 
                results['Clock_Error'] - mae, 
                results['Clock_Error'] + mae, 
                alpha=0.2, label=f'±MAE Band ({mae:.0f}m)')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('outputs/figures/day8_clock_drift_over_time.png', dpi=300, bbox_inches='tight')
print("✅ Saved: outputs/figures/day8_clock_drift_over_time.png")
plt.close()

# Figure 4: Scatter Plots - Actual vs Predicted
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Day 8: Actual vs Predicted Scatter Plots', fontsize=16, fontweight='bold')

for idx, (err, title) in enumerate(zip(error_types, titles)):
    ax = axes[idx // 2, idx % 2]
    
    actual = results[f'{err}_actual']
    pred = results[err]
    
    ax.scatter(actual, pred, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(actual.min(), pred.min())
    max_val = max(actual.max(), pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Actual Error (meters)', fontsize=10)
    ax.set_ylabel('Predicted Error (meters)', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Calculate R²
    ss_res = np.sum((actual - pred)**2)
    ss_tot = np.sum((actual - actual.mean())**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig('outputs/figures/day8_scatter_plots.png', dpi=300, bbox_inches='tight')
print("✅ Saved: outputs/figures/day8_scatter_plots.png")
plt.close()

# Figure 5: Error by Satellite
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Day 8: Prediction Error by Satellite', fontsize=16, fontweight='bold')

for idx, (err, title) in enumerate(zip(error_types, titles)):
    ax = axes[idx // 2, idx % 2]
    
    # Calculate MAE per satellite
    sat_mae = []
    sat_ids = []
    for sat in sorted(results['sat_id'].unique()):
        sat_data = results[results['sat_id'] == sat]
        mae = np.mean(np.abs(sat_data[f'{err}_actual'] - sat_data[err]))
        sat_mae.append(mae)
        sat_ids.append(sat)
    
    bars = ax.bar(range(len(sat_ids)), sat_mae, alpha=0.7, edgecolor='black')
    ax.set_title(f'{title} - MAE by Satellite', fontsize=12, fontweight='bold')
    ax.set_xlabel('Satellite ID', fontsize=10)
    ax.set_ylabel('MAE (meters)', fontsize=10)
    ax.set_xticks(range(len(sat_ids)))
    ax.set_xticklabels(sat_ids, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Color bars by performance
    mean_mae = np.mean(sat_mae)
    for i, (bar, mae) in enumerate(zip(bars, sat_mae)):
        if mae < mean_mae:
            bar.set_color('green')
            bar.set_alpha(0.6)
        else:
            bar.set_color('orange')
            bar.set_alpha(0.6)
    
    ax.axhline(mean_mae, color='red', linestyle='--', linewidth=2, 
               label=f'Mean MAE: {mean_mae:.2f}m')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('outputs/figures/day8_error_by_satellite.png', dpi=300, bbox_inches='tight')
print("✅ Saved: outputs/figures/day8_error_by_satellite.png")
plt.close()

# ============================================================================
# STEP 8: Generate Summary Report
# ============================================================================
print("\n📝 STEP 8: Generating Summary Report...")

summary_file = 'outputs/day8_prediction_summary.txt'
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("   GNSS DAY 8 ERROR PREDICTION - SMART INDIA HACKATHON SUBMISSION\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("DATASET INFORMATION:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total Day 8 Samples: {len(results)}\n")
    f.write(f"Unique Satellites: {results['sat_id'].nunique()}\n")
    f.write(f"Time Range: {results['timestamp'].min()} to {results['timestamp'].max()}\n\n")
    
    f.write("MODEL INFORMATION:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Model: Enhanced Deep Neural Network\n")
    f.write(f"Architecture: 256→128→64→32→4 with BatchNorm & Dropout\n")
    f.write(f"Total Parameters: {model.count_params():,}\n")
    f.write(f"Input Features: 52 (9 time + 32 satellite + 7 Keplerian + 4 velocity)\n\n")
    
    f.write("PREDICTION PERFORMANCE:\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Error Type':<20} {'MAE (m)':<15} {'RMSE (m)':<15} {'R²':<10}\n")
    f.write("-" * 80 + "\n")
    
    for col in error_types:
        actual = results[f'{col}_actual']
        pred = results[col]
        mae = np.mean(np.abs(actual - pred))
        rmse = np.sqrt(np.mean((actual - pred)**2))
        ss_res = np.sum((actual - pred)**2)
        ss_tot = np.sum((actual - actual.mean())**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        f.write(f"{col:<20} {mae:<15.2f} {rmse:<15.2f} {r2:<10.4f}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("DELIVERABLES:\n")
    f.write("-" * 80 + "\n")
    f.write("✅ outputs/predicted_errors_day8.csv - Prediction results\n")
    f.write("✅ outputs/figures/day8_actual_vs_predicted.png - Line charts\n")
    f.write("✅ outputs/figures/day8_residuals_histogram.png - Error distribution\n")
    f.write("✅ outputs/figures/day8_clock_drift_over_time.png - Clock analysis\n")
    f.write("✅ outputs/figures/day8_scatter_plots.png - Correlation plots\n")
    f.write("✅ outputs/figures/day8_error_by_satellite.png - Satellite-wise analysis\n")
    f.write("✅ outputs/day8_prediction_summary.txt - This report\n")
    f.write("=" * 80 + "\n")

print(f"✅ Saved: {summary_file}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ DAY 8 PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\n📊 Predicted {len(results)} samples for {results['sat_id'].nunique()} satellites")
print(f"📁 All outputs saved in: outputs/")
print(f"🖼️  All figures saved in: outputs/figures/")
print("\n🎯 READY FOR SIH SUBMISSION!\n")
print("=" * 80)
