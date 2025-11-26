# -*- coding: utf-8 -*-
"""
Cleanup Script - Remove Unnecessary Files
This script helps clean up old datasets and legacy models that have been superseded.

WHAT WILL BE REMOVED:
1. Old error datasets without UTC timestamps
2. Legacy model files that are superseded
3. Intermediate data files (optional - commented out by default)

WHAT WILL BE KEPT:
- XGBoost models (BEST performers)
- Enhanced deep learning models
- UTC-based datasets and models
- Master dataset (real_data.csv)
- All documentation files
"""

import os
import shutil

print("=" * 70)
print("GNSS ERROR PREDICTION - CLEANUP SCRIPT")
print("=" * 70)

# Track what's being removed
removed_files = []
errors = []

# ============================================================================
# CATEGORY 1: Old Error Datasets (Without UTC Timestamps)
# ============================================================================
print("\n[CATEGORY 1] Old error datasets (without UTC timestamps)")
print("These are replaced by train/test/validation_errors_utc.csv")

old_datasets = [
    '../data/splits/train_errors.csv',
    '../data/splits/test_errors.csv',
    '../data/splits/validation_errors.csv'
]

for file_path in old_datasets:
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            removed_files.append(file_path)
            print(f"  ✓ Removed: {file_path}")
        except Exception as e:
            errors.append(f"Error removing {file_path}: {e}")
            print(f"  ✗ Error: {file_path} - {e}")
    else:
        print(f"  - Not found: {file_path}")

# ============================================================================
# CATEGORY 2: Legacy Model Files
# ============================================================================
print("\n[CATEGORY 2] Legacy model files (superseded by XGBoost/Enhanced)")

legacy_models = [
    '../03_models/gnss_error_model.keras',
    '../03_models/best_gnss_error_model.keras',
    '../03_models/best_gnss_model.keras',
    '../03_models/scaler_errors.pkl',
    '../03_models/scaler.pkl'
]

for file_path in legacy_models:
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            removed_files.append(file_path)
            print(f"  ✓ Removed: {file_path}")
        except Exception as e:
            errors.append(f"Error removing {file_path}: {e}")
            print(f"  ✗ Error: {file_path} - {e}")
    else:
        print(f"  - Not found: {file_path}")

# ============================================================================
# CATEGORY 3: Duplicate Output Files
# ============================================================================
print("\n[CATEGORY 3] Duplicate output files in 04_results/")

duplicate_outputs = [
    '../04_results/gnss_error_data.csv',  # Duplicate of data/processed/
    '../04_results/day8_predictions.csv'   # Old predictions
]

for file_path in duplicate_outputs:
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            removed_files.append(file_path)
            print(f"  ✓ Removed: {file_path}")
        except Exception as e:
            errors.append(f"Error removing {file_path}: {e}")
            print(f"  ✗ Error: {file_path} - {e}")
    else:
        print(f"  - Not found: {file_path}")

# ============================================================================
# CATEGORY 4: Intermediate Data (OPTIONAL - Commented Out by Default)
# ============================================================================
print("\n[CATEGORY 4] Intermediate data files (SKIPPED - uncomment to remove)")
print("These files were used during data processing but aren't needed for inference.")
print("Uncomment the code below if you want to remove them.")

# UNCOMMENT TO REMOVE INTERMEDIATE FILES:
"""
intermediate_files = [
    '../data/intermediate/gnss_error_output.csv',
    '../data/intermediate/GNSS_kepler_elements.csv',
    '../data/intermediate/GNSS_kepler_elements_clean.csv',
    '../data/intermediate/real_gnss_clock_errors.csv'
]

for file_path in intermediate_files:
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            removed_files.append(file_path)
            print(f"  ✓ Removed: {file_path}")
        except Exception as e:
            errors.append(f"Error removing {file_path}: {e}")
            print(f"  ✗ Error: {file_path} - {e}")
    else:
        print(f"  - Not found: {file_path}")
"""
print("  [SKIPPED] Keeping intermediate files for reproducibility")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("CLEANUP SUMMARY")
print("=" * 70)

print(f"\n✓ Successfully removed: {len(removed_files)} files")
for file_path in removed_files:
    print(f"  - {os.path.basename(file_path)}")

if errors:
    print(f"\n✗ Errors encountered: {len(errors)}")
    for error in errors:
        print(f"  - {error}")
else:
    print(f"\n✓ No errors encountered")

# ============================================================================
# REMAINING FILES SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("REMAINING IMPORTANT FILES")
print("=" * 70)

print("\n📊 DATASETS:")
print("  ✓ real_data.csv                  - Master dataset (4,310 records)")
print("  ✓ train_errors_utc.csv           - Training set with UTC (3,448 rows)")
print("  ✓ test_errors_utc.csv            - Test set with UTC (646 rows)")
print("  ✓ validation_errors_utc.csv      - Validation set with UTC (216 rows)")

print("\n🏆 BEST MODELS (XGBoost):")
print("  ✓ xgb_x_error_model.pkl          - Position X prediction")
print("  ✓ xgb_y_error_model.pkl          - Position Y prediction")
print("  ✓ xgb_z_error_model.pkl          - Position Z prediction")
print("  ✓ xgb_clock_model.pkl            - Clock error prediction")
print("  ✓ scaler_enhanced.pkl            - Feature normalization")
print("  ✓ label_encoder_sat.pkl          - Satellite ID encoding")
print("  ✓ feature_names_enhanced.pkl     - Feature order")

print("\n🧠 DEEP LEARNING MODELS:")
print("  ✓ position_model.keras           - Enhanced position model")
print("  ✓ clock_model.keras              - Enhanced clock model")
print("  ✓ best_position_model.keras      - Best position checkpoint")
print("  ✓ best_clock_model.keras         - Best clock checkpoint")

print("\n⏰ UTC-ONLY MODELS:")
print("  ✓ gnss_utc_model.keras           - Time-only features model")
print("  ✓ best_utc_model.keras           - Best UTC checkpoint")
print("  ✓ scaler_utc.pkl                 - UTC scaler")
print("  ✓ features_utc.pkl               - UTC features")

print("\n📖 DOCUMENTATION:")
print("  ✓ PROJECT_SUMMARY.md             - Complete overview (you are here!)")
print("  ✓ MODEL_COMPARISON_REPORT.md     - Detailed 3-model comparison")
print("  ✓ BEST_MODEL_GUIDE.md            - XGBoost deployment guide")

print("\n💾 TOTAL DISK SPACE:")
# Calculate approximate sizes
models_size = 0
data_size = 0

# Count model files
for root, dirs, files in os.walk('../03_models'):
    for file in files:
        file_path = os.path.join(root, file)
        if os.path.exists(file_path):
            models_size += os.path.getsize(file_path)

# Count data files
for root, dirs, files in os.walk('../data'):
    for file in files:
        file_path = os.path.join(root, file)
        if os.path.exists(file_path):
            data_size += os.path.getsize(file_path)

print(f"  Models:   {models_size / (1024*1024):.1f} MB")
print(f"  Data:     {data_size / (1024*1024):.1f} MB")
print(f"  Total:    {(models_size + data_size) / (1024*1024):.1f} MB")

print("\n" + "=" * 70)
print("✅ CLEANUP COMPLETE!")
print("=" * 70)

print("\nNEXT STEPS:")
print("1. Review PROJECT_SUMMARY.md for complete overview")
print("2. Use XGBoost models for production (best performance)")
print("3. Check MODEL_COMPARISON_REPORT.md for detailed analysis")
print("4. See BEST_MODEL_GUIDE.md for deployment examples")
