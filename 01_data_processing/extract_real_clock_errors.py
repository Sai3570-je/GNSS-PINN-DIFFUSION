import os
import georinex as gr
import pandas as pd
import numpy as np
from datetime import datetime

print("🔄 Extracting REAL clock data from RINEX files...")

rinex_dir = 'data/rinex_nav'
records = []

for file in sorted(os.listdir(rinex_dir)):
    if file.endswith('.rnx'):
        file_path = os.path.join(rinex_dir, file)
        print(f'Processing {file}...')
        
        try:
            nav = gr.load(file_path, use='G')  # GPS only
            
            for sv in nav.sv.values:
                svdata = nav.sel(sv=sv)
                
                for time_idx in range(len(svdata.time)):
                    d = svdata.isel(time=time_idx)
                    
                    try:
                        timestamp = pd.to_datetime(str(d.time.values))
                        
                        # Extract REAL clock parameters from RINEX
                        a0 = float(d.SVclockBias.values)      # Clock bias (seconds)
                        a1 = float(d.SVclockDrift.values)     # Clock drift (sec/sec)
                        a2 = float(d.SVclockDriftRate.values) # Clock drift rate (sec/sec²)
                        
                        # Reference time for clock data
                        toc = float(d.TransTime.values)  # Time of clock (GPS seconds of week)
                        
                        # Get additional parameters
                        tgd = float(d.TGD.values) if 'TGD' in d else 0  # Group delay (seconds)
                        
                        # Calculate time difference from reference (in seconds)
                        # For simplicity, using 0 as we're at the epoch time
                        # In real application, you'd calculate t - toc properly
                        
                        # Satellite clock correction (in seconds)
                        dt = 0  # Time difference from toc (would be calculated from user time)
                        clock_correction_sec = a0 + a1 * dt + a2 * dt**2
                        
                        # Convert to meters (multiply by speed of light)
                        c = 299792458  # m/s
                        clock_bias_meters = clock_correction_sec * c
                        clock_drift_meters_per_sec = a1 * c
                        clock_drift_rate_meters_per_sec2 = a2 * c
                        tgd_meters = tgd * c
                        
                        # Relativistic correction (simplified)
                        # F = -2 * sqrt(GM) / c^2 ≈ -4.442807633e-10
                        sqrtA = float(d.sqrtA.values)
                        e = float(d.Eccentricity.values)
                        M0 = float(d.M0.values)
                        
                        # Solve for Eccentric Anomaly
                        E = M0
                        for _ in range(5):
                            E = E - (E - e * np.sin(E) - M0) / (1 - e * np.cos(E))
                        
                        F = -4.442807633e-10
                        relativistic_correction_sec = F * sqrtA**2 * e * np.sin(E)
                        relativistic_correction_meters = relativistic_correction_sec * c
                        
                        # Total clock error (in meters)
                        total_clock_error = clock_bias_meters + relativistic_correction_meters + tgd_meters
                        
                        row = {
                            'timestamp': timestamp,
                            'sat_id': str(sv),
                            'clock_bias_a0_sec': a0,
                            'clock_drift_a1_sec_per_sec': a1,
                            'clock_drift_rate_a2': a2,
                            'toc_gps_sec': toc,
                            'group_delay_tgd_sec': tgd,
                            'clock_bias_meters': clock_bias_meters,
                            'clock_drift_meters_per_sec': clock_drift_meters_per_sec,
                            'clock_drift_rate_meters_per_sec2': clock_drift_rate_meters_per_sec2,
                            'group_delay_meters': tgd_meters,
                            'relativistic_correction_meters': relativistic_correction_meters,
                            'total_clock_error_meters': total_clock_error,
                            # Include orbital elements for reference
                            'sqrtA': sqrtA,
                            'eccentricity': e,
                            'M0': M0
                        }
                        records.append(row)
                        
                    except Exception as e:
                        continue
            
            print(f'  ✓ Extracted {len([r for r in records if file in str(r.get("timestamp", ""))])} clock records')
                        
        except Exception as e:
            print(f'  ✗ Error: {e}')
            continue

# Create DataFrame
df = pd.DataFrame(records)

if len(df) > 0:
    # Save to CSV
    output_file = 'real_gnss_clock_errors.csv'
    df.to_csv(output_file, index=False)
    
    print(f'\n🟢 SUCCESS! Extracted REAL clock data')
    print(f'\n📊 Dataset Summary:')
    print(f'   Total records: {len(df)}')
    print(f'   Satellites: {df["sat_id"].nunique()}')
    print(f'   Date range: {df["timestamp"].min()} to {df["timestamp"].max()}')
    
    print(f'\n⏰ Clock Error Statistics (in meters):')
    print(f'   Clock Bias (a0):')
    print(f'     Mean: {df["clock_bias_meters"].mean():.3f} m')
    print(f'     Std:  {df["clock_bias_meters"].std():.3f} m')
    print(f'     Range: [{df["clock_bias_meters"].min():.3f}, {df["clock_bias_meters"].max():.3f}] m')
    
    print(f'\n   Total Clock Error (bias + relativistic + TGD):')
    print(f'     Mean: {df["total_clock_error_meters"].mean():.3f} m')
    print(f'     Std:  {df["total_clock_error_meters"].std():.3f} m')
    print(f'     Range: [{df["total_clock_error_meters"].min():.3f}, {df["total_clock_error_meters"].max():.3f}] m')
    
    print(f'\n   Relativistic Correction:')
    print(f'     Mean: {df["relativistic_correction_meters"].mean():.6f} m')
    print(f'     Range: [{df["relativistic_correction_meters"].min():.6f}, {df["relativistic_correction_meters"].max():.6f}] m')
    
    print(f'\n   Group Delay (TGD):')
    print(f'     Mean: {df["group_delay_meters"].mean():.3f} m')
    print(f'     Range: [{df["group_delay_meters"].min():.3f}, {df["group_delay_meters"].max():.3f}] m')
    
    print(f'\n💾 Saved to: {output_file}')
    print(f'\n📋 Sample of first 5 records:')
    print(df[['timestamp', 'sat_id', 'clock_bias_meters', 'relativistic_correction_meters', 
              'group_delay_meters', 'total_clock_error_meters']].head())
    
else:
    print('\n❌ No records extracted!')
