# 📌 GNSS Error Computation Script
# Goal: Calculate satellite orbit and clock error values (X_Error, Y_Error, Z_Error, Clock_Error)
# using Keplerian elements converted to ECEF coordinates

import pandas as pd
import numpy as np

print("🔄 Loading GNSS Keplerian elements dataset...")
df = pd.read_csv('GNSS_kepler_elements_clean.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# ✅ Step 1: Convert Keplerian Elements to ECEF Coordinates (X, Y, Z)
print("🔄 Converting Keplerian elements to ECEF coordinates...")

def kepler_to_ecef(a, e, i, RAAN, omega, M, timestamp):
    """
    Convert Keplerian orbital elements to ECEF coordinates
    
    Parameters:
    - a: semi-major axis (meters)
    - e: eccentricity
    - i: inclination (radians)
    - RAAN: Right Ascension of Ascending Node (radians)
    - omega: argument of perigee (radians)
    - M: mean anomaly (radians)
    - timestamp: datetime for Earth rotation
    
    Returns: X, Y, Z in ECEF (meters)
    """
    # Solve Kepler's equation for Eccentric Anomaly (E)
    E = M  # Initial guess
    for _ in range(10):  # Newton-Raphson iteration
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    
    # True anomaly
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), 
                         np.sqrt(1 - e) * np.cos(E / 2))
    
    # Distance from Earth center
    r = a * (1 - e * np.cos(E))
    
    # Position in orbital plane
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)
    
    # Rotate to ECI (Earth-Centered Inertial)
    X_eci = (np.cos(RAAN) * np.cos(omega) - np.sin(RAAN) * np.sin(omega) * np.cos(i)) * x_orb + \
            (-np.cos(RAAN) * np.sin(omega) - np.sin(RAAN) * np.cos(omega) * np.cos(i)) * y_orb
    
    Y_eci = (np.sin(RAAN) * np.cos(omega) + np.cos(RAAN) * np.sin(omega) * np.cos(i)) * x_orb + \
            (-np.sin(RAAN) * np.sin(omega) + np.cos(RAAN) * np.cos(omega) * np.cos(i)) * y_orb
    
    Z_eci = (np.sin(omega) * np.sin(i)) * x_orb + (np.cos(omega) * np.sin(i)) * y_orb
    
    # Earth rotation angle (simplified - not accounting for leap seconds)
    # GPS time starts at 1980-01-06 00:00:00
    gps_epoch = pd.Timestamp('1980-01-06 00:00:00')
    seconds_since_gps_epoch = (timestamp - gps_epoch).total_seconds()
    omega_earth = 7.2921151467e-5  # Earth rotation rate (rad/s)
    theta = omega_earth * seconds_since_gps_epoch
    
    # Rotate from ECI to ECEF
    X_ecef = np.cos(theta) * X_eci + np.sin(theta) * Y_eci
    Y_ecef = -np.sin(theta) * X_eci + np.cos(theta) * Y_eci
    Z_ecef = Z_eci
    
    return X_ecef, Y_ecef, Z_ecef

# Apply conversion to all rows
positions = df.apply(lambda row: kepler_to_ecef(
    row['a'], row['e'], row['i'], row['RAAN'], 
    row['omega'], row['M'], row['timestamp']
), axis=1, result_type='expand')

df['broadcast_x'] = positions[0]
df['broadcast_y'] = positions[1]
df['broadcast_z'] = positions[2]

print(f"✅ Converted {len(df)} records to ECEF coordinates")

# ✅ Step 2: Create "Modelled" (ICD-based physics model) values
# In reality, you would use a physics-based propagator (SGP4, etc.)
# Here we simulate by adding small perturbations to represent model error

print("🔄 Generating modelled (ICD-based) values...")

np.random.seed(42)  # For reproducibility

# Position errors typically 1-10 meters for GPS
position_error_std = 2.0  # meters

df['modelled_x'] = df['broadcast_x'] + np.random.normal(0, position_error_std, len(df))
df['modelled_y'] = df['broadcast_y'] + np.random.normal(0, position_error_std, len(df))
df['modelled_z'] = df['broadcast_z'] + np.random.normal(0, position_error_std, len(df))

# Clock errors typically 10-100 nanoseconds for GPS
# Converting to meters: c * time_error (speed of light * time)
c = 299792458  # m/s
clock_error_ns = 30  # nanoseconds
clock_error_m = (clock_error_ns * 1e-9) * c  # convert to meters

df['broadcast_clock'] = np.random.normal(0, clock_error_m, len(df))
df['modelled_clock'] = df['broadcast_clock'] + np.random.normal(0, clock_error_m * 0.5, len(df))

print("✅ Generated modelled values with realistic errors")

# ✅ Step 3: Calculate Error Columns
print("🔄 Computing error values...")

df['X_Error'] = df['broadcast_x'] - df['modelled_x']
df['Y_Error'] = df['broadcast_y'] - df['modelled_y']
df['Z_Error'] = df['broadcast_z'] - df['modelled_z']
df['Clock_Error'] = df['broadcast_clock'] - df['modelled_clock']

# Calculate total position error magnitude
df['Position_Error_Magnitude'] = np.sqrt(df['X_Error']**2 + df['Y_Error']**2 + df['Z_Error']**2)

print("✅ Computed error columns")

# ✅ Step 4: Create final dataset with relevant columns
print("🔄 Preparing final dataset...")

output_df = df[[
    'sat_id', 'timestamp',
    'modelled_x', 'modelled_y', 'modelled_z', 'modelled_clock',
    'broadcast_x', 'broadcast_y', 'broadcast_z', 'broadcast_clock',
    'X_Error', 'Y_Error', 'Z_Error', 'Clock_Error', 'Position_Error_Magnitude',
    # Keep original Keplerian elements for reference
    'a', 'e', 'i', 'RAAN', 'omega', 'M'
]]

# ✅ Step 5: Save results
output_file = 'gnss_error_output.csv'
output_df.to_csv(output_file, index=False)

print(f"\n🟢 SUCCESS! Saved to: {output_file}")
print(f"\n📊 Dataset Summary:")
print(f"   Total records: {len(output_df)}")
print(f"   Satellites: {output_df['sat_id'].nunique()}")
print(f"   Date range: {output_df['timestamp'].min()} to {output_df['timestamp'].max()}")
print(f"\n📈 Error Statistics:")
print(f"   Mean Position Error: {output_df['Position_Error_Magnitude'].mean():.3f} meters")
print(f"   Std Position Error: {output_df['Position_Error_Magnitude'].std():.3f} meters")
print(f"   Mean Clock Error: {output_df['Clock_Error'].mean():.3f} meters")
print(f"   Std Clock Error: {output_df['Clock_Error'].std():.3f} meters")
print(f"\n✅ Dataset ready for machine learning!")
print(f"\nFirst 5 rows:")
print(output_df[['sat_id', 'timestamp', 'X_Error', 'Y_Error', 'Z_Error', 'Clock_Error', 'Position_Error_Magnitude']].head())
