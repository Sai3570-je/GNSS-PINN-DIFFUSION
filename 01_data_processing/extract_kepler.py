import os
import georinex as gr
import pandas as pd

# Parse RINEX files and extract Keplerian elements
def extract_kepler_elements(rinex_dir):
    records = []
    
    for file in sorted(os.listdir(rinex_dir)):
        if file.endswith('.rnx'):
            file_path = os.path.join(rinex_dir, file)
            print(f'Processing {file}...')
            
            try:
                nav = gr.load(file_path, use='G')  # Only load GPS satellites
                
                # Get available variables
                print(f'  Available fields: {list(nav.data_vars)[:10]}...')
                
                for sv in nav.sv.values:
                    svdata = nav.sel(sv=sv)
                    for time_idx in range(len(svdata.time)):
                        d = svdata.isel(time=time_idx)
                        
                        try:
                            row = {
                                'timestamp': pd.to_datetime(str(d.time.values)),
                                'sat_id': str(sv),
                                'sqrtA': float(d.sqrtA.values),
                                'a': float(d.sqrtA.values)**2,
                                'e': float(d.Eccentricity.values),
                                'i': float(d.Io.values),
                                'RAAN': float(d.Omega0.values),
                                'omega': float(d.omega.values),
                                'M': float(d.M0.values)
                            }
                            records.append(row)
                        except Exception as e:
                            print(f'  Error extracting data for {sv} at time {time_idx}: {e}')
                            continue
                
                print(f'  Extracted {len([r for r in records if file in str(r)])} records from {file}')
                
            except Exception as e:
                print(f'  Error loading {file}: {e}')
                continue
    
    return pd.DataFrame(records)

# Run extraction
df = extract_kepler_elements('data/rinex_nav')
print(f'\nTotal records extracted: {len(df)}')

if len(df) > 0:
    df.to_csv('GNSS_kepler_elements.csv', index=False)
    print(f'Saved to GNSS_kepler_elements.csv')
    print(f'\nSample data:')
    print(df.head(10))
    print(f'\nUnique satellites: {df["sat_id"].nunique()}')
else:
    print('No records extracted')
