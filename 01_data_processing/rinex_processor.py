import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import getpass
import gzip
import shutil
import georinex as gr

# User configuration
USERNAME = 'sai3570'
PASSWORD = 'Sai@35703570'
START_DATE = '2024-01-01'
END_DATE = '2024-01-07'
OUTPUT_DIR = 'data/rinex_nav'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Utility: Build RINEX CDDIS URL - Using alternative source (IGS)
def build_rinex_url(date):
    day_of_year = date.timetuple().tm_yday
    year_short = date.strftime('%y')
    # Using IGS (International GNSS Service) as alternative to CDDIS
    return (
        f"https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{date.year}/{day_of_year:03d}/"
        f"BRDC00IGS_R_{date.year}{day_of_year:03d}0000_01D_MN.rnx.gz"
    )

# (a) Download function - IGS server doesn't require authentication
def download_rinex_nav_files(start_date, end_date, output_dir):
    session = requests.Session()
    
    dates = pd.date_range(start=start_date, end=end_date)
    for date in dates:
        url = build_rinex_url(date)
        day_of_year = date.timetuple().tm_yday
        local_file = os.path.join(output_dir, f"BRDC00IGS_R_{date.year}{day_of_year:03d}0000_01D_MN.rnx.gz")
        
        if os.path.exists(local_file) and os.path.getsize(local_file) > 1000:
            print(f'Already downloaded: {os.path.basename(local_file)}')
            continue
            
        print(f'Downloading {url}')
        try:
            r = session.get(url, timeout=60)
            
            if r.status_code == 200:
                with open(local_file, 'wb') as f:
                    f.write(r.content)
                print(f'Downloaded: {os.path.basename(local_file)} ({len(r.content)} bytes)')
            else:
                print(f'Failed: HTTP {r.status_code} - {url}')
        except Exception as e:
            print(f'Error downloading {url}: {e}')

# (b) Decompress .gz and .Z files
def decompress_files(output_dir):
    for file in os.listdir(output_dir):
        if file.endswith('.gz'):
            file_path = os.path.join(output_dir, file)
            output_path = file_path[:-3]  # Remove .gz extension
            
            if os.path.exists(output_path):
                print(f'Already decompressed: {file}')
                continue
                
            print(f'Decompressing {file}')
            try:
                with gzip.open(file_path, 'rb') as compressed:
                    with open(output_path, 'wb') as decompressed:
                        shutil.copyfileobj(compressed, decompressed)
                os.remove(file_path)
                print(f'Successfully decompressed: {file}')
            except Exception as e:
                print(f'Decompression failed for {file}: {e}')
                
        elif file.endswith('.Z'):
            file_path = os.path.join(output_dir, file)
            output_path = file_path[:-2]  # Remove .Z extension
            
            if os.path.exists(output_path):
                print(f'Already decompressed: {file}')
                continue
                
            print(f'Decompressing {file}')
            
            # Try gzip first
            try:
                with gzip.open(file_path, 'rb') as compressed:
                    with open(output_path, 'wb') as decompressed:
                        shutil.copyfileobj(compressed, decompressed)
                os.remove(file_path)
                print(f'Successfully decompressed (gzip): {file}')
                continue
            except:
                pass
            
            # Try ncompress for Unix compress format
            try:
                import ncompress
                with open(file_path, 'rb') as compressed:
                    content = ncompress.decompress(compressed.read())
                with open(output_path, 'wb') as decompressed:
                    decompressed.write(content)
                os.remove(file_path)
                print(f'Successfully decompressed (ncompress): {file}')
            except Exception as e:
                print(f'Decompression failed for {file}: {e}')

# (c) Parse with georinex and extract Keplerian elements
def parse_rinex_and_extract_elements(output_dir):
    records = []
    for file in os.listdir(output_dir):
        if file.endswith('.rnx') or file.endswith('.24n') or (file.endswith('MN.rnx') == False and 'BRDC' in file):
            file_path = os.path.join(output_dir, file)
            print(f'Parsing {file}')
            try:
                nav = gr.load(file_path)
                for prn in nav.sv.values:
                    svdata = nav.sel(sv=prn)
                    for ts in svdata.time.values:
                        d = svdata.sel(time=ts)
                        row = {
                            'timestamp': pd.to_datetime(str(ts)),
                            'sat_id': str(prn),
                            'a': float(d.sqrtA.data)**2,
                            'e': float(d.Eccentricity.data),
                            'i': float(d.Io.data),
                            'RAAN': float(d.Omega0.data),
                            'omega': float(d.omega.data),
                            'M': float(d.M0.data)
                        }
                        records.append(row)
            except Exception as e:
                print(f'Error parsing {file}: {e}')
    return pd.DataFrame(records)

# (d) Write DataFrame to CSV
def write_to_csv(df, filename):
    df.to_csv(filename, index=False)

# Main workflow
if __name__ == '__main__':
    download_rinex_nav_files(START_DATE, END_DATE, OUTPUT_DIR)
    decompress_files(OUTPUT_DIR)
    df = parse_rinex_and_extract_elements(OUTPUT_DIR)
    write_to_csv(df, 'GNSS_kepler_elements.csv')
