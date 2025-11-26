import pandas as pd

df = pd.read_csv('GNSS_kepler_elements_clean.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date

unique_dates = df['date'].unique()

print(f'Total days of data: {len(unique_dates)} days')
print(f'\nDate range: {df["timestamp"].min()} to {df["timestamp"].max()}')
print(f'\nBreakdown by date:')
for date in sorted(unique_dates):
    count = len(df[df['date'] == date])
    print(f'  {date}: {count} records')
