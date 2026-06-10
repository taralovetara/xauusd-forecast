"""
Merge raw XAUUSD 5-min XLSX files into one clean CSV.
Files are semicolon-separated: Date;Open;High;Low;Close;Volume
"""
import pandas as pd
import os
import glob

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
OUTPUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'xauusd_5m_merged.csv')

def merge_files(raw_dir=None, output=None):
    raw_dir = raw_dir or RAW_DIR
    output = output or OUTPUT
    
    all_dfs = []
    files = sorted(glob.glob(os.path.join(raw_dir, 'xau5m*.xlsx')))
    
    for f in files:
        df = pd.read_excel(f)
        valid = df.iloc[:, 0].dropna()
        parsed = valid.str.split(';', expand=True)
        parsed.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        parsed['datetime'] = pd.to_datetime(parsed['datetime'], format='%Y.%m.%d %H:%M')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            parsed[col] = pd.to_numeric(parsed[col], errors='coerce')
        parsed = parsed.dropna()
        all_dfs.append(parsed)
        print(f"  {os.path.basename(f)}: {len(parsed):,} rows")
    
    merged = pd.concat(all_dfs, ignore_index=True)
    merged = merged.sort_values('datetime').drop_duplicates(subset='datetime', keep='first').reset_index(drop=True)
    merged.to_csv(output, index=False)
    print(f"\nMerged: {len(merged):,} rows → {output}")
    return merged

if __name__ == '__main__':
    merge_files()
