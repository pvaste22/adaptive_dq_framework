# scripts/data_inspector.py
"""
Simple data inspector to examine VIAVI CSV files before processing.
This helps us understand the data structure without complex processing.
"""

import sys
import os
import pandas as pd
from pathlib import Path

def inspect_csv_file(file_path, file_type):
    """Simple inspection of a CSV file"""
    print(f"\n{'='*50}")
    print(f"INSPECTING {file_type.upper()} FILE")
    print(f"{'='*50}")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        print(f"Please place your VIAVI CSV files in the data/raw/ directory")
        return False
    
    try:
        # Read just first few rows to inspect structure
        df = pd.read_csv(file_path, nrows=10)
        
        print(f"File: {file_path}")
        print(f"Columns ({len(df.columns)}): {list(df.columns)}")
        print(f"First 3 rows:")
        print(df.head(3).to_string())
        
        # Check file size
        full_df = pd.read_csv(file_path)
        print(f"\nTotal rows: {len(full_df)}")
        
        if file_type == "cell":
            if 'Viavi.Cell.Name' in full_df.columns:
                unique_cells = full_df['Viavi.Cell.Name'].nunique()
                cell_examples = full_df['Viavi.Cell.Name'].unique()[:5]
                print(f"Unique cells: {unique_cells} (expected: 52)")
                print(f"Cell examples: {cell_examples}")
            else:
                print("Warning: 'Viavi.Cell.Name' column not found")
                
        elif file_type == "ue":
            if 'Viavi.UE.Name' in full_df.columns:
                unique_ues = full_df['Viavi.UE.Name'].nunique()
                ue_examples = full_df['Viavi.UE.Name'].unique()[:5]
                print(f"Unique UEs: {unique_ues} (expected: 48)")
                print(f"UE examples: {ue_examples}")
            else:
                print("Warning: 'Viavi.UE.Name' column not found")
        
        # Check timestamp column
        if 'timestamp' in full_df.columns:
            print(f"Timestamp range: {full_df['timestamp'].min()} to {full_df['timestamp'].max()}")
            # Convert to datetime for readability
            try:
                ts_start = pd.to_datetime(full_df['timestamp'].min(), unit='s')
                ts_end = pd.to_datetime(full_df['timestamp'].max(), unit='s')
                print(f"Date range: {ts_start} to {ts_end}")
            except:
                print("Could not convert timestamps to dates")
        
        # Check for critical columns based on documentation
        if file_type == "cell":
            critical_cols = ['RRU.PrbUsedDl', 'RRU.PrbTotDl', 'PEE.Energy']
            for col in critical_cols:
                if col in full_df.columns:
                    sample_vals = full_df[col].head(3).values
                    print(f"{col} samples: {sample_vals}")
                else:
                    print(f"Warning: Critical column missing: {col}")
        
        return True
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

def main():
    """Main inspection function"""
    print("VIAVI Dataset Inspector")
    print("This script examines your CSV files without processing them")
    
    # Default file paths
    project_root = Path.cwd()
    cell_file = project_root / 'data' / 'raw' / 'CellReports_v0.csv'
    ue_file = project_root / 'data' / 'raw' / 'UEReports_v0.csv'
    
    # Check cell reports
    cell_ok = inspect_csv_file(cell_file, "cell")
    
    # Check UE reports  
    ue_ok = inspect_csv_file(ue_file, "ue")
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    
    if cell_ok and ue_ok:
        print("Both files found and readable!")
        print("Ready for next step: Create unit converter")
    else:
        print("Issues found with data files")
        print("Make sure you have:")
        print("- data/raw/CellReports_v0.csv")
        print("- data/raw/UEReports_v0.csv")

if __name__ == "__main__":
    main()