
import sys
from datetime import datetime
from common.logger import get_phase1_logger
from common.constants import PATHS
from data_processing.data_loader import DataLoader
from data_processing.unit_converter import UnitConverter
from data_processing.window_generator import WindowGenerator
from training.inject_faults import inject_faults_inplace_by_windows
from pathlib import Path

def main():
    log = get_phase1_logger("build_windows_only")
    log.info("="*60)
    log.info("WINDOW BUILD: load -> convert -> generate 5m windows -> save")
    log.info("="*60)

    # Step 1: Load raw data
    dl = DataLoader()
    cell_df, ue_df = dl.load_data()
    log.info(f"Loaded cells={len(cell_df)} rows, ues={len(ue_df)} rows")

    # Step 2: Unit conversions (same comprehensive call)
    uc = UnitConverter()
    cell_df, ue_df = uc.standardize_units_comprehensive(cell_df, ue_df)
    conv_sum = uc.get_conversion_summary(cell_df, ue_df)
    log.info(f"Conversions: cells={conv_sum['cell_data']['conversions_applied']} "
             f"| new_cols={conv_sum['cell_data']['new_columns_created']} "
             f"| flags={conv_sum['cell_data']['flags_added']}")
    # step 3: Fault injection
    cell_df2, cell_manifest = inject_faults_inplace_by_windows(
        cell_df,
        is_cell=True,
        ts_col="timestamp",
        window_seconds=300,
        overlap=0.8,
        faulty_window_fraction=0.15,     # 10–20% ke beech rakho
        timestamps_per_faulty_window=1,  # per faulty window 1 timestamp par fault
        rows_fraction_per_timestamp=0.10,# us ts group ka ~10% rows corrupt
        random_state=42,
    )

    ue_df2, ue_manifest = inject_faults_inplace_by_windows(
        ue_df,
        is_cell=False,
        ts_col="timestamp",
        window_seconds=300,
        overlap=0.8,
        faulty_window_fraction=0.15,
        timestamps_per_faulty_window=2, 
        rows_fraction_per_timestamp=0.25,
        random_state=42,
    )

    # Save manifests for verification
    cell_manifest.to_csv("./data/raw/CellReports_fault_manifest.csv", index=False)
    ue_manifest.to_csv("./data/raw/UEReports_fault_manifest.csv", index=False)
    cell_df2.to_csv("./data/raw/CellReports_fault_dt.csv", index=False)
    ue_df2.to_csv("./data/raw/UEReports_fault_dt.csv", index=False)
    
    # Step 4: Generate & save windows
    wg = WindowGenerator()
    windows = wg.generate_windows(cell_df2, ue_df2)
    stats = wg.get_window_statistics(windows)
    out_dir = PATHS['training'] / 'windows'
    wg.save_all_windows(windows, out_dir)

    log.info(f"Generated windows={stats['window_count']} "
             f"| span(hours)={stats['time_span']['total_hours']:.1f} "
             f"| mean_completeness={stats['completeness']['mean']:.3f}")
    log.info(f"Saved windows to: {out_dir}")
    log.info("DONE.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
