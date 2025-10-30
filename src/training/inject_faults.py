
import argparse
import random
from pathlib import Path
from typing import Tuple,  Optional, Union, List, Dict
import pandas as pd
import numpy as np
import random
import math



MICRO_JITTER_PROB = 0.65       # per chosen timestamp
MACRO_JITTER_PROB = 0.35
DROP_TS_PROB      = 0.20       # set ts -> NaT for all rows at that timestamp
LOW_THP_PROB      = 0.30       # set tiny positive throughput (min variability)
PRB_ZERO_POS_THP_PROB = 0.25   # cell: PRBUsed=0 & THP>0  (violation)
PRB_GT_AVAIL_PROB = 0.25       # cell: PRBUsed > Avail   (violation)
CONN_MEAN_GT_MAX_PROB = 0.20   # cell: ConnMean > ConnMax
ENERGY_SPIKE_PROB = 0.20       # cell: PEE.Energy / AvgPower mismatch
RECON_MISMATCH_PROB = 0.20     # UE vs Cell mild mismatch



def _inject_cell_faults(df: pd.DataFrame, fault_fraction: float) -> pd.DataFrame:
    """Return a DataFrame containing a subset of rows with injected faults.

    Parameters
    ----------
    df : pd.DataFrame
        Original cell reports data.
    fault_fraction : float
        Fraction of rows to duplicate and corrupt (between 0 and 1).

    Returns
    -------
    pd.DataFrame
        New DataFrame containing only the faulty rows.
    """
    n_rows = len(df)
    n_faults = max(1, int(n_rows * fault_fraction))
    # Sample without replacement to avoid duplicate indices
    faulty_rows = df.sample(n=n_faults, random_state=42).copy(deep=True)

    # Inject various faults
    for idx, row in faulty_rows.iterrows():
        # Randomly pick a fault type
        fault_type = random.choice(["missing", "out_of_range", "inconsistent", "timeliness"])
        if fault_type == "missing":
            # Drop a mandatory field such as PRB availability
            if "RRU.PrbAvailDl" in row:
                row["RRU.PrbAvailDl"] = pd.NA
            if "RRU.PrbAvailUl" in row:
                row["RRU.PrbAvailUl"] = pd.NA
        elif fault_type == "out_of_range":
            # Inflate PRB total percentage beyond 100 and throughput beyond typical limits
            if "RRU.PrbTotDl" in row:
                row["RRU.PrbTotDl"] = 120.0  # percentage >100% invalid
            if "RRU.PrbTotUl" in row:
                row["RRU.PrbTotUl"] = 120.0
            # Throughput unrealistic high
            for col in ["DRB.UEThpDl", "DRB.UEThpUl"]:
                if col in row and pd.notna(row[col]):
                    row[col] = row[col] * 10.0
        elif fault_type == "inconsistent":
            # Set PRB used greater than available
            if "RRU.PrbUsedDl" in row and "RRU.PrbAvailDl" in row:
                row["RRU.PrbUsedDl"] = row["RRU.PrbAvailDl"] + 10
            if "RRU.PrbUsedUl" in row and "RRU.PrbAvailUl" in row:
                row["RRU.PrbUsedUl"] = row["RRU.PrbAvailUl"] + 10
            # Zero throughput but non‑zero connections
            if "RRC.ConnMean" in row and "DRB.UEThpDl" in row:
                row["RRC.ConnMean"] = 5
                row["DRB.UEThpDl"] = 0.0
        elif fault_type == "timeliness":
            timestamp_cols = [c for c in row.index if isinstance(c, str) and ("time" in c.lower() or "timestamp" in c.lower())]
            for col in timestamp_cols:
                orig_val = pd.to_datetime(row[col], utc=True, errors="coerce")
                if pd.notna(orig_val):
                    # random large jitter ±61, ±90, ±120 s
                    jitter = random.choice([-240, -180, -120, 120, 180, 240])
                    row[col] = orig_val + pd.Timedelta(seconds=jitter)
                else:
                    # for string timestamps, append suffix
                    row[col] = row[col]
        # Assign the modified row back
        faulty_rows.loc[idx] = row
    return faulty_rows


def _inject_ue_faults(df: pd.DataFrame, fault_fraction: float) -> pd.DataFrame:
    """Return a DataFrame containing a subset of UE rows with injected faults.

    Parameters
    ----------
    df : pd.DataFrame
        Original UE reports data.
    fault_fraction : float
        Fraction of rows to duplicate and corrupt (between 0 and 1).

    Returns
    -------
    pd.DataFrame
        New DataFrame containing only the faulty rows.
    """
    n_rows = len(df)
    n_faults = max(1, int(n_rows * fault_fraction))
    faulty_rows = df.sample(n=n_faults, random_state=42).copy(deep=True)
    for idx, row in faulty_rows.iterrows():
        fault_type = random.choice(["missing", "out_of_range", "inconsistent", "timeliness"])
        if fault_type == "missing":
            # Missing CQI values
            if "DRB.UECqiDl" in row:
                row["DRB.UECqiDl"] = pd.NA
            if "DRB.UECqiUl" in row:
                row["DRB.UECqiUl"] = pd.NA
        elif fault_type == "out_of_range":
            # CQI out of valid range
            if "DRB.UECqiDl" in row:
                row["DRB.UECqiDl"] = 20  # >15 invalid
            if "DRB.UECqiUl" in row:
                row["DRB.UECqiUl"] = 20
            # Inflate PRB used beyond typical counts
            for col in ["RRU.PrbUsedDl", "RRU.PrbUsedUl"]:
                if col in row and pd.notna(row[col]):
                    row[col] = row[col] * 5.0
        elif fault_type == "inconsistent":
            # Set throughput but zero PRB used (inefficient scenario)
            for col in ["RRU.PrbUsedDl", "RRU.PrbUsedUl"]:
                if col in row:
                    row[col] = 0
            for col in ["DRB.UEThpDl", "DRB.UEThpUl"]:
                if col in row:
                    row[col] = 100.0  # some positive throughput
        elif fault_type == "timeliness":
            timestamp_cols = [c for c in row.index if isinstance(c, str) and ("time" in c.lower() or "timestamp" in c.lower())]
            for col in timestamp_cols:
                orig_val = pd.to_datetime(row[col], utc=True, errors="coerce")
                if pd.notna(orig_val):
                    # random large jitter ±61, ±90, ±120 s
                    jitter = random.choice([-240, -180, -120, 120, 180, 240])
                    row[col] = orig_val + pd.Timedelta(seconds=jitter)
                else:
                    # for string timestamps, append suffix
                    row[col] = row[col]
           
        faulty_rows.loc[idx] = row
    return faulty_rows


def inject_faults(
    cell_input: Path,
    ue_input: Path,
    cell_output: Path,
    ue_output: Path,
    fault_fraction: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Inject faults into cell and UE datasets and write output files.

    Parameters
    ----------
    cell_input : Path
        Path to the input cell report CSV.
    ue_input : Path
        Path to the input UE report CSV.
    cell_output : Path
        Path to write the faulty cell report CSV.
    ue_output : Path
        Path to write the faulty UE report CSV.
    fault_fraction : float, optional
        Fraction of rows to duplicate and corrupt (default 0.1).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The augmented cell and UE DataFrames.
    """
    # Read the datasets
    cell_df = pd.read_csv(cell_input)
    ue_df = pd.read_csv(ue_input)

    faulty_cell = _inject_cell_faults(cell_df, fault_fraction)
    faulty_ue = _inject_ue_faults(ue_df, fault_fraction)

    # Append faulty rows to original datasets
    augmented_cell = pd.concat([cell_df, faulty_cell], ignore_index=True)
    augmented_ue = pd.concat([ue_df, faulty_ue], ignore_index=True)

    # Write out the new datasets
    augmented_cell.to_csv(cell_output, index=False)
    augmented_ue.to_csv(ue_output, index=False)

    return augmented_cell, augmented_ue




FAULT_TYPES = ["missing", "out_of_range", "inconsistent", "timeliness"]

def _find_ts_col(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for c in ["timestamp","time","datetime","event_time","ts"]:
        if c in df.columns: return c
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]): return c
    raise ValueError("No timestamp column found; pass ts_col.")

def _ensure_dt(df: pd.DataFrame, ts_col: str):
    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    if df[ts_col].isna().all():
        raise ValueError(f"Could not parse datetimes in '{ts_col}'.")

def _mutate_row_values_cell(row: pd.Series, fault_type: str) -> pd.Series:
    if fault_type == "missing":
        for c in ["RRU.PrbAvailDl","RRU.PrbAvailUl","DRB.UEThpDl","DRB.UEThpUl"]:
            if c in row: row[c] = pd.NA
    elif fault_type == "out_of_range":
        for c in ["RRU.PrbTotDl","RRU.PrbTotUl"]: 
            if c in row and pd.notna(row[c]): row[c] = float(row[c]) * 3.0 + 50.0
        for c in ["DRB.UEThpDl","DRB.UEThpUl"]:
            if c in row and pd.notna(row[c]): row[c] = float(row[c]) * 8.0
    elif fault_type == "inconsistent":
        if "RRU.PrbUsedDl" in row and "RRU.PrbAvailDl" in row: row["RRU.PrbUsedDl"] = row["RRU.PrbAvailDl"] + 10
        if "RRU.PrbUsedUl" in row and "RRU.PrbAvailUl" in row: row["RRU.PrbUsedUl"] = row["RRU.PrbAvailUl"] + 10
        if "RRC.ConnMean" in row: row["RRC.ConnMean"] = 3
    return row

def _mutate_row_values_ue(row: pd.Series, fault_type: str) -> pd.Series:
    if fault_type == "missing":
        for c in ["DRB.UECqiDl","DRB.UECqiUl","DRB.UEThpDl","DRB.UEThpUl"]:
            if c in row: row[c] = pd.NA
    elif fault_type == "out_of_range":
        for c in ["DRB.UECqiDl","DRB.UECqiUl"]:
            if c in row: row[c] = 25
        for c in ["RRU.PrbUsedDl","RRU.PrbUsedUl"]:
            if c in row and pd.notna(row[c]): row[c] = float(row[c]) * 5.0
    elif fault_type == "inconsistent":
        for c in ["RRU.PrbUsedDl","RRU.PrbUsedUl"]:
            if c in row: row[c] = 0
        for c in ["DRB.UEThpDl","DRB.UEThpUl"]:
            if c in row: row[c] = 120.0
    return row



def _cell_updates(row, fault_type):
    u = {}
    if fault_type == "missing":
        for c in ["RRU.PrbAvailDl","RRU.PrbAvailUl","DRB.UEThpDl","DRB.UEThpUl",
                  "RRU.PrbUsedDl","RRU.PrbUsedUl","RRC.ConnMean", "Viavi.Cell.Name"]:
            if c in row: u[c] = np.nan
    elif fault_type == "out_of_range":
        for c in ["RRU.PrbTotDl","RRU.PrbTotUl"]:
            if c in row and pd.notna(row[c]): u[c] = float(row[c]) * 1.5 + 20.0
        for c in ["DRB.UEThpDl","DRB.UEThpUl"]:
            if c in row and pd.notna(row[c]): u[c] = float(row[c]) * 3.0
    elif fault_type == "inconsistent":
        if "RRU.PrbUsedDl" in row and "RRU.PrbAvailDl" in row:
            u["RRU.PrbUsedDl"] = row["RRU.PrbAvailDl"] + 5
        if "RRU.PrbUsedUl" in row and "RRU.PrbAvailUl" in row:
            u["RRU.PrbUsedUl"] = row["RRU.PrbAvailUl"] + 5
        if "PEE.AvgPower" in row:
            u["PEE.AvgPower"] = float(row["PEE.AvgPower"]) * 0.6
        if "RRC.ConnMean" in row:
            u["RRC.ConnMean"] = 3
    elif fault_type == "timeliness":
        ts_col = "timestamp"  
        ts = row.get(ts_col, None)
        if ts is not None and pd.notna(ts):
            ts = pd.to_datetime(ts, utc=True, errors="coerce")
            if pd.notna(ts):
                jitter = random.choice([-240, -180, -120, 120, 180, 240]) # secs
                u[ts_col] = pd.to_datetime(row[ts_col]) + pd.Timedelta(seconds=jitter)    
    # --- NEW: violations & energy ---
    if random.random() < PRB_ZERO_POS_THP_PROB:
        for side in ("Dl","Ul"):
            thp = f"DRB.UEThp{side}"
            prb = f"RRU.PrbUsed{side}"
            if thp in row and prb in row:
                u[prb] = 0
                # tiny positive THP to trigger 'thp_pos_prb_zero'
                u[thp] = max(0.01, float(row.get(thp, 0)) or 0.01)

    if random.random() < PRB_GT_AVAIL_PROB:
        for side in ("Dl","Ul"):
            used = f"RRU.PrbUsed{side}"
            avail = f"RRU.PrbAvail{side}"
            if used in row and avail in row and pd.notna(row[avail]):
                u[used] = float(row[avail]) + random.randint(1, 6)

    if random.random() < CONN_MEAN_GT_MAX_PROB:
        if "RRC.ConnMean" in row and "RRC.ConnMax" in row:
            cm = float(pd.to_numeric(row["RRC.ConnMean"], errors="coerce") or 3)
            u["RRC.ConnMax"] = max(0, cm - random.randint(1, 3))  # force mean>max

    if random.random() < ENERGY_SPIKE_PROB:
        # mismatch between ΔEnergy and Power*Δt
        if "PEE.Energy" in row:
            u["PEE.Energy"] = float(pd.to_numeric(row["PEE.Energy"], errors="coerce") or 0) * random.uniform(1.3, 2.0)
        elif "PEE.AvgPower" in row:
            u["PEE.AvgPower"] = float(pd.to_numeric(row["PEE.AvgPower"], errors="coerce") or 0) * random.uniform(1.5, 2.5)

    if random.random() < LOW_THP_PROB:
        for side in ("Dl","Ul"):
            thp = f"DRB.UEThp{side}"
            if thp in row:
                u[thp] = random.choice([0.0, 0.001, 0.005])  # affect *_min

    return u

def _ue_updates(row, fault_type):
    u = {}
    if fault_type == "missing":
        for c in ["DRB.UECqiDl","DRB.UECqiUl","DRB.UEThpDl","DRB.UEThpUl","Viavi.UE.Name"]:
            if c in row: u[c] = np.nan
    elif fault_type == "out_of_range":
        for c in ["DRB.UECqiDl","DRB.UECqiUl"]:
            if c in row: u[c] = 25
        for c in ["RRU.PrbUsedDl","RRU.PrbUsedUl"]:
            if c in row and pd.notna(row[c]): u[c] = float(row[c]) * 3.0
    elif fault_type == "inconsistent":
        for c in ["RRU.PrbUsedDl","RRU.PrbUsedUl"]:
            if c in row: u[c] = 0
        for c in ["DRB.UEThpDl","DRB.UEThpUl"]:
            if c in row: u[c] = 50.0
    elif fault_type == "timeliness":
        ts_col = "timestamp"  
        ts = row.get(ts_col, None)
        if ts is not None and pd.notna(ts):
            ts = pd.to_datetime(ts, utc=True, errors="coerce")
            if pd.notna(ts):
                jitter = random.choice([-240, -180, -120, 120, 180, 240])  # secs
                u[ts_col] = pd.to_datetime(row[ts_col]) + pd.Timedelta(seconds=jitter)  
    if random.random() < RECON_MISMATCH_PROB:
        # small random scale on UE PRB/THP to create ue/cell ratio drift
        for side in ("Dl","Ul"):
            for col in (f"RRU.PrbUsed{side}", f"DRB.UEThp{side}"):
                if col in row and pd.notna(row[col]):
                    u[col] = float(row[col]) * random.uniform(0.6, 1.4)                         
    return u

def inject_faults_inplace_by_windows(
    df: pd.DataFrame,
    is_cell: bool,
    ts_col: Optional[str] = None,
    window_seconds: int = 300,
    overlap: float = 0.8,
    faulty_window_fraction: float = 0.15,
    timestamps_per_faulty_window: int = 2,
    rows_fraction_per_timestamp: float = 0.3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sliding-window aware fault injection (in-place):
    - Builds overlapping windows of length `window_seconds` and stride = window_seconds*(1-overlap)
    - Randomly selects ~faulty_window_fraction of windows to be 'faulty'
    - In each selected window, pick up to `timestamps_per_faulty_window` timestamps and corrupt a fraction of rows
    - Keeps row count per timestamp unchanged; returns (df_with_flags, manifest)
    """

    if not (0 <= overlap < 1):
        raise ValueError("overlap must be in [0, 1).")
    stride = max(1, int(round(window_seconds * (1 - overlap))))
    rnd = np.random.RandomState(random_state)
    random.seed(random_state)

    ts = _find_ts_col(df, ts_col)
    _ensure_dt(df, ts)
    out = df.copy(deep=True)

    # add flags
    if "dq_fault_flag" not in out.columns: out["dq_fault_flag"] = False
    if "dq_fault_type" not in out.columns: out["dq_fault_type"] = pd.NA
    if "dq_fault_window" not in out.columns: out["dq_fault_window"] = pd.NA

    # Build sliding windows
    tmin = out[ts].min()
    tmax = out[ts].max()
    if pd.isna(tmin) or pd.isna(tmax):
        raise ValueError("Dataset has no valid timestamps.")

    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    start = tmin
    while start <= tmax:
        end = start + pd.Timedelta(seconds=window_seconds)
        windows.append((start, end))
        start = start + pd.Timedelta(seconds=stride)

    nW = len(windows)
    n_faulty = max(1, int(round(nW * faulty_window_fraction)))
    faulty_idxs = rnd.choice(np.arange(nW), size=n_faulty, replace=False)

    manifest_rows: List[Dict] = []

    # Precompute timestamps -> indices map for speed
    out_sorted = out.sort_values(ts)
    # Iterate windows
    for w_idx in sorted(faulty_idxs):
        w_start, w_end = windows[w_idx]
        mask = (out_sorted[ts] >= w_start) & (out_sorted[ts] < w_end)
        g = out_sorted.loc[mask]
        if g.empty:
            continue

        uniq_ts = g[ts].dropna().unique()
        pick_n = min(timestamps_per_faulty_window, len(uniq_ts))
        chosen_ts = rnd.choice(uniq_ts, size=pick_n, replace=False)

        for tstamp in chosen_ts:
            apply_drop = rnd.random() < DROP_TS_PROB
            if apply_drop:
                out.loc[out_sorted.index[(out_sorted[ts] == tstamp)], ts] = pd.NaT
            else:
                if rnd.random() < MICRO_JITTER_PROB:
                    jitter_sec = rnd.randint(7, 16) * rnd.choice([-1, 1])
                elif rnd.random() < MACRO_JITTER_PROB:
                    jitter_sec = rnd.choice([-240, -180, -120, 120, 180, 240])
                else:
                    jitter_sec = 0
                if jitter_sec != 0:
                    idxs = g[g[ts] == tstamp].index
                    out.loc[idxs, ts] = out.loc[idxs, ts] + pd.Timedelta(seconds=int(jitter_sec)) 
            sel = g[g[ts] == tstamp].index
            if len(sel) == 0:
                continue
            n_rows = max(1, int(round(len(sel) * rows_fraction_per_timestamp)))
            chosen_rows = rnd.choice(sel, size=n_rows, replace=False)

            ftypes = []
            for rid in chosen_rows:
                # Guarantee timeliness faults in every 3rd faulty window
                if w_idx % 2 == 0 or rnd.random() < 0.5:
                    ftype = "timeliness"
                else:
                    ftype = rnd.choice(FAULT_TYPES)
                
                if ftype == "timeliness":
                    # ts_col  timestamp 
                    #orig_ts = out.at[rid, ts]
                    #orig_dt = pd.to_datetime(orig_ts, errors="coerce")
                    #if pd.notna(orig_dt):
                        # cadence offset ±61/90/120s 
                        #jitter = random.choice([-240, -180, -120, 120, 180, 240])
                        #out.at[rid, ts] = orig_dt + pd.Timedelta(seconds=jitter)
                    #else:
                        # string timestamp 
                        #out.at[rid, ts] = f"{orig_ts}_jitter"
                    if pd.notna(out.at[rid, ts]):
                        jitter = int(rnd.choice([-240, -180, -120, 120, 180, 240]))
                        out.at[rid, ts] = out.at[rid, ts] + pd.Timedelta(seconds=jitter)
                else:
                    if is_cell:
                        updates = _cell_updates(out.loc[rid], ftype)
                    else:
                        updates = _ue_updates(out.loc[rid], ftype)
                    for col, val in updates.items():
                        out.at[rid, col] = val
                ftypes.append(ftype)

            manifest_rows.append({
                "window_index": int(w_idx),
                "window_start": pd.to_datetime(w_start),
                "window_end":   pd.to_datetime(w_end),
                "timestamp":    pd.to_datetime(tstamp),
                "count_faulted": int(len(chosen_rows)),
                "indices":      list(map(int, chosen_rows)),
                "fault_types":  ftypes,
            })

    manifest = pd.DataFrame(manifest_rows).sort_values(["window_index","timestamp"])
    out.drop(columns=["dq_fault_flag","dq_fault_type","dq_fault_window"],
        errors="ignore", inplace=True)
    return out, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject synthetic faults into O-RAN datasets.")
    parser.add_argument("--cell-input", type=Path, required=True, help="Path to input cell report CSV")
    parser.add_argument("--ue-input", type=Path, required=True, help="Path to input UE report CSV")
    parser.add_argument("--cell-output", type=Path, required=True, help="Path to output cell CSV with faults")
    parser.add_argument("--ue-output", type=Path, required=True, help="Path to output UE CSV with faults")
    parser.add_argument(
        "--fault-fraction",
        type=float,
        default=0.1,
        help="Fraction of rows to duplicate and corrupt (0 < f ≤ 1; default=0.1)",
    )
    args = parser.parse_args()
    inject_faults(args.cell_input, args.ue_input, args.cell_output, args.ue_output, args.fault_fraction)


if __name__ == "__main__":
    main()