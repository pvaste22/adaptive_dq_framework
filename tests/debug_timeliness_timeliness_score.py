import argparse
import os
import sys
import json
from pathlib import Path
import random
import pandas as pd
import numpy as np

def read_window(window_dir: Path):
    """Read one window folder with cell_data.parquet and ue_data.parquet"""
    cell_pq = window_dir / "cell_data.parquet"
    ue_pq   = window_dir / "ue_data.parquet"
    meta_js = window_dir / "metadata.json"
    if not cell_pq.exists() or not ue_pq.exists():
        raise FileNotFoundError(f"Missing parquet files in: {window_dir}")
    df_cell = pd.read_parquet(cell_pq)
    df_ue   = pd.read_parquet(ue_pq)
    # enforce datetime tz-aware
    for df in (df_cell, df_ue):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    meta = {}
    if meta_js.exists():
        try:
            meta = json.loads(meta_js.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    return df_cell, df_ue, meta

def _expected_grid(min_ts: pd.Timestamp, max_ts: pd.Timestamp, cadence_sec: int):
    # align start to exact cadence boundary based on epoch
    start = pd.Timestamp(int((min_ts.value // 10**9) // cadence_sec * cadence_sec), unit="s", tz="UTC")
    # ensure we cover max_ts
    end = pd.Timestamp(int((max_ts.value // 10**9 + cadence_sec - 1) // cadence_sec * cadence_sec), unit="s", tz="UTC")
    num = int((end - start).total_seconds() // cadence_sec) + 1
    return [start + pd.to_timedelta(i * cadence_sec, unit="s") for i in range(num)]

def t2_grid_alignment_score(df_cell: pd.DataFrame, df_ue: pd.DataFrame, cadence_sec: int, tolerance_sec: int):
    """Compute a simple grid-alignment score (T2-like): fraction of expected cadence slots covered by any timestamp within tolerance."""
    # union of timestamps across cell+ue
    ts_cell = df_cell["timestamp"].dropna().unique() if "timestamp" in df_cell.columns else np.array([], dtype="datetime64[ns]")
    ts_ue   = df_ue["timestamp"].dropna().unique() if "timestamp" in df_ue.columns else np.array([], dtype="datetime64[ns]")
    if len(ts_cell) == 0 and len(ts_ue) == 0:
        return 1.0, 0, 0  # empty window treated as perfect (or skip?)
    ts_all = pd.to_datetime(pd.Series(np.concatenate([ts_cell, ts_ue])), utc=True)
    min_ts = ts_all.min()
    max_ts = ts_all.max()
    expected = _expected_grid(min_ts, max_ts, cadence_sec)
    tol = pd.to_timedelta(tolerance_sec, unit="s")
    # mark a slot "covered" if any actual timestamp within tolerance
    covered = 0
    for slot in expected:
        # compute min abs diff to any ts
        diffs = (ts_all - slot).abs()
        if (diffs <= tol).any():
            covered += 1
    score = covered / max(len(expected), 1)
    return score, covered, len(expected)

def inject_timeliness_fault_full_slot(df_cell: pd.DataFrame, df_ue: pd.DataFrame, cadence_sec: int, jitter_sec: int, random_state: int = 7):
    """Pick one cadence slot present in the window; shift ALL rows at that exact timestamp (cell+UE) by jitter_sec.
       This should remove coverage for that slot and drop T2 score.
    """
    rng = random.Random(random_state)
    # find candidate slot from actual timestamps (use the mode minute)
    if "timestamp" not in df_cell.columns and "timestamp" not in df_ue.columns:
        return df_cell, df_ue, None, 0
    ts_all = pd.Series([], dtype="datetime64[ns, UTC]")
    if "timestamp" in df_cell.columns:
        ts_all = pd.concat([ts_all, pd.to_datetime(df_cell["timestamp"], utc=True, errors="coerce")])
    if "timestamp" in df_ue.columns:
        ts_all = pd.concat([ts_all, pd.to_datetime(df_ue["timestamp"], utc=True, errors="coerce")])
    ts_all = ts_all.dropna()
    if ts_all.empty:
        return df_cell, df_ue, None, 0
    # choose a slot that has the most rows at exact same second (to maximize impact)
    counts = ts_all.value_counts()
    chosen_ts = counts.sort_values(ascending=False).index[0]
    # apply shift
    shift_td = pd.to_timedelta(jitter_sec, unit="s")
    n_shifted = 0
    if "timestamp" in df_cell.columns:
        mask_c = df_cell["timestamp"] == chosen_ts
        n_shifted += int(mask_c.sum())
        df_cell.loc[mask_c, "timestamp"] = df_cell.loc[mask_c, "timestamp"] + shift_td
    if "timestamp" in df_ue.columns:
        mask_u = df_ue["timestamp"] == chosen_ts
        n_shifted += int(mask_u.sum())
        df_ue.loc[mask_u, "timestamp"] = df_ue.loc[mask_u, "timestamp"] + shift_td
    return df_cell, df_ue, chosen_ts, n_shifted

def main():
    ap = argparse.ArgumentParser(description="Debug Timeliness: read windows, inject one timeliness fault, recompute T2 score.")
    ap.add_argument("--windows_dir", required=True, help="Path to windows root (each subfolder is a window with parquet files).")
    ap.add_argument("--cadence_sec", type=int, default=60)
    ap.add_argument("--tolerance_sec", type=int, default=5, help="Timeliness tolerance (ts_resolution_sec).")
    ap.add_argument("--jitter_sec", type=int, default=90, help="Seconds to shift for the injected slot (e.g., 61/90/120).")
    ap.add_argument("--max_windows", type=int, default=20, help="Limit number of windows to test.")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    windows_root = Path(args.windows_dir)
    if not windows_root.exists():
        print(f"ERROR: windows_dir not found: {windows_root}", file=sys.stderr)
        sys.exit(1)

    # collect window folders (dirs containing parquet files)
    win_dirs = [p for p in sorted(windows_root.iterdir()) if p.is_dir()]
    if not win_dirs:
        print("No window folders found.")
        sys.exit(0)

    random.seed(args.seed)
    rows = []
    tested = 0
    for wdir in win_dirs:
        if tested >= args.max_windows:
            break
        try:
            df_cell, df_ue, meta = read_window(wdir)
        except Exception as e:
            print(f"Skip {wdir.name}: {e}")
            continue
        # compute baseline T2
        score_before, covered_before, expected_before = t2_grid_alignment_score(df_cell, df_ue, args.cadence_sec, args.tolerance_sec)
        # inject one full-slot shift
        df_cell2 = df_cell.copy()
        df_ue2   = df_ue.copy()
        df_cell2, df_ue2, chosen_ts, n_shifted = inject_timeliness_fault_full_slot(df_cell2, df_ue2, args.cadence_sec, args.jitter_sec, random_state=random.randint(0, 10**6))
        # recompute T2
        score_after, covered_after, expected_after = t2_grid_alignment_score(df_cell2, df_ue2, args.cadence_sec, args.tolerance_sec)

        rows.append({
            "window": wdir.name,
            "chosen_ts": str(chosen_ts) if chosen_ts is not None else None,
            "n_shifted_rows": n_shifted,
            "t2_score_before": round(score_before, 6),
            "t2_score_after": round(score_after, 6),
            "covered_before": covered_before,
            "covered_after": covered_after,
            "expected_slots": expected_before,
        })
        tested += 1

    out_df = pd.DataFrame(rows)
    out_path = Path("timeliness_debug_summary.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Summary written to {out_path.resolve()}")
    # show quick stats
    if not out_df.empty:
        drops = (out_df["t2_score_after"] < out_df["t2_score_before"]).sum()
        print(f"Windows tested: {len(out_df)} | Score drops: {drops} | No change: {(len(out_df)-drops)}")
        print(out_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()