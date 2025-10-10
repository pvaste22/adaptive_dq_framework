# timeliness.py
"""
Phase: 2
Purpose: Timeliness dimension calculator (T1-T3)
Depends on: dq_baseline (cadence_sec, ts_resolution_sec)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from datetime import datetime

from .base_dimension import BaseDimension
from common.constants import COLUMN_NAMES, MEAS_INTERVAL_SEC

class TimelinessDimension(BaseDimension):
    """Calculates timeliness score for a window."""

    def __init__(self, baselines_path: Optional[Path] = None):
        super().__init__('timeliness')
        # Pull cadence / resolution from self-calibrated dq_baseline (Phase-1)
        dq = self.get_dq_baseline() or {}
        self.cadence_sec: float = float(dq.get('cadence_sec', MEAS_INTERVAL_SEC))
        self.ts_resolution_sec: float = float(dq.get('ts_resolution_sec', 1.0))
        # Cols
        self.ts_col = COLUMN_NAMES.get('timestamp', 'timestamp')
        self.cell_col = COLUMN_NAMES.get('cell_entity', 'Viavi.Cell.Name')
        self.ue_col = COLUMN_NAMES.get('ue_entity', 'Viavi.UE.Name')
        #self.logger.info(f"Timeliness initialized: cadence={self.cadence_sec}s, ts_resolution={self.ts_resolution_sec}s")
        self.logger.info("Timeliness initiated")

    def calculate_score(self, window_data: Dict) -> Dict:
        ok, err = self.validate_window_data(window_data)
        if not ok:
            return {'score': 0.0, 'coverage': 0.0, 'status': 'ERROR', 'details': {'validation_error': err}}

        cell = window_data.get('cell_data', pd.DataFrame())
        ue   = window_data.get('ue_data', pd.DataFrame())

        # T1: Inter-arrival adherence (row-wise)
        t1_series, t1_details = self._row_interarrival_ok(cell, ue)

        # T2: Cadence grid alignment coverage (aggregate)
        t2_tuple, t2_details = self._grid_alignment(cell, ue)

        # T3: Monotonicity / no out-of-order (row-wise)
        t3_series, t3_details = self._row_monotonic_ok(cell, ue)

        #self.logger.info(f"Timeliness initialized: T1 pass={t1_details['passed'] / t1_details['applicable']}, t3 pass={t3_details['passed'] / t3_details['applicable']}")

        apr, mpr, coverage, fails = self._apr_mpr(
            check_series_list=[t1_series, t3_series],
            check_tuples_list=[t2_tuple]
        )

        return {
            'score': mpr,
            'apr': apr,
            'coverage': coverage,
            'details': {
                'T1_interarrival': t1_details,
                'T2_grid_alignment': t2_details,
                'T3_monotonicity': t3_details,
                'fail_counts': fails,
                'cadence_sec': self.cadence_sec,
                'ts_resolution_sec': self.ts_resolution_sec
            }
        }

    # ---------- helpers ----------

    def _row_interarrival_ok(self, cell: pd.DataFrame, ue: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """
        True if per-entity Δt within cadence ± ts_resolution.
        First row per entity is NA (not counted).
        """
        details = {}
        s_all = []

        def per_df(df: pd.DataFrame, ent_col: str) -> Optional[pd.Series]:
            if df.empty or self.ts_col not in df.columns or ent_col not in df.columns:
                return None
            df = df[[ent_col, self.ts_col]].dropna().copy()
            if df.empty:
                return None
            ts = pd.to_datetime(df[self.ts_col], errors='coerce')
            df = df.assign(__ts=ts).dropna(subset=['__ts']).sort_values([ent_col, '__ts'])
            dt = df.groupby(ent_col)['__ts'].diff().dt.total_seconds()
            # tolerance purely from training-derived resolution (no hardcoded tolerance)
            tol = self.ts_resolution_sec
            ok = dt.sub(self.cadence_sec).abs() <= tol
            # First samples per entity have NA diff -> not applicable
            ok = ok.astype('float')
            ok[dt.isna()] = np.nan
            return ok.reset_index(drop=True)

        s1 = per_df(cell, self.cell_col)
        s2 = per_df(ue, self.ue_col)
        if s1 is not None: s_all.append(s1)
        if s2 is not None: s_all.append(s2)

        series = pd.concat(s_all, axis=0, ignore_index=True) if s_all else pd.Series([], dtype='float')

        total_applicable = int(series.notna().sum())
        t1total = int(series.sum())
        passed = int((series == 1.0).sum())
        details.update({'applicable': total_applicable, 'passed': passed})
        return series, details

    def _grid_alignment(self, cell: pd.DataFrame, ue: pd.DataFrame) -> Tuple[Tuple[int, int], Dict]:
        details: Dict = {}
        ts_parts: List[pd.Series] = []
        for df in (cell, ue):
            if not df.empty and self.ts_col in df.columns:
                ts_parts.append(pd.to_datetime(df[self.ts_col], errors='coerce').dropna())

        if not ts_parts:
            return (0, 0), {'note': 'no timestamps in window'}

        tsu = pd.to_datetime(pd.Series(pd.concat(ts_parts).unique())).sort_values()
        if tsu.empty:
            return (0, 0), {'note': 'no valid timestamps after parsing'}

        min_ts = tsu.iloc[0]
        max_ts = tsu.iloc[-1]
        step = pd.to_timedelta(self.cadence_sec, unit='s')

        # Anchor grid to cadence boundary (e.g., top-of-minute for 60s)
        anchor = pd.to_datetime(min_ts).floor(f'{int(self.cadence_sec)}S')

        # Expected grid from anchor to max_ts (inclusive)
        grid = pd.date_range(start=anchor, end=max_ts, freq=step)
        total_expected = int(len(grid)) if len(grid) > 0 else 0
        if total_expected == 0:
            return (0, 0), {'note': 'degenerate window'}

        diffs_sec = (tsu - anchor).dt.total_seconds()
        k = np.rint(diffs_sec / self.cadence_sec).astype(int)
        snapped = anchor + pd.to_timedelta(k * self.cadence_sec, unit='s')
        tol_sec = min(self.ts_resolution_sec, 0.1 * self.cadence_sec)
        within_tol = (tsu - snapped).abs() <= pd.to_timedelta(tol_sec, unit='s')
        grid_end   = grid[-1]
        snapped_ts = snapped
        valid_idx = snapped_ts <= grid_end
        

        # Unique covered grid slots only
        covered_k = pd.unique(k[within_tol & valid_idx])
        passed    = int(len(covered_k))
        total = total_expected

        details.update({
            'unique_ts': int(tsu.shape[0]),
            'expected_grid_points': total_expected,
            'aligned_slots': passed
        })
        return (passed, total), details


    def _row_monotonic_ok(self, cell: pd.DataFrame, ue: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """
        True if per-entity timestamps are non-decreasing (no out-of-order).
        First row per entity is NA (not counted).
        """
        details = {}
        s_all = []

        def per_df(df: pd.DataFrame, ent_col: str) -> Optional[pd.Series]:
            if df.empty or self.ts_col not in df.columns or ent_col not in df.columns:
                return None
            df = df[[ent_col, self.ts_col]].dropna().copy()
            if df.empty:
                return None
            df['__ts'] = pd.to_datetime(df[self.ts_col], errors='coerce')
            df = df.dropna(subset=['__ts'])
            # out-of-order if diff < 0; OK when diff >= 0
            d = df.groupby(ent_col, sort=False)['__ts'].diff().dt.total_seconds()
            ok = d.ge(0).astype('float')
            ok[d.isna()] = np.nan 
            return ok.reset_index(drop=True)

        s1 = per_df(cell, self.cell_col)
        s2 = per_df(ue, self.ue_col)
        if s1 is not None: s_all.append(s1)
        if s2 is not None: s_all.append(s2)

        series = pd.concat(s_all, axis=0, ignore_index=True) if s_all else pd.Series([], dtype='float')
        t3total = int(series.sum())
        total_applicable = int(series.notna().sum())
        passed = int((series == 1.0).sum())
        details.update({'applicable': total_applicable, 'passed': passed})
        return series, details
