# validity.py
"""
Phase: 2
Dimension: Validity (types/ranges/enums/non-negatives/PRB constraints)
Inputs: cell_df, ue_df, baselines (field_ranges), config (via constants)
Scoring: Row-wise boolean series -> _apr_mpr()
"""

from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd

from .base_dimension import BaseDimension
from common.constants import COLUMN_NAMES

class ValidityDimension(BaseDimension):
    def __init__(self):
        super().__init__('validity')
        # Artifacts/Config
        self.field_ranges = self.load_artifact_baseline('field_ranges') or {}
        # Column names (config-first)
        self.ts_col   = COLUMN_NAMES.get('timestamp', 'timestamp')
        self.cell_col = COLUMN_NAMES.get('cell_entity', 'Viavi.Cell.Name')
        self.ue_col   = COLUMN_NAMES.get('ue_entity', 'Viavi.UE.Name')

        # Known numeric fields by table
        self.numeric_cell = [
            'DRB.UEThpDl','DRB.UEThpUl',
            'RRU.PrbUsedDl','RRU.PrbUsedUl',
            'RRU.PrbAvailDl','RRU.PrbAvailUl',
            'RRU.PrbTotDl','RRU.PrbTotUl',
            'RRC.ConnMean','RRC.ConnMax',
            'PEE.AvgPower','PEE.Energy','PEE.Energy_interval'
        ]
        self.numeric_ue = [
            'DRB.UEThpDl','DRB.UEThpUl',
            'RRU.PrbUsedDl','RRU.PrbUsedUl',
            'DRB.UECqiDl','DRB.UECqiUl'
        ]

        # Enums (if present) — keep optional
        self.enum_fields = {
            # 'DuplexMode': {'FDD','TDD'},
            # 'Tech': {'LTE','NR'},
        }

        # Static sanity ranges (only when field_ranges missing)
        self.fallback_ranges = {
            # percentages
            'RRU.PrbTotDl': (0.0, 100.0),
            'RRU.PrbTotUl': (0.0, 100.0),
            # CQI
            'DRB.UECqiDl': (0, 15),
            'DRB.UECqiUl': (0, 15),
            # BLER example (if present later)
            'Mac.BlerDl': (0.0, 1.0),
            'Mac.BlerUl': (0.0, 1.0),
        }

        # Non-negative list
        self.non_negative = [
            'RRU.PrbUsedDl','RRU.PrbUsedUl',
            'RRU.PrbAvailDl','RRU.PrbAvailUl',
            'RRC.ConnMean','RRC.ConnMax',
            'PEE.Energy','PEE.Energy_interval',
            # volumes/counters if present:
            'QosFlow.TotPdcpPduVolumeDl','QosFlow.TotPdcpPduVolumeUl',
        ]

    # ---------- Public API ----------
    def calculate_score(self, window_data: Dict) -> Dict:
        ok, err = self.validate_window_data(window_data)
        if not ok:
            return {'score': 0.0, 'coverage': 0.0, 'status': 'ERROR', 'details': {'validation_error': err}}

        cell = window_data.get('cell_data', pd.DataFrame())
        ue   = window_data.get('ue_data', pd.DataFrame())

        # V1 Types/parsability (row-wise)
        v1_series, v1_details = self._v1_types_parsability(cell, ue)

        # V2 Ranges (row-wise, config/field_ranges-driven)
        v2_series, v2_details = self._v2_ranges(cell, ue)

        # V3 Enums (row-wise; only if present)
        v3_series, v3_details = self._v3_enums(cell, ue)

        # V4 Non-negative & basic PRB constraints (row-wise)
        v4_series, v4_details = self._v4_nonneg_and_prb(cell, ue)

        apr, mpr, coverage, fails = self._apr_mpr(
            check_series_list=[v1_series, v2_series, v3_series, v4_series],
            check_tuples_list=[]  # no aggregate checks in validity for now
        )

        return {
            'score': mpr,
            'apr': apr,
            'coverage': coverage,
            'details': {
                'V1_types_parsability': v1_details,
                'V2_ranges': v2_details,
                'V3_enums': v3_details,
                'V4_nonneg_prb': v4_details,
                'fail_counts': fails
            }
        }

    # ---------- Helpers ----------

    def _v1_types_parsability(self, cell: pd.DataFrame, ue: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """
        Rule: numeric fields should be numeric (strings like 'abc' fail),
        but missing values are NA (not counted here; Completeness handles nulls).
        """
        checks: List[pd.Series] = []
        det: Dict = {}

        def series_for(df: pd.DataFrame, cols: List[str], tag: str):
            if df.empty: return
            for col in cols:
                if col in df.columns:
                    s_num = pd.to_numeric(df[col], errors='coerce')  # invalid → NaN
                    # Pass if value is numeric OR is NaN (NA not counted)
                    ok = s_num.notna()
                    ok = ok.astype('float')
                    ok[~df[col].notna()] = np.nan  # pure missing → NA
                    checks.append(ok)
                    det[f'parsable_{tag}_{col}'] = {
                        'applicable': int(ok.notna().sum()),
                        'passed':     int((ok == 1.0).sum())
                    }

        series_for(cell, self.numeric_cell, 'cell')
        series_for(ue,   self.numeric_ue,   'ue')

        series = pd.concat(checks, axis=0, ignore_index=True).astype('float') if checks else pd.Series([], dtype='float')
        return series, {'applicable': int(series.notna().sum()), 'passed': int((series == 1.0).sum())}

    def _v2_ranges(self, cell: pd.DataFrame, ue: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """
        Rule: each numeric field lies within an allowed range.
        Priority: field_ranges (train) -> fallback_ranges -> skip.
        """
        checks: List[pd.Series] = []
        det: Dict = {}

        def bounds_for(field: str):
            # Try training ranges
            for sec in ('cell_metrics','ue_metrics'):
                secmap = self.field_ranges.get(sec, {})
                if field in secmap:
                    lo = secmap[field].get('min')
                    hi = secmap[field].get('max')
                    if lo is not None and hi is not None:
                        return float(lo), float(hi)
            # Fallback static
            if field in self.fallback_ranges:
                return self.fallback_ranges[field]
            return None

        def apply_ranges(df: pd.DataFrame, cols: List[str], tag: str):
            if df.empty: return
            for col in cols:
                if col in df.columns:
                    b = bounds_for(col)
                    if not b:  # no range known → skip
                        continue
                    lo, hi = b
                    s = pd.to_numeric(df[col], errors='coerce')
                    ok = s.ge(lo) & s.le(hi)
                    ok = ok.astype('float')
                    ok[s.isna()] = np.nan
                    checks.append(ok)
                    det[f'range_{tag}_{col}'] = {
                        'lo': lo, 'hi': hi,
                        'applicable': int(ok.notna().sum()),
                        'passed':     int((ok == 1.0).sum())
                    }

        apply_ranges(cell, self.numeric_cell, 'cell')
        apply_ranges(ue,   self.numeric_ue,   'ue')

        series = pd.concat(checks, axis=0, ignore_index=True).astype('float') if checks else pd.Series([], dtype='float')
        return series, {'applicable': int(series.notna().sum()), 'passed': int((series == 1.0).sum())}

    def _v3_enums(self, cell: pd.DataFrame, ue: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """
        Rule: enum fields (if present) must be in allowed set.
        """
        checks: List[pd.Series] = []
        det: Dict = {}

        def apply_enum(df: pd.DataFrame, tag: str):
            if df.empty: return
            for col, allowed in self.enum_fields.items():
                if col in df.columns and allowed:
                    s = df[col].astype('string')
                    ok = s.isin(list(allowed))
                    ok = ok.astype('float')
                    ok[s.isna()] = np.nan
                    checks.append(ok)
                    det[f'enum_{tag}_{col}'] = {
                        'allowed': list(allowed),
                        'applicable': int(ok.notna().sum()),
                        'passed':     int((ok == 1.0).sum())
                    }

        apply_enum(cell, 'cell')
        apply_enum(ue,   'ue')

        series = pd.concat(checks, axis=0, ignore_index=True).astype('float') if checks else pd.Series([], dtype='float')
        return series, {'applicable': int(series.notna().sum()), 'passed': int((series == 1.0).sum())}

    def _v4_nonneg_and_prb(self, cell: pd.DataFrame, ue: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """
        Rules:
          - Non-negative counters/metrics.
          - PRB logical constraints:
              * Used <= Avail (per direction) when both exist.
              * PRB% ∈ [0,100] (if percentage columns present).
        """
        checks: List[pd.Series] = []
        det: Dict = {}

        def nonneg(df: pd.DataFrame, cols: List[str], tag: str):
            if df.empty: return
            for col in cols:
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors='coerce')
                    ok = s.ge(0)
                    ok = ok.astype('float')
                    ok[s.isna()] = np.nan
                    checks.append(ok)
                    det[f'nonneg_{tag}_{col}'] = {
                        'applicable': int(ok.notna().sum()),
                        'passed':     int((ok == 1.0).sum())
                    }

        nonneg(cell, self.non_negative, 'cell')
        nonneg(ue,   [c for c in self.numeric_ue if c not in ('DRB.UECqiDl','DRB.UECqiUl')], 'ue')

        # PRB Used <= Avail (cells)
        def prb_used_le_avail(df: pd.DataFrame, tag: str):
            pairs = [('RRU.PrbUsedDl','RRU.PrbAvailDl'), ('RRU.PrbUsedUl','RRU.PrbAvailUl')]
            for used, avail in pairs:
                if {used, avail}.issubset(df.columns):
                    u = pd.to_numeric(df[used],  errors='coerce')
                    a = pd.to_numeric(df[avail], errors='coerce')
                    ok = u.le(a)
                    ok = ok.astype('float')
                    ok[u.isna() | a.isna()] = np.nan
                    checks.append(ok)
                    det[f'prb_used_le_avail_{tag}_{used}'] = {
                        'applicable': int(ok.notna().sum()),
                        'passed':     int((ok == 1.0).sum())
                    }

        prb_used_le_avail(cell, 'cell')

        # PRB% in [0,100] if present
        def prb_percent_bounds(df: pd.DataFrame, tag: str):
            for pct in ('RRU.PrbTotDl','RRU.PrbTotUl'):
                if pct in df.columns:
                    p = pd.to_numeric(df[pct], errors='coerce')
                    ok = p.ge(0.0) & p.le(100.0)
                    ok = ok.astype('float')
                    ok[p.isna()] = np.nan
                    checks.append(ok)
                    det[f'prb_pct_bounds_{tag}_{pct}'] = {
                        'applicable': int(ok.notna().sum()),
                        'passed':     int((ok == 1.0).sum())
                    }

        prb_percent_bounds(cell, 'cell')

        series = pd.concat(checks, axis=0, ignore_index=True).astype('float') if checks else pd.Series([], dtype='float')
        return series, {'applicable': int(series.notna().sum()), 'passed': int((series == 1.0).sum())}
