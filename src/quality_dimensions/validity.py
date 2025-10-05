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
        dq = self.get_dq_baseline() or {}
        self.prb_pct_decimals = dq.get('prb_pct_decimals', None)

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

        v3_series, v3_details = self._v3_prb_usage_only(cell)
        
        v4_series, v4_details = self._v4_business_rules(cell)

        v5_series, v5_details = self._v5_prb_percent_identity(cell) 

        series_all = [v1_series, v2_series, v3_series, v4_series, v5_series]
        active = [s for s in series_all if s.notna().any()]

        apr, mpr, coverage, fails = self._apr_mpr(
            check_series_list=active,
            check_tuples_list=[]  # no aggregate checks in validity for now
        )

        return {
            'score': mpr,
            'apr': apr,
            'coverage': coverage,
            'details': {
                'V1_types_parsability': v1_details,
                'V2_ranges': v2_details,
                'V3_prb_usage': v3_details,
                'V4_biz_rules': v4_details,
                'V5_prb_percent_identity': v5_details,
                'fail_counts': fails
            }
        }

    # ---------- Helpers ----------

    def _has_range_in_baseline(self, field: str) -> bool:
        for sec in ('cell_metrics','ue_metrics'):
            secmap = self.field_ranges.get(sec, {}) or {}
            if field in secmap:
                lo, hi = secmap[field].get('min'), secmap[field].get('max')
                if lo is not None and hi is not None:
                    return True
        return False

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

    def _v3_prb_usage_only(self, cell: pd.DataFrame):
        checks = []; det = {}
        if cell.empty:
            return pd.Series([], dtype='float'), det

        pairs = [('RRU.PrbUsedDl','RRU.PrbAvailDl'), ('RRU.PrbUsedUl','RRU.PrbAvailUl')]

        # Non-negative (only for Used/Avail)
        for col in ['RRU.PrbUsedDl','RRU.PrbUsedUl','RRU.PrbAvailDl','RRU.PrbAvailUl']:
            if col in cell.columns:
                x = pd.to_numeric(cell[col], errors='coerce')
                ok = x.ge(0).astype('float'); ok[x.isna()] = np.nan
                checks.append(ok)
                det[f'nonneg_{col}'] = {'applicable': int(ok.notna().sum()), 'passed': int((ok==1.0).sum())}

        # Used ≤ Avail
        for used, avail in pairs:
            if {used, avail}.issubset(cell.columns):
                u = pd.to_numeric(cell[used],  errors='coerce')
                a = pd.to_numeric(cell[avail], errors='coerce')
                ok = u.le(a).astype('float'); ok[u.isna() | a.isna()] = np.nan
                checks.append(ok)
                det[f'used_le_avail_{used}'] = {'applicable': int(ok.notna().sum()), 'passed': int((ok==1.0).sum())}

        series = (pd.concat(checks, axis=0, ignore_index=True).astype('float')
                if checks else pd.Series([], dtype='float'))
        return series, det


    def _v4_business_rules(self, cell: pd.DataFrame):
        checks = []; det = {}
        if cell.empty:
            return pd.Series([], dtype='float'), det

        # If absolute totals present, enforce Avail ≤ Tot_abs and Used ≤ Tot_abs
        for side in ('Dl','Ul'):
            tot = f'RRU.PrbTot{side}_abs'
            avail = f'RRU.PrbAvail{side}'
            used  = f'RRU.PrbUsed{side}'
            if tot in cell.columns and avail in cell.columns:
                t = pd.to_numeric(cell[tot], errors='coerce')
                a = pd.to_numeric(cell[avail], errors='coerce')
                ok = a.le(t).astype('float'); ok[t.isna() | a.isna()] = np.nan
                checks.append(ok)
                det[f'avail_le_tot_{side}'] = {'applicable': int(ok.notna().sum()), 'passed': int((ok==1.0).sum())}
            if tot in cell.columns and used in cell.columns:
                t = pd.to_numeric(cell[tot], errors='coerce')
                u = pd.to_numeric(cell[used], errors='coerce')
                ok = u.le(t).astype('float'); ok[t.isna() | u.isna()] = np.nan
                checks.append(ok)
                det[f'used_le_tot_{side}'] = {'applicable': int(ok.notna().sum()), 'passed': int((ok==1.0).sum())}

        # If Avail == 0 ⇒ Used must be 0 (cells)
        for used, avail in [('RRU.PrbUsedDl','RRU.PrbAvailDl'), ('RRU.PrbUsedUl','RRU.PrbAvailUl')]:
            if {used, avail}.issubset(cell.columns):
                u = pd.to_numeric(cell[used],  errors='coerce')
                a = pd.to_numeric(cell[avail], errors='coerce')
                cond = a.eq(0) & a.notna() & u.notna()
                s = pd.Series(np.nan, index=cell.index, dtype='float')
                if cond.any():
                    s.loc[cond] = (u.loc[cond] == 0).astype('float')
                checks.append(s)
                det[f'avail0_implies_used0_{used}'] = {
                    'applicable': int(cond.sum()),
                    'passed':     int((s == 1.0).sum())
                }

        # PRB% in [0,100] if present (kept here as a “business constraint”)
        for pct in ('RRU.PrbTotDl','RRU.PrbTotUl'):
            if pct in cell.columns and not self._has_range_in_baseline(pct):
                p = pd.to_numeric(cell[pct], errors='coerce')
                ok = (p.ge(0.0) & p.le(100.0)).astype('float'); ok[p.isna()] = np.nan
                checks.append(ok)
                det[f'prb_pct_bounds_{pct}'] = {'applicable': int(ok.notna().sum()), 'passed': int((ok==1.0).sum())}

        series = (pd.concat(checks, axis=0, ignore_index=True).astype('float')
                if checks else pd.Series([], dtype='float'))
        return series, det


    def _v5_prb_percent_identity(self, cell: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """
        Row-wise identity using learned decimals:
          round(Used/Avail*100, d_side) == round(PrbTot, d_side)
        Only applies if decimals present and Avail>0.
        """
        details: Dict = {}
        if cell.empty or self.prb_pct_decimals is None:
            return pd.Series([], dtype='float'), {'note': 'no decimals baseline'}

        checks: List[pd.Series] = []

        for side, used_col, avail_col, pct_col in [
            ('dl', 'RRU.PrbUsedDl', 'RRU.PrbAvailDl', 'RRU.PrbTotDl'),
            ('ul', 'RRU.PrbUsedUl', 'RRU.PrbAvailUl', 'RRU.PrbTotUl'),
        ]:
            if not {used_col, avail_col, pct_col}.issubset(cell.columns):
                continue
            d = int(self.prb_pct_decimals.get(side, 2))
            used  = pd.to_numeric(cell[used_col],  errors='coerce')
            avail = pd.to_numeric(cell[avail_col], errors='coerce')
            pct   = pd.to_numeric(cell[pct_col],   errors='coerce')

            s = pd.Series(np.nan, index=cell.index, dtype='float')
            valid = used.notna() & avail.notna() & pct.notna() & (avail > 0)
            if valid.any():
                calc = (used / avail) * 100.0
                ok = np.round(calc, d) == np.round(pct, d)
                s.loc[valid] = ok.loc[valid].astype('float')

            checks.append(s)

        series = (pd.concat(checks, axis=0, ignore_index=True).astype('float')
                  if checks else pd.Series([], dtype='float'))
        details['applicable'] = int(series.notna().sum())
        details['passed']     = int((series == 1.0).sum())
        return series, details