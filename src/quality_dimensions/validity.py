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
        self.prb_rule_slack   = dq.get("prb_rule_slack_prb") or {}

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
        active = [s for s in series_all if (s is not None and s.size > 0 and s.notna().any())]
        combined = (
            pd.concat(active, axis=0, ignore_index=True).astype('float')
            if active else pd.Series([], dtype='float')
        )

        apr, mpr, coverage, fails = self._apr_mpr(
            check_series_list=[combined],
            check_tuples_list=[]  # no aggregate checks in validity for now
        )
        self.logger.info(f"validity: v4 details = {v4_details}")
        return {
            'score': mpr,
            'apr': apr,
            'coverage': coverage,
            'details': {
                'V1_types_parsability': v1_details,
                'V2_ranges': v2_details,
                'V3_prb_usage': v3_details,
                'V4_business_rules': v4_details,
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
    
    def _rowwise_all(self, df: pd.DataFrame, passes: Dict[str, pd.Series]) -> pd.Series:
        """
        Combine many per-field boolean series into one per-row series:
          - Treat missing for a field as NA (not applicable)
          - Row passes if ALL applicable fields pass
          - Row is NA if NO field is applicable
        'passes' dict: {col_name: bool_series_with_NA}
        """
        if df is None or df.empty or not passes:
            return pd.Series([], dtype='float')

        # align, ensure boolean with NA
        aligned = []
        present_masks = []
        for key, val in passes.items():
            s = val[0] if isinstance(val, tuple) else val
            if not isinstance(s, pd.Series):
                continue
            # ensure boolean-with-NA semantics preserved
            s = s.reindex(df.index)
            # boolean OK / FAIL, NA where not applicable
            aligned.append(s)
            present_masks.append(s.notna())
        if not aligned:
            return pd.Series([], dtype='float')

        # any field present on the row?
        any_present = pd.concat(present_masks, axis=1).any(axis=1)

        # AND across fields, treating NA as True (so it doesn't fail the row)
        filled = [s.fillna(True) for s in aligned]
        row_ok = pd.concat(filled, axis=1).all(axis=1)

        out = pd.Series(np.nan, index=df.index, dtype='float')
        out.loc[any_present] = row_ok.loc[any_present].astype('float')
        return out

    def _slack_prb_series(self, rule: str, side: str, avail: pd.Series, pct_col: str) -> pd.Series:
        """
        Per-row PRB slack for V4 inequalities.
        rule ∈ {"tot_le_avail","used_le_tot"}
        side ∈ {"Dl","Ul"}
        Uses baseline:
        - learned fixed slack (integer PRBs) from dq_baseline['prb_rule_slack_prb']
        - decimals d from dq_baseline['prb_pct_decimals'] for pct_col
        Returns: Series aligned to avail.index with slack in PRBs (float).
        """
        idx = avail.index

        # 0) config cap (default 3)
        cap = int(getattr(self, 'prb_rule_slack_cap_prb', 3))

        # 1) learned fixed slack
        learned_val = 0
        if isinstance(self.prb_rule_slack, dict):
            key = 'dl' if side.lower() == 'dl' else 'ul'
            try:
                v = self.prb_rule_slack.get(rule, {}).get(key, None)
                if v is not None:
                    learned_val = int(max(0, min(int(v), cap)))
            except Exception:
                learned_val = 0

        learned = pd.Series(float(learned_val), index=idx, dtype='float')

        # 2) decimals-derived slack from baseline d (half ULP in % mapped to PRBs)
        d = 2
        if isinstance(self.prb_pct_decimals, dict):
            d = int(self.prb_pct_decimals.get(pct_col, self.prb_pct_decimals.get(side.lower(), 2)))
        tol_pct = 0.5 * (10.0 ** (-d))                         # half ULP at d decimals
        avail_num = pd.to_numeric(avail, errors='coerce')
        dec_slack = np.ceil((tol_pct / 100.0) * avail_num).astype('float')  # PRBs
        # ensure at least 1 PRB when applicable; then cap
        dec_slack = dec_slack.clip(lower=1.0, upper=float(cap)).fillna(1.0)

        # 3) hybrid: max(learned_fixed, decimals_derived)
        slack = np.fmax(learned.values, dec_slack.values)
        return pd.Series(slack, index=idx, dtype='float')


    def _v1_types_parsability(self, cell: pd.DataFrame, ue: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        # CELL
        cell_pass = {}
        if not cell.empty:
            for col in self.numeric_cell:
                if col in cell.columns:
                    x = pd.to_numeric(cell[col], errors='coerce')
                    # pass if numeric; NA for missing raw values
                    ok = x.notna().astype('float')
                    ok[~cell[col].notna()] = np.nan
                    cell_pass[col] = ok

        s_cell = self._rowwise_all(cell, cell_pass) if cell_pass else pd.Series([], dtype='float')

        # UE
        ue_pass = {}
        if not ue.empty:
            for col in self.numeric_ue:
                if col in ue.columns:
                    x = pd.to_numeric(ue[col], errors='coerce')
                    ok = x.notna().astype('float')
                    ok[~ue[col].notna()] = np.nan
                    ue_pass[col] = ok

        s_ue = self._rowwise_all(ue, ue_pass) if ue_pass else pd.Series([], dtype='float')

        # one series for the component = stack cell+ue row series
        series = (pd.concat([s_cell, s_ue], axis=0, ignore_index=True).astype('float')
                if (s_cell.size or s_ue.size) else pd.Series([], dtype='float'))

        details = {
            'applicable': int(series.notna().sum()),
            'passed':     int((series == 1.0).sum()),
            'cell_rows':  int(s_cell.notna().sum()) if s_cell.size else 0,
            'ue_rows':    int(s_ue.notna().sum()) if s_ue.size else 0,
        }
        return series, details


    def _v2_ranges(self, cell: pd.DataFrame, ue: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        def bounds_for(field: str):
            for sec in ('cell_metrics','ue_metrics'):
                secmap = self.field_ranges.get(sec, {}) or {}
                if field in secmap:
                    ent = secmap[field]
                    lo = ent.get('p01', ent.get('min'))
                    hi = ent.get('p99', ent.get('max'))
                    if lo is not None and hi is not None:
                        lo = float(lo); hi = float(hi)
                        if lo > hi:
                            lo, hi = hi, lo
                        return float(lo), float(hi)
            return self.fallback_ranges.get(field, None)

        # CELL
        cell_pass = {}
        if not cell.empty:
            for col in self.numeric_cell:
                if col in cell.columns:
                    b = bounds_for(col)
                    if not b:
                        continue
                    lo, hi = b
                    s = pd.to_numeric(cell[col], errors='coerce')
                    ok = (s.ge(lo) & s.le(hi)).astype('float')
                    ok[s.isna()] = np.nan
                    cell_pass[col] = ok

        s_cell = self._rowwise_all(cell, cell_pass) if cell_pass else pd.Series([], dtype='float')

        # UE
        ue_pass = {}
        if not ue.empty:
            for col in self.numeric_ue:
                if col in ue.columns:
                    b = bounds_for(col)
                    if not b:
                        continue
                    lo, hi = b
                    s = pd.to_numeric(ue[col], errors='coerce')
                    ok = (s.ge(lo) & s.le(hi)).astype('float')
                    ok[s.isna()] = np.nan
                    ue_pass[col] = ok

        s_ue = self._rowwise_all(ue, ue_pass) if ue_pass else pd.Series([], dtype='float')

        series = (pd.concat([s_cell, s_ue], axis=0, ignore_index=True).astype('float')
                if (s_cell.size or s_ue.size) else pd.Series([], dtype='float'))

        details = {
            'applicable': int(series.notna().sum()),
            'passed':     int((series == 1.0).sum()),
            'cell_rows':  int(s_cell.notna().sum()) if s_cell.size else 0,
            'ue_rows':    int(s_ue.notna().sum()) if s_ue.size else 0,
        }
        return series, details

    def _v3_prb_usage_only(self, cell: pd.DataFrame):
        if cell.empty:
            return pd.Series([], dtype='float'), {}

        passes = {}

        # Non-negative Used/Avail
        for col in ['RRU.PrbUsedDl','RRU.PrbUsedUl','RRU.PrbAvailDl','RRU.PrbAvailUl']:
            if col in cell.columns:
                x = pd.to_numeric(cell[col], errors='coerce')
                ok = x.ge(0).astype('float'); ok[x.isna()] = np.nan
                passes[f'nonneg_{col}'] = ok

        # Used ≤ Avail (DL, UL)
        for used, avail in [('RRU.PrbUsedDl','RRU.PrbAvailDl'), ('RRU.PrbUsedUl','RRU.PrbAvailUl')]:
            if {used, avail}.issubset(cell.columns):
                u = pd.to_numeric(cell[used],  errors='coerce')
                a = pd.to_numeric(cell[avail], errors='coerce')
                ok = (u <= a * (1 + 1e-6)).astype('float')
                ok[u.isna() | a.isna()] = np.nan   
                passes[f'used_le_avail_{used}'] = ok

        series = self._rowwise_all(cell, passes) if passes else pd.Series([], dtype='float')
        details = {'applicable': int(series.notna().sum()), 'passed': int((series == 1.0).sum())}
        return series, details



    def _v4_business_rules(self, cell: pd.DataFrame):
        if cell.empty:
            return pd.Series([], dtype='float'), {}

        passes = {}
        eps = 1e-6

        for side in ('Dl','Ul'):
            tot   = f'RRU.PrbTot{side}_abs'
            avail = f'RRU.PrbAvail{side}'
            used  = f'RRU.PrbUsed{side}'
            pct   = f'RRU.PrbTot{side}'

            # build per-row slack from baseline (hybrid)
            slack = None
            if avail in cell.columns:
                a_ser = pd.to_numeric(cell[avail], errors='coerce')
                slack = self._slack_prb_series(rule='tot_le_avail', side=side, avail=a_ser, pct_col=pct)

            # tot_abs ≤ avail (+ slack)
            if {tot, avail}.issubset(cell.columns):
                t = pd.to_numeric(cell[tot],   errors='coerce')
                a = pd.to_numeric(cell[avail], errors='coerce')
                if slack is None:
                    slack = pd.Series(1.0, index=cell.index, dtype='float')
                thresh = a + slack
                ok = (t <= (thresh * (1 + eps))).astype('float')
                ok[t.isna() | a.isna() | thresh.isna()] = np.nan
                passes[f'tot_abs_le_avail_{side}'] = ok

            # used ≤ tot_abs (+ slack)  — use the same hybrid idea (rule='used_le_tot')
            if {tot, used}.issubset(cell.columns):
                t = pd.to_numeric(cell[tot],  errors='coerce')
                u = pd.to_numeric(cell[used], errors='coerce')
                # if avail missing above, still compute dec-based slack from pct_col:
                if slack is None and avail in cell.columns:
                    a_ser = pd.to_numeric(cell[avail], errors='coerce')
                if avail in cell.columns:
                    slack_used = self._slack_prb_series(rule='used_le_tot', side=side, avail=a_ser, pct_col=pct)
                else:
                    slack_used = pd.Series(float(getattr(self, 'prb_rule_slack_cap_prb', 3)), index=cell.index, dtype='float')

                thresh = t + slack_used
                ok = (u <= (thresh * (1 + eps))).astype('float')
                ok[t.isna() | u.isna() | thresh.isna()] = np.nan
                passes[f'used_le_tot_abs_{side}'] = ok

            # Avail == 0  =>  Used == 0
            if {used, avail}.issubset(cell.columns):
                u = pd.to_numeric(cell[used],  errors='coerce')
                a = pd.to_numeric(cell[avail], errors='coerce')
                cond = a.eq(0) & a.notna() & u.notna()
                s = pd.Series(np.nan, index=cell.index, dtype='float')
                if cond.any():
                    s.loc[cond] = (u.loc[cond] == 0).astype('float')
                passes[f'avail0_implies_used0_{side}'] = s

        # PRB% bounds only if not already covered by field_ranges
        for pct in ('RRU.PrbTotDl','RRU.PrbTotUl'):
            if pct in cell.columns and not self._has_range_in_baseline(pct):
                p = pd.to_numeric(cell[pct], errors='coerce')
                ok = (p.ge(0.0) & p.le(100.0)).astype('float')
                ok[p.isna()] = np.nan
                passes[f'prb_pct_bounds_{pct}'] = ok

        #self.logger.info(f"DEBUG: passes dictionary keys = {passes.keys()}")
        #self.logger.info(f"DEBUG: passes dictionary = {passes}")
        series = self._rowwise_all(cell, passes) if passes else pd.Series([], dtype='float')
        details = {'applicable': int(series.notna().sum()), 'passed': int((series == 1.0).sum())}
        return series, details


    def _v5_prb_percent_identity(self, cell: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """
        Row-wise PRB % identity using learned decimals from dq_baseline:
        If RRU.PrbTot{Dl,Ul}_abs is present:
            pct_calc = (PrbTot_abs / PrbAvail) * 100
            pass if |pct_calc - RRU.PrbTot%| <= 0.5 * 10^{-d} (+ tiny eps)
        Decimals are stored keyed by percent column name in dq_baseline['prb_pct_decimals'].
        If _abs is missing for a side, we skip that side (NA) instead of forcing a check.
        """
        details: Dict = {}
        # require a decimals map
        if cell.empty or not isinstance(self.prb_pct_decimals, dict):
            return pd.Series([], dtype='float'), {'note': 'no decimals baseline'}

        checks: Dict[str, pd.Series] = {}

        for side, avail_col, pct_col in [
            ('dl', 'RRU.PrbAvailDl', 'RRU.PrbTotDl'),
            ('ul', 'RRU.PrbAvailUl', 'RRU.PrbTotUl'),
        ]:
            if not {avail_col, pct_col}.issubset(cell.columns):
                continue

            tot_abs_col = pct_col + '_abs'  # e.g., RRU.PrbTotDl_abs
            # If absolute total not available, mark NA for this side (not applicable)
            if tot_abs_col not in cell.columns:
                checks[f'identity_{side}'] = pd.Series(np.nan, index=cell.index, dtype='float')
                continue

            avail = pd.to_numeric(cell[avail_col], errors='coerce')
            tot_abs = pd.to_numeric(cell[tot_abs_col], errors='coerce')
            pct    = pd.to_numeric(cell[pct_col],   errors='coerce')

            s = pd.Series(np.nan, index=cell.index, dtype='float')
            valid = avail.notna() & tot_abs.notna() & pct.notna() & (avail > 0)

            if valid.any():
                # decimals keyed by the percent column name
                d = int(self.prb_pct_decimals.get(pct_col, 2))
                tol = (10.0 ** (-d)) / 2.0 + 1e-9

                calc = (tot_abs / avail) * 100.0
                # clamp to sane range
                calc = calc.clip(lower=0.0, upper=100.0)
                tgt  = pct.clip(lower=0.0, upper=100.0)

                ok = (calc.clip(0,100) - tgt.clip(0,100)).abs() <= tol  # keep Series (not ndarray)
                s.loc[valid] = ok.loc[valid].astype('float')

            checks[f'identity_{side}'] = s

        # Combine DL & UL identity per row (NA if neither side applicable)
        series = self._rowwise_all(cell, checks) if checks else pd.Series([], dtype='float')
        details['applicable'] = int(series.notna().sum())
        details['passed']     = int((series == 1.0).sum())
        
        return series, details

      