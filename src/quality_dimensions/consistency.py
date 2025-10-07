# consistency.py
"""
Phase: 2
Purpose: Consistency dimension (CS1 intra-record, CS2 energy↔power)
Depends on: dq_baseline (cadence_sec, energy_power_band), field_ranges (optional)
"""

from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd

from .base_dimension import BaseDimension
from common.constants import COLUMN_NAMES, MEAS_INTERVAL_SEC

class ConsistencyDimension(BaseDimension):
    def __init__(self):
        super().__init__('consistency')
        dq = self.get_dq_baseline() or {}
        self.cadence_sec: float = float(dq.get('cadence_sec', MEAS_INTERVAL_SEC))
        self.recon_band_thp  = dq.get('ue_cell_thp_ratio') or {} 
        self.recon_band_prb = dq.get('ue_cell_prb_ratio') or {}
        #self.logger.info(f"Consistency init: thp band={self.recon_band_thp }, prb_band={self.recon_band_prb}")
        self.recon_min_n = int(getattr(self, 'reconciliation_min_samples', 30))
        self.recon_window_min_samples = 5
        self.ts_col   = COLUMN_NAMES.get('timestamp', 'timestamp')
        self.cell_col = COLUMN_NAMES.get('cell_entity', 'Viavi.Cell.Name')
        self.ue_col   = COLUMN_NAMES.get('ue_entity', 'Viavi.UE.Name')
        # Learned band for CS2 (IQR band from Phase-1); used if present
        self.energy_band = (dq.get('energy_power_band') or {})
        # Optional: field ranges to infer 'near-zero throughput' floor (no hardcode)
        self.field_ranges = self.load_artifact_baseline('field_ranges') or {}
        #self.logger.info(f"Consistency init: cadence={self.cadence_sec}s, energy_band={self.energy_band or 'NA'}")
        self.logger.info("Consistency initiated")

    def calculate_score(self, window_data: Dict) -> Dict:
        ok, err = self.validate_window_data(window_data)
        if not ok:
            return {'score': 0.0, 'coverage': 0.0, 'status': 'ERROR',
                    'details': {'validation_error': err}}

        cell = window_data.get('cell_data', pd.DataFrame())
        ue   = window_data.get('ue_data', pd.DataFrame())

        # CS1: Intra-record relationships (row-wise)
        cs1_series, cs1_details = self._cs1_intra_record(cell, ue)

        # CS2: Energy↔Power temporal identity (aggregate)
        cs2_tuple, cs2_details = self._cs2_energy_identity(cell)

        #CS3: throughput reconciliation
        cs3t_tuple, cs3t_details = self._cs3_throughput_reconciliation(cell, ue)

        #CS3: prb reconciliation
        cs3p_tuple, cs3p_details = self._cs3_prb_reconciliation(cell, ue)

        tuples_all = [cs2_tuple, cs3t_tuple, cs3p_tuple]
        active_tuples = [t for t in tuples_all if (t[1] > 0)]
        series_all = [cs1_series]
        active_series = [s for s in series_all if s.notna().any()]

        apr, mpr, coverage, fails = self._apr_mpr(
            check_series_list=active_series,
            check_tuples_list=active_tuples
        )
        #self.logger.info(f"Consistency: cs2 ={cs2_details}, cs3 thp={cs3t_details}, cs3 prb={cs3p_details}")
        return {
            'score': mpr,
            'apr': apr,
            'coverage': coverage,
            'details': {
                'CS1_intra_record': cs1_details,
                'CS2_energy_identity': cs2_details,
                'CS3_throughput_reconciliation': cs3t_details,
                'CS3_prb_reconciliation': cs3p_details,
                'fail_counts': fails
            }
        }

    # ------------ CS1: Intra-record (row-wise) ------------
    def _cs1_intra_record(self, cell: pd.DataFrame, ue: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """
        Constraints (applied if columns exist):
          1) RRC.ConnMean ≤ RRC.ConnMax  [cells]
          2) If ConnMean == 0 ⇒ throughput ≈ 0 (Dl & Ul) [cells, ues]
          3) If throughput > 0 ⇒ PRB Used > 0  [cells]
        Rows where a sub-check is inapplicable → NaN (not counted).
        """
        details: Dict = {}
        checks: List[pd.Series] = []

        # Helper to get a data-driven "near-zero throughput" floor from field_ranges (p01)
        def zero_floor_gbps() -> float:
            fr = self.field_ranges or {}
            c = fr.get('cell_metrics', {})
            u = fr.get('ue_metrics', {})
            vals = []
            for key in ['DRB.UEThpDl','DRB.UEThpUl']:
                if key in c and 'p01' in c[key]: vals.append(float(c[key]['p01']))
                if key in u and 'p01' in u[key]: vals.append(float(u[key]['p01']))
            # fallback to tiny 1 Mbps in Gbps if ranges missing
            return float(np.nanmedian(vals)) if vals else 0.001  # 0.001 Gbps = 1 Mbps

        zfloor = zero_floor_gbps()

        # 1) ConnMean ≤ ConnMax (cells)
        if not cell.empty and {'RRC.ConnMean','RRC.ConnMax'}.issubset(cell.columns):
            a = pd.to_numeric(cell['RRC.ConnMean'], errors='coerce')
            b = pd.to_numeric(cell['RRC.ConnMax'],  errors='coerce')
            s = (a <= b)
            checks.append(s.astype('float'))
            details['conn_mean_le_max_total']  = int(s.notna().sum())
            details['conn_mean_le_max_passed'] = int(s.sum())

        # 2) If ConnMean == 0 ⇒ throughput ≈ 0 (cells)
        if not cell.empty and {'RRC.ConnMean','DRB.UEThpDl','DRB.UEThpUl'}.issubset(cell.columns):
            c0 = pd.to_numeric(cell['RRC.ConnMean'], errors='coerce').fillna(-1).eq(0)
            dl = pd.to_numeric(cell['DRB.UEThpDl'], errors='coerce')
            ul = pd.to_numeric(cell['DRB.UEThpUl'], errors='coerce')
            cond = c0
            s = pd.Series(np.nan, index=cell.index, dtype='float')
            if cond.any():
                ok = dl.abs().le(zfloor) & ul.abs().le(zfloor)
                s.loc[cond] = ok.loc[cond].astype('float')
            checks.append(s)
            details['conn0_cells_applicable'] = int(cond.sum())
            details['conn0_cells_passed']     = int((s == 1.0).sum())

        # 2b) If ConnMean == 0 ⇒ throughput ≈ 0 (UEs) — UE table may not have ConnMean
        if not ue.empty and {'DRB.UEThpDl','DRB.UEThpUl','RRC.ConnMean'}.issubset(ue.columns):
            cond = pd.to_numeric(ue['RRC.ConnMean'], errors='coerce').fillna(-1).eq(0)
            dl = pd.to_numeric(ue['DRB.UEThpDl'], errors='coerce')
            ul = pd.to_numeric(ue['DRB.UEThpUl'], errors='coerce')
            s = pd.Series(np.nan, index=ue.index, dtype='float')
            if cond.any():
                ok = dl.abs().le(zfloor) & ul.abs().le(zfloor)
                s.loc[cond] = ok.loc[cond].astype('float')
            checks.append(s)
            details['conn0_ues_applicable'] = int(cond.sum())
            details['conn0_ues_passed']     = int((s == 1.0).sum())

        # 3) If throughput > 0 ⇒ PRB Used > 0 (cells)
        if not cell.empty and {'DRB.UEThpDl','DRB.UEThpUl','RRU.PrbUsedDl','RRU.PrbUsedUl'}.issubset(cell.columns):
            dl = pd.to_numeric(cell['DRB.UEThpDl'], errors='coerce')
            ul = pd.to_numeric(cell['DRB.UEThpUl'], errors='coerce')
            pudl = pd.to_numeric(cell['RRU.PrbUsedDl'], errors='coerce')
            puul = pd.to_numeric(cell['RRU.PrbUsedUl'], errors='coerce')
            cond = (dl.gt(zfloor) | ul.gt(zfloor))
            s = pd.Series(np.nan, index=cell.index, dtype='float')
            if cond.any():
                ok = pudl.gt(0) | puul.gt(0)
                s.loc[cond] = ok.loc[cond].astype('float')
            checks.append(s)
            details['thp_pos_applicable'] = int(cond.sum())
            details['thp_pos_passed']     = int((s == 1.0).sum())

        series = (pd.concat(checks, axis=0, ignore_index=True).astype('float')
                  if checks else pd.Series([], dtype='float'))
        details['applicable'] = int(series.notna().sum())
        details['passed']     = int((series == 1.0).sum())
        return series, details

    # ------------ CS2: Energy↔Power identity (aggregate) ------------
    def _cs2_energy_identity(self, cell: pd.DataFrame) -> Tuple[Tuple[int, int], Dict]:
        """
        Window-level energy identity using actual per-row Δt:
           ratio = Σ Energy_interval  /  Σ (AvgPower[kW] * Δt_hours)
        Pass if ratio in learned IQR band from dq_baseline (fallback: 0.8–1.2).
        """
        details: Dict = {}
        if cell.empty or not {self.ts_col, self.cell_col}.issubset(cell.columns):
            return (0, 0), {'note': 'no cell data or missing timestamp/entity'}

        en_col = 'PEE.Energy_interval'
        pw_col = COLUMN_NAMES.get('avg_power', 'PEE.AvgPower')

        if en_col not in cell.columns or pw_col not in cell.columns:
            return (0, 0), {'note': 'missing energy/power columns'}

        df = cell[[self.cell_col, self.ts_col, en_col, pw_col] +
                  [c for c in ['PEE.Energy_reset','PEE.Energy_imputed'] if c in cell.columns]].copy()

        # Parse & sort
        df[self.ts_col] = pd.to_datetime(df[self.ts_col], errors='coerce')
        df[en_col] = pd.to_numeric(df[en_col], errors='coerce')
        df[pw_col] = pd.to_numeric(df[pw_col], errors='coerce')
        df = df.dropna(subset=[self.ts_col]).sort_values([self.cell_col, self.ts_col])

        # Actual per-entity Δt (h), fallback to configured cadence if first row
        dt_h = df.groupby(self.cell_col)[self.ts_col].diff().dt.total_seconds().div(3600.0)
        dt_h = dt_h.fillna(self.cadence_sec / 3600.0).clip(lower=1e-9)

        # Exclude resets/imputed intervals from both sums if flags exist
        mask = df[en_col].notna() & df[pw_col].notna() & dt_h.notna()
        if 'PEE.Energy_reset' in df.columns:
            mask &= (df['PEE.Energy_reset'] != True)
        if 'PEE.Energy_imputed' in df.columns:
            mask &= (df['PEE.Energy_imputed'] != True)

        if not mask.any():
            return (0, 0), {'note': 'no valid rows for identity after filters'}

        energy_sum = float(df.loc[mask, en_col].sum())
        power_kWh_sum = float(((df.loc[mask, pw_col] / 1000.0) * dt_h.loc[mask]).sum())

        if power_kWh_sum <= 0:
            return (0, 0), {'note': 'zero denominator (power×time)'}

        ratio = energy_sum / power_kWh_sum
        # Learned band (IQR) from dq_baseline (Phase-1); fallback 0.5–2.0 if missing
        
        # --- pass/fail: learned band (if reliable) OR coarse guard-rails ---
        min_n = self.recon_min_n  # 30 by default
        band = None
        band_n = 0
        if self.energy_band:
            band_n = int(self.energy_band.get('n', 0))
            has_band = ('ratio_q1' in self.energy_band) and ('ratio_q3' in self.energy_band)
            reliable = has_band and (band_n == 0 or band_n >= min_n)
            if reliable:
                band = (float(self.energy_band['ratio_q1']),
                        float(self.energy_band['ratio_q3']))

        # coarse rails from design v2.1
        coarse_lo, coarse_hi = 0.5, 2.0

        band_pass   = (band is not None) and (band[0] <= ratio <= band[1])
        coarse_pass = (coarse_lo <= ratio <= coarse_hi)
        passed, total = int(band_pass or coarse_pass), 1

        details.update({
            'ratio': ratio,
            'q1': (band[0] if band else None),
            'q3': (band[1] if band else None),
            'coarse_lo': coarse_lo, 'coarse_hi': coarse_hi,
            'used_band': ('learned' if band else 'coarse'),
            'band_n': band_n,
            'energy_sum_kWh': energy_sum,
            'power_time_kWh': power_kWh_sum,
            'rows_used': int(mask.sum())
        })
        return (passed, total), details


    def _cs3_throughput_reconciliation(self, cell: pd.DataFrame, ue: pd.DataFrame):
        details = {}

        ts = self.ts_col
        thp_dl, thp_ul = 'DRB.UEThpDl', 'DRB.UEThpUl'
        need_cell = {ts, thp_dl, thp_ul}.issubset(cell.columns)
        need_ue   = {ts, thp_dl, thp_ul}.issubset(ue.columns)
        if not (need_cell and need_ue):
            return (0, 0), {'note': 'missing throughput columns'}

        if not (self.recon_band_thp .get('dl') or self.recon_band_thp .get('ul')):
            return (0, 0), {'note': 'reconciliation band unavailable'}


        c = cell[[ts, thp_dl, thp_ul]].copy()
        u = ue[[ts, thp_dl, thp_ul]].copy()
        c[ts] = pd.to_datetime(c[ts], errors='coerce'); c = c.dropna(subset=[ts])
        u[ts] = pd.to_datetime(u[ts], errors='coerce'); u = u.dropna(subset=[ts])

        # --- SNAP to cadence grid so Cell & UE align on same ticks ---
        step   = pd.to_timedelta(self.cadence_sec, unit='s')
        anchor = min(c[ts].min(), u[ts].min()).floor(f'{int(self.cadence_sec)}S')

        def snap(s: pd.Series) -> pd.Series:
            k = np.rint((s - anchor) / step).astype(int)
            return anchor + k * step

        c['__t'] = snap(c[ts]); u['__t'] = snap(u[ts])

        cg = c.groupby('__t', as_index=True).agg({thp_dl:'sum', thp_ul:'sum'}).rename(
            columns={thp_dl:'cell_dl', thp_ul:'cell_ul'}
        )
        ug = u.groupby('__t', as_index=True).agg({thp_dl:'sum', thp_ul:'sum'}).rename(
            columns={thp_dl:'ue_dl',   thp_ul:'ue_ul'}
        )
        g = cg.join(ug, how='inner')
        if g.empty:
            return (0, 0), {'note': 'no overlapping timestamps after snap'}


        rows = []

        # DL
        if self.recon_band_thp.get('dl'):
            mask = g['cell_dl'] > 0
            r = (g.loc[mask, 'ue_dl'] / g.loc[mask, 'cell_dl']).replace([np.inf, -np.inf], np.nan).dropna()
            if r.size >= self.recon_window_min_samples:
                q25 = float(self.recon_band_thp['dl']['q25']); q75 = float(self.recon_band_thp['dl']['q75'])
                iqr = max(q75 - q25, 1e-9)
                lo  = q25 - 1.5 * iqr
                hi  = q75 + 1.5 * iqr

                med = float(np.median(r))
                cov = float(((r >= q25) & (r <= q75)).mean())
                band_pass   = (lo <= med <= hi)
                coverage_ok = (cov >= 0.5)

                passed, total = int(band_pass or coverage_ok), 1
                rows.append((passed, total, {
                    'side':'dl','q25':q25,'q75':q75,'iqr':iqr,'lo':lo,'hi':hi,
                    'median_win':med,'coverage_in_iqr':cov,'n_win':int(r.size)
                }))
            else:
                rows.append((0, 0, {'side':'dl','note':f'insufficient window samples ({r.size}<{self.recon_window_min_samples})'}))

        # UL
        if self.recon_band_thp.get('ul'):
            mask = g['cell_ul'] > 0
            r = (g.loc[mask, 'ue_ul'] / g.loc[mask, 'cell_ul']).replace([np.inf, -np.inf], np.nan).dropna()
            if r.size >= self.recon_window_min_samples:
                q25 = float(self.recon_band_thp['ul']['q25']); q75 = float(self.recon_band_thp['ul']['q75'])
                iqr = max(q75 - q25, 1e-9)
                lo  = q25 - 1.5 * iqr
                hi  = q75 + 1.5 * iqr

                med = float(np.median(r))
                cov = float(((r >= q25) & (r <= q75)).mean())
                band_pass   = (lo <= med <= hi)
                coverage_ok = (cov >= 0.5)

                passed, total = int(band_pass or coverage_ok), 1
                rows.append((passed, total, {
                    'side':'ul','q25':q25,'q75':q75,'iqr':iqr,'lo':lo,'hi':hi,
                    'median_win':med,'coverage_in_iqr':cov,'n_win':int(r.size)
                }))
            else:
                rows.append((0, 0, {'side':'ul','note':f'insufficient window samples ({r.size}<{self.recon_window_min_samples})'}))
        if not rows:
            return (0, 0), {'note': 'no valid ratios in window'}

        # Combine DL+UL
        passed = sum(p for p, t, _ in rows)
        total  = sum(t for _, t, _ in rows)
        details = {'subchecks': [d for _, _, d in rows]}
        return (passed, total), details

    def _cs3_prb_reconciliation(self, cell: pd.DataFrame, ue: pd.DataFrame):
        details = {}
        ts = self.ts_col
        prb_dl, prb_ul = 'RRU.PrbUsedDl', 'RRU.PrbUsedUl'
        need_cell = {ts, prb_dl, prb_ul}.issubset(cell.columns)
        need_ue   = {ts, prb_dl, prb_ul}.issubset(ue.columns)
        if not (need_cell and need_ue):
            return (0, 0), {'note': 'missing PRB columns'}

        if not (self.recon_band_prb.get('dl') or self.recon_band_prb.get('ul')):
            return (0, 0), {'note': 'PRB reconciliation band unavailable'}

        c = cell[[ts, prb_dl, prb_ul]].copy()
        u = ue[[ts, prb_dl, prb_ul]].copy()
        c[ts] = pd.to_datetime(c[ts], errors='coerce'); c = c.dropna(subset=[ts])
        u[ts] = pd.to_datetime(u[ts], errors='coerce'); u = u.dropna(subset=[ts])

        # --- SNAP to cadence grid so Cell & UE align on same ticks ---
        step   = pd.to_timedelta(self.cadence_sec, unit='s')
        anchor = min(c[ts].min(), u[ts].min()).floor(f'{int(self.cadence_sec)}S')

        def snap(s: pd.Series) -> pd.Series:
            k = np.rint((s - anchor) / step).astype(int)
            return anchor + k * step

        c['__t'] = snap(c[ts]); u['__t'] = snap(u[ts])

        cg = c.groupby('__t', as_index=True).agg({prb_dl:'sum', prb_ul:'sum'}).rename(
            columns={prb_dl:'cell_dl', prb_ul:'cell_ul'}
        )
        ug = u.groupby('__t', as_index=True).agg({prb_dl:'sum', prb_ul:'sum'}).rename(
            columns={prb_dl:'ue_dl',   prb_ul:'ue_ul'}
        )
        g = cg.join(ug, how='inner')
        if g.empty:
            return (0, 0), {'note': 'no overlapping timestamps after snap'}


        rows = []
        # DL
        if self.recon_band_prb.get('dl'):
            mask = g['cell_dl'] > 0
            r = (g.loc[mask, 'ue_dl'] / g.loc[mask, 'cell_dl']).replace([np.inf, -np.inf], np.nan).dropna()
            if r.size >= self.recon_window_min_samples:
                q25 = float(self.recon_band_prb['dl']['q25']); q75 = float(self.recon_band_prb['dl']['q75'])
                iqr = max(q75 - q25, 1e-9); lo = q25 - 1.5*iqr; hi = q75 + 1.5*iqr
                med = float(np.median(r));  cov = float(((r >= q25) & (r <= q75)).mean())
                passed, total = int((lo <= med <= hi) or (cov >= 0.5)), 1
                rows.append((passed, total, {
                    'side':'dl','q25':q25,'q75':q75,'iqr':iqr,'lo':lo,'hi':hi,
                    'median_win':med,'coverage_in_iqr':cov,'n_win':int(r.size)
                }))
            else:
                rows.append((0, 0, {'side':'dl','note':f'insufficient window samples ({r.size}<{self.recon_window_min_samples})'}))

        # UL
        if self.recon_band_prb.get('ul'):
            mask = g['cell_ul'] > 0
            r = (g.loc[mask, 'ue_ul'] / g.loc[mask, 'cell_ul']).replace([np.inf, -np.inf], np.nan).dropna()
            if r.size >= self.recon_window_min_samples:
                q25 = float(self.recon_band_prb['ul']['q25']); q75 = float(self.recon_band_prb['ul']['q75'])
                iqr = max(q75 - q25, 1e-9); lo = q25 - 1.5*iqr; hi = q75 + 1.5*iqr
                med = float(np.median(r));  cov = float(((r >= q25) & (r <= q75)).mean())
                passed, total = int((lo <= med <= hi) or (cov >= 0.5)), 1
                rows.append((passed, total, {
                    'side':'ul','q25':q25,'q75':q75,'iqr':iqr,'lo':lo,'hi':hi,
                    'median_win':med,'coverage_in_iqr':cov,'n_win':int(r.size)
                }))
            else:
                rows.append((0, 0, {'side':'ul','note':f'insufficient window samples ({r.size}<{self.recon_window_min_samples})'}))


        if not rows:
            return (0, 0), {'note': 'no valid PRB ratios in window'}

        passed = sum(p for p,t,_ in rows)
        total  = sum(t for _,t,_ in rows)
        details = {'subchecks': [d for _,_,d in rows]}
        return (passed, total), details