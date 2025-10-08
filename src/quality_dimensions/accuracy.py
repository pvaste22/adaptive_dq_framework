import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from common.constants import COLUMN_NAMES, NEAR_ZERO_THRESHOLDS, PATTERNS, BAND_SPECS
from .base_dimension import BaseDimension

class AccuracyDimension(BaseDimension):
    """
    Dimension 5: ACCURACY
    A1: Throughput per-PRB efficiency (aggregate, DL/UL) within training IQR bands
    A2: Spectral efficiency trip-wire (row-wise) vs cap (global ∧ p99-based)
    A3: Context rule (row-wise): Thp > eps  =>  PRB_Used > 0
    Notes:
      - Throughput columns are in Gbps (dataset spec), using 180 kHz/PRB unless overridden by baseline.
      - No hardcoded tolerances; everything from dq_baseline or config thresholds.
    """

    def __init__(self):
        super().__init__(name='accuracy')

        # Column names (fallbacks kept robust)
        cn = COLUMN_NAMES
        self.ts_col         = cn.get('timestamp'     , 'timestamp')
        self.cell_ent_col   = cn.get('cell_entity'   , 'Viavi.Cell.Name')
        self.ue_ent_col     = cn.get('ue_entity'     , 'Viavi.UE.Name')
        self.thp_dl_col     = cn.get('throughput_dl' , 'DRB.UEThpDl')   # Gbps
        self.thp_ul_col     = cn.get('throughput_ul' , 'DRB.UEThpUl')   # Gbps
        self.prb_used_dl    = cn.get('prb_used_dl'   , 'RRU.PrbUsedDl')
        self.prb_used_ul    = cn.get('prb_used_ul'   , 'RRU.PrbUsedUl')
        self.prb_avail_dl   = cn.get('prb_avail_dl'  , 'RRU.PrbAvailDl')
        self.prb_avail_ul   = cn.get('prb_avail_ul'  , 'RRU.PrbAvailUl')
        # thresholds
        self.eps_gbps       = NEAR_ZERO_THRESHOLDS.get('throughput_gbps', 0.001)  # ~1 Mbps

        # ---- Baselines from Phase-1 (dq_baseline) ----
        dq = self.get_dq_baseline() or {}
        # A1 bands (Mbps/PRB), structure: {'dl': {'q25':..,'q50':..,'q75':..,'n':..}, 'ul': {...}}
        self.a1_bands = dq.get('accuracy_thp_per_prb_band', {}) or {}
        # A2 spectral-eff info
        se_info = dq.get('accuracy_spectral_eff', {}) or {}
        self.per_prb_khz      = float(se_info.get('per_prb_khz', 180.0))  # LTE-like default
        self.se_dl_p99        = se_info.get('dl_p99', None)
        self.se_ul_p99        = se_info.get('ul_p99', None)
        self.se_abs_cap_glob  = float(se_info.get('abs_cap_global', 30.0))
        self.se_abs_cap_byband= se_info.get('abs_cap_by_band', None)  # optional; not required

        self.logger.info("Accuracy initiated")

        #self.logger.info(
            #f"Accuracy initialized: A1 bands present={bool(self.a1_bands)}, "
            #f"A2 caps (global={self.se_abs_cap_glob}, p99_dl={self.se_dl_p99}, p99_ul={self.se_ul_p99}), "
            #f"eps_gbps={self.eps_gbps}, per_prb_khz={self.per_prb_khz}"
        #)

    # ----------------------------- Public API ------------------------------
    def calculate_score(self, window_data: Dict) -> Dict:
        ok, err = self.validate_window_data(window_data)
        if not ok:
            return {'score': 0.0, 'coverage': 0.0, 'status': 'ERROR',
                    'details': {'validation_error': err}}

        cell = window_data.get('cell_data', pd.DataFrame())
        ue   = window_data.get('ue_data',   pd.DataFrame())

        # ---------- A1: efficiency per PRB (aggregate tuples) ----------
        tuples = []
        a1_details = {}
        for side, thp_col, prb_col in [
            ('dl', self.thp_dl_col, self.prb_used_dl),
            ('ul', self.thp_ul_col, self.prb_used_ul),
        ]:
            t, d = self._a1_eff_tuple(cell, thp_col, prb_col, side)
            if t is not None:
                tuples.append(t)
            a1_details[side] = d

        # ---------- A2: spectral efficiency trip-wire (row-wise) ----------
        a2_series_cell, a2_details = self._a2_se_series(cell)

        # ---------- A3: context rule (row-wise) ----------
        a3_series_cell, a3_details = self._a3_context_series(cell)

        # Aggregate via standard APR/MPR
        apr, mpr, coverage, fails = self._apr_mpr(
            check_series_list=[a2_series_cell, a3_series_cell],
            check_tuples_list=tuples
        )

        return {
            'score': mpr,
            'apr': apr,
            'coverage': coverage,
            'details': {
                'A1_efficiency_bands': a1_details,
                'A2_spectral_efficiency': a2_details,
                'A3_context_rule': a3_details,
                'fail_counts': fails
            }
        }

    # ----------------------------- A1 helper ------------------------------
    def _a1_eff_tuple(self, df: pd.DataFrame, thp_col: str, prb_col: str, side: str) -> Tuple[Optional[Tuple[int,int]], Dict]:
        """
        Window-level tuple: median( (Thp_Gbps*1000) / PRB_Used ) ∈ [q25, q75]
        Returns: ((passed, total), details) or (None, details) if not applicable
        """
        det: Dict = {'side': side, 'status': 'NA'}
        if df.empty or thp_col not in df.columns or prb_col not in df.columns:
            det['reason'] = 'missing columns or empty df'
            return None, det

        thp_gbps = pd.to_numeric(df[thp_col], errors='coerce')
        prb_used = pd.to_numeric(df[prb_col], errors='coerce').clip(lower=0)
        s = (thp_gbps * 1000.0) / prb_used.replace(0, np.nan)   # Mbps/PRB
        s = s.replace([np.inf, -np.inf], np.nan).dropna()

        if s.size < 10:
            det.update({'reason': 'insufficient data', 'n': int(s.size)})
            return None, det

        med = float(np.median(s))
        band = self.a1_bands.get(side, None)
        det.update({'median_mbps_per_prb': med, 'band': band})

        if not band or any(k not in band for k in ('q25', 'q75')):
            det['reason'] = 'no baseline band'
            return None, det

        q25, q75 = float(band['q25']), float(band['q75'])
        passed = int(q25 <= med <= q75)
        total  = 1
        det.update({'q25': q25, 'q75': q75, 'passed': passed, 'total': total, 'n': int(s.size)})
        return (passed, total), det

    # ----------------------------- A2 helper ------------------------------
    def _a2_se_series(self, df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """
        Row-wise spectral efficiency trip-wire:
        SE = (Thp_gbps * 1e9) / (PRB_used * per_prb_Hz)
        Pass if SE <= min(band_cap, 1.2×p99) (when p99 available).
        Uses Band column if present, else extracts via PATTERNS['band_extractor'].
        Treat very tiny PRB_used (<3) as NA to avoid numerical jitter.
        """
        det: Dict = {}
        req = {self.thp_dl_col, self.thp_ul_col, self.prb_used_dl, self.prb_used_ul, self.cell_ent_col}
        if df.empty or not req.issubset(df.columns):
            return pd.Series([], dtype='float'), {'note': 'missing columns or empty df'}

        # Band per row (prefer precomputed column)
        if 'Band' in df.columns:
            band_series = df['Band'].astype(str)
        else:
            band_series = df[self.cell_ent_col].astype(str).str.extract(PATTERNS['band_extractor'])[0]

        # Per-band PRB width (kHz) and spectral-efficiency cap
        # If band missing → fallback to self.per_prb_khz, self.se_abs_cap_glob
        band_to_prb_khz = {}
        band_to_cap = {}
        for band, spec in (BAND_SPECS or {}).items():
            per_khz = spec.get('per_prb_khz')
            if per_khz is None:
                bw_mhz = spec.get('bandwidth_mhz'); prb_cnt = spec.get('prb_count')
                if bw_mhz and prb_cnt:
                    per_khz = (float(bw_mhz) * 1000.0) / float(prb_cnt)
            if per_khz:
                band_to_prb_khz[band] = float(per_khz)
            cap = spec.get('spectral_efficiency_max')
            if cap is not None:
                band_to_cap[band] = float(cap)

        perprb_khz = band_series.map(band_to_prb_khz).fillna(float(self.per_prb_khz)).astype(float)
        band_cap   = band_series.map(band_to_cap).fillna(float(self.se_abs_cap_glob)).astype(float)

        # p99-based soft cap (same for all rows; keep original behavior)
        p99_dl_cap = float('inf')
        if getattr(self, 'se_dl_p99', None):
            p99_dl_cap = 1.2 * float(self.se_dl_p99)
        p99_ul_cap = float('inf')
        if getattr(self, 'se_ul_p99', None):
            p99_ul_cap = 1.2 * float(self.se_ul_p99)

        # --- Downlink ---
        thp_dl   = pd.to_numeric(df[self.thp_dl_col], errors='coerce')              # Gbps
        used_dl  = pd.to_numeric(df[self.prb_used_dl], errors='coerce')             # PRBs (used)
        # very small PRB_used → NA (avoid jitter)
        valid_dl = used_dl.ge(3) & thp_dl.notna() & perprb_khz.notna()
        denom_hz_dl = used_dl * (perprb_khz * 1e3)
        se_dl = pd.Series(np.nan, index=df.index, dtype='float')
        se_dl.loc[valid_dl] = (thp_dl.loc[valid_dl] * 1e9) / denom_hz_dl.loc[valid_dl].replace(0, np.nan)
        se_dl = se_dl.replace([np.inf, -np.inf], np.nan)

        cap_row_dl = np.minimum(band_cap, p99_dl_cap)  # per-row: min(band cap, 1.2×p99)
        ok_dl = (se_dl <= cap_row_dl).astype('float')
        ok_dl[se_dl.isna() | cap_row_dl.isna()] = np.nan

        # --- Uplink ---
        thp_ul   = pd.to_numeric(df[self.thp_ul_col], errors='coerce')
        used_ul  = pd.to_numeric(df[self.prb_used_ul], errors='coerce')
        valid_ul = used_ul.ge(3) & thp_ul.notna() & perprb_khz.notna()
        denom_hz_ul = used_ul * (perprb_khz * 1e3)
        se_ul = pd.Series(np.nan, index=df.index, dtype='float')
        se_ul.loc[valid_ul] = (thp_ul.loc[valid_ul] * 1e9) / denom_hz_ul.loc[valid_ul].replace(0, np.nan)
        se_ul = se_ul.replace([np.inf, -np.inf], np.nan)

        cap_row_ul = np.minimum(band_cap, p99_ul_cap)
        ok_ul = (se_ul <= cap_row_ul).astype('float')
        ok_ul[se_ul.isna() | cap_row_ul.isna()] = np.nan

        # Combine + details
        series = pd.concat([ok_dl, ok_ul], ignore_index=True) if (len(ok_dl) or len(ok_ul)) else pd.Series([], dtype='float')
        det['dl'] = {'applicable': int(ok_dl.notna().sum()), 'passed': int((ok_dl == 1.0).sum())}
        det['ul'] = {'applicable': int(ok_ul.notna().sum()), 'passed': int((ok_ul == 1.0).sum())}
        return series, det


    # ----------------------------- A3 helper ------------------------------
    def _a3_context_series(self, df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """
        Row-wise: If Thp > eps (Gbps), then PRB_Used > 0  (both DL & UL)
        """
        det: Dict = {}
        if df.empty or any(c not in df.columns for c in [self.thp_dl_col, self.thp_ul_col, self.prb_used_dl, self.prb_used_ul]):
            return pd.Series([], dtype='float'), {'note': 'missing columns or empty df'}

        def per_side(thp_col, prb_col):
            thp = pd.to_numeric(df[thp_col], errors='coerce')
            prb = pd.to_numeric(df[prb_col], errors='coerce')
            # condition: either throughput is ~0 (<=eps) OR PRB>0
            ok = ((thp <= self.eps_gbps) | (prb > 0)).astype('float')
            ok[thp.isna() | prb.isna()] = np.nan
            return ok, {
                'eps_gbps': float(self.eps_gbps),
                'applicable': int(ok.notna().sum()),
                'passed': int((ok == 1.0).sum()),
            }

        s_dl, d_dl = per_side(self.thp_dl_col, self.prb_used_dl)
        s_ul, d_ul = per_side(self.thp_ul_col, self.prb_used_ul)

        series = pd.concat([s_dl, s_ul], ignore_index=True) if len(s_dl) or len(s_ul) else pd.Series([], dtype='float')
        det.update({'dl': d_dl, 'ul': d_ul})
        return series, det