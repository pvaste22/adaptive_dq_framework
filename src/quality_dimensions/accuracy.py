import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from common.constants import COLUMN_NAMES
from src.quality_dimensions.base_dimension import BaseDimension


class Accuracy(BaseDimension):
    """
    Dimension 5: ACCURACY
    Purpose: Verify that reported KPIs match recomputed reference identities and
             self-calibrated ratio bands learned in Phase-1.

    Components
    ----------
    AC1 Percent identities (row-wise):
        For each mapping: pct_col == (num/den)*100 within decimals-based tolerance.
        Decimals keyed by pct_col from dq_baseline['percent_identities'][pct_col]['decimals'].

    AC2 Ratio identities (aggregate):
        For each mapping: ratio = num/den; check window median within learned IQR
        from dq_baseline['ratio_identities'][name] -> {num, den, q25, q50, q75, n}.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(dimension_name='accuracy', *args, **kwargs)
        dq = self.get_dq_baseline() or {}

        # Baseline-driven identity specs (optional)
        # Shape:
        # percent_identities = {
        #   "SomePctCol": {"num": "ColA", "den": "ColB", "decimals": 1},
        #   ...
        # }
        self.percent_ids: Dict[str, Dict] = dq.get('percent_identities', {}) or {}

        # Shape:
        # ratio_identities = {
        #   "energy_power_ratio": {"num": "PEE.Energy_interval", "den": "PEE.AvgPower_kWh_per_row",
        #                          "q25": 0.9, "q50": 1.0, "q75": 1.1, "n": 1234},
        #   ...
        # }
        self.ratio_ids: Dict[str, Dict] = dq.get('ratio_identities', {}) or {}

        # Column name conveniences
        self.ts_col   = self.column_names.get('timestamp', 'timestamp')
        self.cell_col = self.column_names.get('cell', 'cell')
        self.ue_col   = self.column_names.get('ue', 'ue')

        # Cadence for Δt if needed (seconds)
        self.cadence_sec = float(getattr(self, 'cadence_sec', 60.0))

        self.logger.info(
            f"Accuracy initialized: {len(self.percent_ids)} percent identities, "
            f"{len(self.ratio_ids)} ratio identities."
        )

    # ----------------------------- Public API ------------------------------

    def calculate_score(self, window_data: Dict[str, pd.DataFrame]) -> Dict:
        cell = window_data.get('cell', pd.DataFrame())
        ue   = window_data.get('ue',   pd.DataFrame())

        # AC1: Percent identities (row-wise)
        ac1_series, ac1_details = self._ac1_percent_identities(cell, ue)

        # AC2: Ratio identities (aggregate, using IQR bands)
        ac2_tuples, ac2_details = self._ac2_ratio_identities(cell, ue)

        # Scoring: one combined row-wise series to avoid base axis=1 truncation
        series_all = [ac1_series]
        active_row = [s for s in series_all if (s is not None and s.size > 0 and s.notna().any())]
        combined = (pd.concat(active_row, axis=0, ignore_index=True).astype('float')
                    if active_row else pd.Series([], dtype='float'))

        apr, mpr, coverage, fails = self._apr_mpr(
            check_series_list=[combined],
            check_tuples_list=ac2_tuples
        )

        return {
            'score': mpr,
            'apr': apr,
            'mpr': mpr,
            'coverage': coverage,
            'details': {
                'AC1_percent_identities': ac1_details,
                'AC2_ratio_identities': ac2_details,
                'fail_counts': fails
            }
        }

    # --------------------------- Component AC1 ----------------------------

    def _ac1_percent_identities(self, cell: pd.DataFrame, ue: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """
        Generic percent identities:
          pct_col ≈ (num / den) * 100 within tolerance = half-ULP at learned decimals.
        We allow mapping entries for either table; if a column is missing, it is skipped.
        """
        details: Dict = {'checks': []}
        series_list: List[pd.Series] = []

        if not self.percent_ids:
            return pd.Series([], dtype='float'), {'note': 'no percent identities in baseline'}

        def _build_series(df: pd.DataFrame, name: str, spec: Dict) -> Optional[pd.Series]:
            pct_col = name
            num_col = spec.get('num')
            den_col = spec.get('den')
            d = int(spec.get('decimals', 2))

            if df.empty or not {pct_col, num_col, den_col}.issubset(df.columns):
                return None

            num = pd.to_numeric(df[num_col], errors='coerce')
            den = pd.to_numeric(df[den_col], errors='coerce')
            pct = pd.to_numeric(df[pct_col], errors='coerce')

            s = pd.Series(np.nan, index=df.index, dtype='float')
            valid = num.notna() & den.notna() & pct.notna() & (den > 0)

            if valid.any():
                calc = (num / den) * 100.0
                calc = calc.clip(lower=0.0, upper=100.0)
                tgt  = pct.clip(lower=0.0, upper=100.0)
                tol  = (10.0 ** (-d)) / 2.0 + 1e-9
                ok   = (calc - tgt).abs() <= tol
                s.loc[valid] = ok.loc[valid].astype('float')

            details['checks'].append({
                'pct_col': pct_col, 'num': num_col, 'den': den_col, 'decimals': d,
                'applicable': int(s.notna().sum()), 'passed': int((s == 1.0).sum())
            })
            return s

        # Try to build for both tables; many entries will apply only to cell
        for pct_col, spec in self.percent_ids.items():
            sc = _build_series(cell, pct_col, spec)
            if sc is not None:
                series_list.append(sc)
            su = _build_series(ue, pct_col, spec)
            if su is not None:
                series_list.append(su)

        series = (pd.concat(series_list, axis=0, ignore_index=True).astype('float')
                  if series_list else pd.Series([], dtype='float'))

        details['applicable'] = int(series.notna().sum())
        details['passed']     = int((series == 1.0).sum())
        return series, details

    # --------------------------- Component AC2 ----------------------------

    def _ac2_ratio_identities(self, cell: pd.DataFrame, ue: pd.DataFrame) -> Tuple[List[Tuple[int,int]], Dict]:
        """
        Generic ratio identities with learned IQR bands.
        For each mapping:
           ratio = num / den (per row; rows with den<=0 or NaNs skipped),
           take median over window; pass if median in [q25, q75].
        Returns: list of aggregate tuples [(passed, total), ...]
        """
        tuples: List[Tuple[int, int]] = []
        info: List[Dict] = []

        if not self.ratio_ids:
            return tuples, {'note': 'no ratio identities in baseline'}

        def _window_tuple(df: pd.DataFrame, name: str, spec: Dict) -> Optional[Tuple[Tuple[int,int], Dict]]:
            num_col = spec.get('num')
            den_col = spec.get('den')
            q25 = spec.get('q25'); q75 = spec.get('q75')
            if df.empty or not {num_col, den_col}.issubset(df.columns):
                return None

            num = pd.to_numeric(df[num_col], errors='coerce')
            den = pd.to_numeric(df[den_col], errors='coerce')

            valid = num.notna() & den.notna() & (den > 0)
            if not valid.any():
                return ((0, 0), {'name': name, 'note': 'no valid rows'})

            r = (num.loc[valid] / den.loc[valid]).replace([np.inf, -np.inf], np.nan).dropna()
            if r.empty:
                return ((0, 0), {'name': name, 'note': 'no ratio after filtering'})
            med = float(np.median(r))
            if (q25 is None) or (q75 is None):
                # No band → skip; accuracy component not applicable
                return ((0, 0), {'name': name, 'median': med, 'note': 'missing band'})

            passed = int((med >= float(q25)) and (med <= float(q75)))
            details = {'name': name, 'median': med, 'q25': float(q25), 'q75': float(q75), 'n_rows': int(valid.sum())}
            return ((passed, 1), details)

        # Evaluate on cell and UE where applicable
        for name, spec in self.ratio_ids.items():
            out_c = _window_tuple(cell, name, spec)
            if out_c:
                t, d = out_c; tuples.append(t); info.append(d)
            out_u = _window_tuple(ue,   name, spec)
            if out_u:
                t, d = out_u; tuples.append(t); info.append(d)

        details = {
            'checks': info,
            'applicable': int(sum(t[1] for t in tuples)),  # totals of denominators
            'passed': int(sum(t[0] for t in tuples))
        }
        return tuples, details
