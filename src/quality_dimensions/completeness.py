"""
Module: Quality Dimensions
Phase: 2
Author: Pranjal V
Created: 23/09/2025
Purpose: Completeness dimension calculator.
Implements checks C1-C4 for data completeness assessment.
"""
"""
Completeness dimension calculator.
Pure scoring implementation without judgments.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import sys
import json
import pickle
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from quality_dimensions.base_dimension import BaseDimension
from common.constants import (
    EXPECTED_ENTITIES,
    MEAS_INTERVAL_SEC,
    EXPECTED_PATTERNS,
    DATA_QUIRKS,
    PATHS,
    SCORING_LEVELS
)


class CompletenessDimension(BaseDimension):
    """Calculates completeness score for data windows."""
    
    def __init__(self, baselines_path: Optional[Path] = None):
        """Initialize completeness dimension."""
        super().__init__('completeness')
        
        # Load expected values from configuration (fallbacks only; C2 uses HoD baselines)
        self.expected_cells = EXPECTED_ENTITIES['cells']
        self.expected_ues = EXPECTED_ENTITIES['ues']
        self.measurement_interval = MEAS_INTERVAL_SEC
        
        # Expected patterns from data analysis (informational)
        self.cqi_expected_zero_rate = EXPECTED_PATTERNS.get('cqi_no_measurement_rate', 0.60)
        self.mimo_expected_zero_rate = EXPECTED_PATTERNS.get('mimo_zero_rate', 0.86)
        
        # Define mandatory fields
        self.mandatory_fields_cell = [
            self.timestamp_col,
            self.cell_entity_col,
            'RRU.PrbAvailDl',
            'RRU.PrbAvailUl'
        ]
        self.mandatory_fields_ue = [
            self.timestamp_col,
            self.ue_entity_col
        ]


        # Load Hour-of-Day baselines (temporal_templates) for entity coverage
        self.hod_baselines = self._load_hod_baselines(baselines_path)

        self.logger.info(
            f"Completeness dimension initialized: {self.expected_cells} cells, {self.expected_ues} UEs (fallbacks)."
        )
        self.logger.debug(
            f"Expected patterns - CQI Zero-as-no-report: {self.cqi_expected_zero_rate:.1%}, "
            f"MIMO zeros: {self.mimo_expected_zero_rate:.1%}"
        )

    
    def _load_hod_baselines(self, baselines_path: Optional[Path]) -> Dict:
        """Load Hour-of-Day baselines from temporal_templates artifact."""
        try:
             # Try using parent class method first
            templates = self.load_artifact_baseline('temporal_templates')
    
            if templates:
                self.logger.info("Successfully loaded HoD baselines using base class loader")
                return templates

            # Fallback to manual loading if needed
            if baselines_path and baselines_path.exists():
                try:
                    with open(baselines_path, 'rb') as f:
                        templates = pickle.load(f)
                        self.logger.info("Loaded HoD baselines from provided path")
                        return templates
                except Exception as e:
                    self.logger.error(f"Error loading from provided path: {e}")

            # Final fallback
            self.logger.warning("HoD baselines not found, using default values")
            return self._create_default_hod_baselines()
        except Exception as e:
            self.logger.error(f"Error loading HoD baselines: {e}")
            return self._create_default_hod_baselines()
    
    def _create_default_hod_baselines(self) -> Dict:
        """Create default HoD baselines if file not found."""
        return {
            'cells': {
                'hod_median': [self.expected_cells] * 24,
                'hod_iqr': [5.0] * 24
            },
            'ues': {
                'hod_median': [int(self.expected_ues * 0.7)] * 24,  # Assume 70% UE presence
                'hod_iqr': [10.0] * 24
            }
        }
    
    def calculate_score(self, window_data: Dict) -> Dict:
        """
        Calculate completeness score.
        
        Args:
            window_data: Window data dictionary
            
        Returns:
            Dictionary with score, coverage and measurement details
        """
        is_valid, error_msg = self.validate_window_data(window_data)
        if not is_valid:
            self.logger.error(f"Window validation failed: {error_msg}")
            return {
                'score': 0.0,
                'coverage': 0.0,
                'status': 'ERROR',
                'details': {'validation_error': error_msg}
            }

        cell_data = window_data.get('cell_data', pd.DataFrame())
        ue_data = window_data.get('ue_data', pd.DataFrame())
        window_id = window_data.get('metadata', {}).get('window_id', 'unknown')
        self.logger.debug(f"Processing window {window_id}: {len(cell_data)} cell records, {len(ue_data)} UE records")
        
        # Calculate individual completeness components 
        c1_pass, c1_total, c1_details = self._score_mandatory_fields(cell_data, ue_data)
        c2_pass, c2_total, c2_details = self._score_entity_presence(cell_data, ue_data)
        c3_pass, c3_total, c3_details = self._score_time_continuity(cell_data, ue_data)
        c4_pass, c4_total, c4_details = self._score_field_completeness(cell_data, ue_data)
        
        
        # Collect as pass/total arrays
        check_series = []
        fail_counts = {}
        for cname, p, t in [
            ("mandatory_fields", c1_pass, c1_total),
            ("entity_presence", c2_pass, c2_total),
            ("time_continuity", c3_pass, c3_total),
            ("field_completeness", c4_pass, c4_total)
        ]:
            if p is not None and t is not None and t > 0:
                s = pd.Series([1]*p + [0]*(t-p))
                check_series.append(s)
                fail_counts[cname] = t - p
            else:
                fail_counts[cname] = None

        # Aggregate with base class method
        apr, mpr, coverage = self._apr_mpr(check_series)

        return {
            'score': mpr,
            'apr': apr,
            'coverage': coverage,
            'details': {
                'mandatory_fields': c1_details,
                'entity_presence': c2_details,
                'time_continuity': c3_details,
                'field_completeness': c4_details,
                'fail_counts': fail_counts
            }
        }
    
    def _score_mandatory_fields(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame) -> Tuple[Optional[float], Dict]:
        """
        C1: Fraction of rows with all mandatory fields present (cells+UEs), 

        """
        details = {}
        total_rows = (len(cell_data) if not cell_data.empty else 0) + (len(ue_data) if not ue_data.empty else 0)
        if total_rows == 0:
            return None, {'note': 'no rows in window'}

        # Row-wise completeness: all mandatory fields present
        passed = 0
        if not cell_data.empty:
            present_mask = cell_data[self.mandatory_fields_cell].notna().all(axis=1)
            passed += int(present_mask.sum())
            details['cell_rows_with_all_mandatory'] = int(present_mask.sum())
            details['cell_total_rows'] = int(len(cell_data))

        if not ue_data.empty:
            present_mask = ue_data[self.mandatory_fields_ue].notna().all(axis=1)
            passed += int(present_mask.sum())
            details['ue_rows_with_all_mandatory'] = int(present_mask.sum())
            details['ue_total_rows'] = int(len(ue_data))

        fraction_complete_rows = passed / float(total_rows)
        details['fraction_rows_all_mandatory'] = float(fraction_complete_rows)

    
        self.logger.debug(f"C1: fraction={fraction_complete_rows:.3f} => score={float(fraction_complete_rows)}")
        return passed, total_rows, details
    
    def _score_entity_presence(self, cell_data: pd.DataFrame,
                              ue_data: pd.DataFrame) -> Tuple[Optional[float], Dict]:
        """
        C2: Entity coverage vs HoD baselines (cells & UEs).
             PASS if both meet pass thresholds, FAIL if either below soft thresholds, else SOFT.
        """
        details = {}

        # Determine window hour from timestamps (prefer earliest timestamp present)
        def _infer_hour(df_list):
            for df in df_list:
                if not df.empty and self.timestamp_col in df.columns:
                    ts = df[self.timestamp_col].dropna().iloc[0]
                    try:
                        if isinstance(ts, (int, float, np.integer, np.floating)):
                            return int(pd.to_datetime(ts, unit='s').hour)
                        return int(pd.to_datetime(ts).hour)
                    except Exception:
                        continue
            return None

        hour = _infer_hour([cell_data, ue_data])
        if hour is None:
            return None, None, {'note': 'could not infer hour for HoD baselines'}

        expected_cells = float(self.hod_baselines['cells']['hod_median'][hour])
        expected_ues   = float(self.hod_baselines['ues']['hod_median'][hour])
        details['expected_cells_hod'] = expected_cells
        details['expected_ues_hod'] = expected_ues
        details['hour'] = hour

        actual_cells = 0.0
        if not cell_data.empty and self.timestamp_col in cell_data.columns and self.cell_entity_col in cell_data.columns:
            cells_per_ts = cell_data.groupby(self.timestamp_col)[self.cell_entity_col].nunique()
            actual_cells = float(cells_per_ts.mean()) if len(cells_per_ts) else 0.0

        actual_ues = 0.0
        if not ue_data.empty and self.timestamp_col in ue_data.columns and self.ue_entity_col in ue_data.columns:
            ues_per_ts = ue_data.groupby(self.timestamp_col)[self.ue_entity_col].nunique()
            actual_ues = float(ues_per_ts.mean()) if len(ues_per_ts) else 0.0

        ratio_cells = (actual_cells / expected_cells) if expected_cells > 0 else 0.0
        ratio_ues   = (actual_ues / expected_ues) if expected_ues > 0 else 0.0

        details['actual_cells_avg'] = actual_cells
        details['actual_ues_avg'] = actual_ues
        details['ratio_cells'] = ratio_cells
        details['ratio_ues'] = ratio_ues

        # Convert to pass/total counts
        passed = int(actual_cells + actual_ues)
        total = int(expected_cells + expected_ues)

        return passed, total, details
    
    
    def _score_time_continuity(self, cell_data: pd.DataFrame,
                              ue_data: pd.DataFrame) -> Tuple[Optional[float], Dict]:
        """
        C3: Temporal coverage by unique timestamp count in the window.
        """
        details = {}

        # Union of unique timestamps
        ts_union = set()
        if not cell_data.empty and self.timestamp_col in cell_data.columns:
            ts_union.update(pd.Series(cell_data[self.timestamp_col]).dropna().unique().tolist())
        if not ue_data.empty and self.timestamp_col in ue_data.columns:
            ts_union.update(pd.Series(ue_data[self.timestamp_col]).dropna().unique().tolist())

        ts_count = len(ts_union)
        details['unique_timestamps'] = ts_count
        if ts_count == 0:
            return None, None, details

        dq = self.get_dq_baseline()
        cadence = float(dq.get('cadence_sec', 60.0))
        # res = float(dq.get('ts_resolution_sec', 1.0))  # if needed later

        ts_sorted = sorted(pd.to_datetime(list(ts_union)))
        win_span = (ts_sorted[-1] - ts_sorted[0]).total_seconds() if ts_count > 1 else 0.0
        expected = int(round(win_span / cadence)) + 1 if win_span > 0 else ts_count
        expected = max(expected, 1)

        details['expected_timestamps'] = int(expected)

        # Pass/total counts
        passed = ts_count
        total = expected

        return passed, total, details

    def _score_field_completeness(self, cell_data: pd.DataFrame,
                                 ue_data: pd.DataFrame) -> Tuple[Optional[float], Dict]:
        """
        C4: Field-level completeness with pattern awareness.
            Count fraction of non-mandatory, eligible fields having < null_threshold nulls.
        """
        details = {}

        if DATA_QUIRKS.get('tb_counters_unreliable'):
            skip_fields_exact = {'TB.TotNbrDl', 'TB.TotNbrUl'}
        else:
            skip_fields_exact = set()

        def eligible_fields(df: pd.DataFrame, mandatory: List[str]) -> List[str]:
            if df.empty:
                return []
            return [c for c in df.columns if c not in mandatory and c not in skip_fields_exact]

        cell_fields = eligible_fields(cell_data, self.mandatory_fields_cell)
        ue_fields   = eligible_fields(ue_data, self.mandatory_fields_ue)
        fields = [(cell_data, c) for c in cell_fields] + [(ue_data, c) for c in ue_fields]

        total_fields = len(fields)
        if total_fields == 0:
            return None, None, {'note': 'no eligible fields'}

        non_null_count = 0
        for df, col in fields:
            if len(df) > 0:
                non_null_count += int(df[col].notna().sum())

        total_cells = sum(len(df) for df, _ in fields)
        details['fields_evaluated'] = int(total_fields)
        details['non_null_cells'] = non_null_count
        details['total_cells'] = total_cells

        passed = non_null_count
        total = total_cells

        return passed, total, details