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
    SCORING_LEVELS,
    COMPLETENESS_THRESHOLDS
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

        self.thresholds = self.dimension_thresholds if self.dimension_thresholds else COMPLETENESS_THRESHOLDS
        if not self.thresholds:
            self.logger.warning("No completeness thresholds found, using emergency defaults")
        self.thresholds = {
            "C1_mandatory_fields": {"pass": 0.95, "soft": 0.80},
            "C2_entity_coverage": {
                "cells": {"pass": 0.95, "soft": 0.85},
                "ues": {"pass": 0.80, "soft": 0.60}
            },
            "C3_temporal_coverage": {"pass": 5, "soft": 4},
            "C4_field_completeness": {"pass": 0.80, "soft": 0.60, "null_threshold": 0.20}
        }
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
        
        # Calculate individual completeness components (each returns PASS=1.0 / SOFT=0.5 / FAIL=0.0 / None)
        c1_score, c1_details = self._score_mandatory_fields(cell_data, ue_data)
        c2_score, c2_details = self._score_entity_presence(cell_data, ue_data)
        c3_score, c3_details = self._score_time_continuity(cell_data, ue_data)
        c4_score, c4_details = self._score_field_completeness(cell_data, ue_data)
        
        component_scores = [c1_score, c2_score, c3_score, c4_score]
        valid_scores = [s for s in component_scores if s is not None]
        #overall_score = float(np.mean(valid_scores)) if valid_scores else None
        apr = sum([1 for s in valid_scores if s == 1.0]) / len(valid_scores) if valid_scores else None
        mpr = float(np.mean(valid_scores)) if valid_scores else None
        coverage = len(valid_scores) / 4.0

        self.logger.debug(
            f"Component scores - C1:{c1_score} C2:{c2_score} C3:{c3_score} C4:{c4_score} "
            f"Overall:{mpr} Coverage:{coverage:.2f}"
        )
    
        return {
            'score': mpr,
            'coverage': coverage,
            'details': {
                'mandatory_fields_score': c1_score,
                'entity_presence_score': c2_score,
                'time_continuity_score': c3_score,
                'field_completeness_score': c4_score,
                'measurements': {
                    'mandatory_fields': c1_details,
                    'entity_presence': c2_details,
                    'time_continuity': c3_details,
                    'field_completeness': c4_details
                }
            }
        }
    
    def _score_mandatory_fields(self, cell_data: pd.DataFrame, 
                               ue_data: pd.DataFrame) -> Tuple[Optional[float], Dict]:
        """
        C1: Fraction of rows with all mandatory fields present (cells+UEs), 
            mapped to PASS/SOFT/FAIL via thresholds.
        """
        details = {}
        total_rows = (len(cell_data) if not cell_data.empty else 0) + (len(ue_data) if not ue_data.empty else 0)
        if total_rows == 0:
            return None, {'note': 'no rows in window'}

        # Row-wise completeness: all mandatory fields present
        cell_ok = 0
        if not cell_data.empty:
            present_mask = cell_data[self.mandatory_fields_cell].notna().all(axis=1)
            cell_ok = int(present_mask.sum())
            details['cell_rows_with_all_mandatory'] = cell_ok
            details['cell_total_rows'] = int(len(cell_data))

        ue_ok = 0
        if not ue_data.empty:
            present_mask = ue_data[self.mandatory_fields_ue].notna().all(axis=1)
            ue_ok = int(present_mask.sum())
            details['ue_rows_with_all_mandatory'] = ue_ok
            details['ue_total_rows'] = int(len(ue_data))

        fraction_complete_rows = (cell_ok + ue_ok) / float(total_rows)
        details['fraction_rows_all_mandatory'] = float(fraction_complete_rows)

        t = self.thresholds.get("C1_mandatory_fields", {})
        p, s = t.get("pass", 0.95), t.get("soft", 0.80)
        score = 1.0 if fraction_complete_rows >= p else (0.5 if fraction_complete_rows >= s else 0.0)
        self.logger.debug(f"C1: fraction={fraction_complete_rows:.3f} => score={score}")
        return score, details
    
    def _score_entity_presence(self, cell_data: pd.DataFrame,
                              ue_data: pd.DataFrame) -> Tuple[Optional[float], Dict]:
        """
        C2: Entity coverage vs HoD baselines (cells & UEs).
             PASS if both meet pass thresholds, FAIL if either below soft thresholds, else SOFT.
        """
        details = {}

        # Determine window hour from timestamps (prefer earliest timestamp present)
        def _infer_hour(df_list: List[pd.DataFrame]) -> Optional[int]:
            for df in df_list:
                if not df.empty and self.timestamp_col in df.columns:
                    ts = df[self.timestamp_col].dropna().iloc[0]
                    try:
                        # Handle epoch seconds or ISO-8601
                        if isinstance(ts, (int, float, np.integer, np.floating)):
                            return int(pd.to_datetime(ts, unit='s').hour)
                        return int(pd.to_datetime(ts).hour)
                    except Exception:
                        continue
            return None

        hour = _infer_hour([cell_data, ue_data])
        if hour is None:
            return None, {'note': 'could not infer hour for HoD baselines'}

        # Expected entities (HoD medians)
        expected_cells = float(self.hod_baselines['cells']['hod_median'][hour])
        expected_ues   = float(self.hod_baselines['ues']['hod_median'][hour])
        details['expected_cells_hod'] = expected_cells
        details['expected_ues_hod'] = expected_ues
        details['hour'] = hour

        # Actual average unique entities reporting per timestamp
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

        t = self.thresholds.get("C2_entity_coverage", {})
        c_pass, c_soft = t.get("cells", {}).get("pass", 0.95), t.get("cells", {}).get("soft", 0.85)
        u_pass, u_soft = t.get("ues",   {}).get("pass", 0.80), t.get("ues",   {}).get("soft", 0.60)

        if ratio_cells < c_soft or ratio_ues < u_soft:
            score = 0.0
        elif ratio_cells >= c_pass and ratio_ues >= u_pass:
            score = 1.0
        else:
            score = 0.5

        self.logger.debug(
            f"C2: hour={hour} cells {actual_cells:.2f}/{expected_cells:.0f} ({ratio_cells:.2f}), "
            f"UEs {actual_ues:.2f}/{expected_ues:.0f} ({ratio_ues:.2f}) => score={score}"
        )
        return score, details
    
    def _score_time_continuity(self, cell_data: pd.DataFrame,
                              ue_data: pd.DataFrame) -> Tuple[Optional[float], Dict]:
        """
        C3: Temporal coverage by unique timestamp count in the window.
            PASS if count >= 5, SOFT if >= 4, else FAIL. Thresholds from config.
        """
        details = {}
        # Union of unique timestamps across both datasets
        ts_union = set()
        if not cell_data.empty and self.timestamp_col in cell_data.columns:
            ts_union.update(pd.Series(cell_data[self.timestamp_col]).dropna().unique().tolist())
        if not ue_data.empty and self.timestamp_col in ue_data.columns:
            ts_union.update(pd.Series(ue_data[self.timestamp_col]).dropna().unique().tolist())

        ts_count = len(ts_union)
        details['unique_timestamps'] = ts_count

        if ts_count == 0:
            return None, details  # N/A if no timestamps at all

        t = self.thresholds.get("C3_temporal_coverage", {})
        p, s = int(t.get("pass", 5)), int(t.get("soft", 4))
        score = 1.0 if ts_count >= p else (0.5 if ts_count >= s else 0.0)
        self.logger.debug(f"C3: ts_count={ts_count} => score={score}")
        return score, details
    
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
            cols = []
            for c in df.columns:
                if c in mandatory:
                    continue
                if c in skip_fields_exact:
                    continue
                cols.append(c)
            return cols

        cell_fields = eligible_fields(cell_data, self.mandatory_fields_cell)
        ue_fields   = eligible_fields(ue_data, self.mandatory_fields_ue)
        fields = [(cell_data, c) for c in cell_fields] + [(ue_data, c) for c in ue_fields]

        total_fields = len(fields)
        if total_fields == 0:
            return None, {'note': 'no eligible fields to evaluate'}

        t = self.thresholds.get("C4_field_completeness", {})
        null_thr = float(t.get("null_threshold", 0.20))
        passed = 0

        for df, col in fields:
            # Field passes if < null_threshold nulls in the window
            null_ratio = float(df[col].isna().mean()) if len(df) > 0 else 0.0
            if null_ratio < null_thr:
                passed += 1

        fraction_fields_passing = passed / float(total_fields)
        details['fields_evaluated'] = int(total_fields)
        details['fields_passing'] = int(passed)
        details['fraction_fields_passing'] = float(fraction_fields_passing)

        p, s = float(t.get("pass", 0.80)), float(t.get("soft", 0.60))
        score = 1.0 if fraction_fields_passing >= p else (0.5 if fraction_fields_passing >= s else 0.0)
        self.logger.debug(
            f"C4: {passed}/{total_fields} fields passing (<{null_thr:.2f} nulls) => "
            f"fraction={fraction_fields_passing:.3f} score={score}"
        )
        return score, details
