# scripts/phase1_data_preparation.py
"""
Phase 1 Main Script: Data Preparation and Baseline Creation
Orchestrates the complete Phase 1 implementation following the detailed plan.

Key Steps:
1. Load and preprocess VIAVI dataset
2. Apply critical unit conversions 
3. Generate 5-minute windows
4. Create all baseline artifacts (13 types)
5. Prepare for Phase 2 quality scoring
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings
import pickle
from pathlib import Path



from data_processing.data_loader import DataLoader
from data_processing.unit_converter import UnitConverter
from data_processing.window_generator import WindowGenerator
from common.logger import get_phase1_logger
from common.constants import PATHS, VERSIONING, COLUMN_NAMES, EXPECTED_ENTITIES


class Phase1Orchestrator:
    """
    Orchestrates Phase 1: Data Preparation and Baseline Creation
    
    Implements the complete workflow from raw VIAVI data to processed windows
    and baseline artifacts ready for Phase 2 quality scoring.
    """
    
    def __init__(self):
        """Initialize Phase 1 orchestrator using configuration from constants."""
        self.logger = self._setup_logging()
        
        # Phase 1 artifacts to create
        self.artifacts_to_create = [
            'statistical_baselines',
            'historical_pdfs', 
            'pattern_baselines',
            'temporal_templates',
            'field_ranges',
            'metadata_config',
            'sample_windows',
            'divergence_array_placeholder',
            'dq_baseline'
        ]
        
        # Initialize processors - no config needed, they use constants
        self.data_loader = DataLoader()
        self.unit_converter = UnitConverter()
        self.window_generator = WindowGenerator()
        
        # Artifacts storage
        self.artifacts = {}
        self.windows = []
        
        # Use paths from constants
        self.paths = PATHS
        self.versioning = VERSIONING
        self.column_names = COLUMN_NAMES
        self.expected_entities = EXPECTED_ENTITIES
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for Phase 1"""
        return get_phase1_logger('phase1_orchestrator')
    
    def run_phase1_complete(self, resume_from: Optional[str] = None) -> Dict:
        """
        Run complete Phase 1 workflow.

         Args:
        resume_from: Optional step name to resume from ('step1', 'step2', etc.)
        
        Returns:
            Dictionary with execution summary and artifact locations
        """

        self.logger.info("=" * 60)
        self.logger.info("PHASE 1: DATA PREPARATION AND BASELINE CREATION")
        self.logger.info("=" * 60)
        
        start_time = datetime.now()
        results = {}
        
        try:

            # Step 1: Load and preprocess data

            self.logger.info("Step 1: Loading and preprocessing data...")
            cell_data, ue_data = self._step1_load_and_preprocess()
            results['step1'] = {'status': 'completed', 'records': {'cells': len(cell_data), 'ues': len(ue_data)}}
            
            # Step 2: Apply unit conversions
            self.logger.info("Step 2: Applying critical unit conversions...")
            cell_data, ue_data = self._step2_apply_conversions(cell_data, ue_data)
            results['step2'] = {'status': 'completed', 'conversions_applied': True}
            
            # Step 3: Generate windows
            self.logger.info("Step 3: Generating 5-minute windows...")
            windows = self._step3_generate_windows(cell_data, ue_data)
            results['step3'] = {'status': 'completed', 'windows_generated': windows['window_count']}
            
            # Step 4: Create baselines
            self.logger.info("Step 4: Creating baseline artifacts...")
            artifacts = self._step4_create_baselines(cell_data, ue_data, windows)
            results['step4'] = {'status': 'completed', 'artifacts_created': len(artifacts)}
            
            # Step 5: Save artifacts and prepare for Phase 2
            self.logger.info("Step 5: Saving artifacts and preparing for Phase 2...")
            artifact_paths = self._step5_save_and_prepare(cell_data, ue_data, windows, artifacts)
            results['step5'] = {'status': 'completed', 'artifact_paths': artifact_paths}
            
            # Step 6: Generate comprehensive report
            self.logger.info("Step 6: Generating Phase 1 completion report...")
            report = self._step6_generate_report(results, artifacts, windows)
            results['step6'] = {'status': 'completed', 'report_path': report['report_path']}
            
            execution_time = (datetime.now() - start_time).total_seconds()
            results['execution_summary'] = {
                'total_time_seconds': execution_time,
                'status': 'SUCCESS',
                'ready_for_phase2': True
            }
            
            self.logger.info("="*60)
            self.logger.info(f"PHASE 1 COMPLETED SUCCESSFULLY in {execution_time:.1f} seconds")
            self.logger.info("Ready for Phase 2: Quality Scoring and Model Training")
            self.logger.info("="*60)
            
        except Exception as e:
            self.logger.error(f"Phase 1 failed: {str(e)}", exc_info=True)
            results['execution_summary'] = {
                'status': 'FAILED',
                'error': str(e),
                'ready_for_phase2': False
            }
            raise
        
        return results

    def _step1_load_and_preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Step 1: Load and preprocess raw VIAVI data"""
        
        # Load data directly - DataLoader uses constants
        cell_data, ue_data = self.data_loader.load_data()
        
        # Get data summary
        data_summary = self.data_loader.get_data_summary(cell_data, ue_data)
        
        # Log summary
        self.logger.info(f"Loaded {data_summary['cell_data']['total_records']} cell records")
        self.logger.info(f"Loaded {data_summary['ue_data']['total_records']} UE records")
        
        if data_summary['cell_data']['date_range'][0]:
            self.logger.info(f"Date range: {data_summary['cell_data']['date_range'][0]} to {data_summary['cell_data']['date_range'][1]}")
        
        
        return cell_data, ue_data
    
    def _step2_apply_conversions(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Step 2: Apply critical unit conversions"""
        
        # Apply comprehensive unit conversions
        cell_data_converted, ue_data_converted = self.unit_converter.standardize_units_comprehensive(
            cell_data, ue_data
        )
        
        # Get conversion summary (not validation)
        conversion_summary = self.unit_converter.get_conversion_summary(
            cell_data_converted, ue_data_converted
        )
        
        # Log what was done
        self.logger.info("Unit conversion summary:")
        self.logger.info(f"  Cell data conversions: {conversion_summary['cell_data']['conversions_applied']}")
        self.logger.info(f"  New columns created: {conversion_summary['cell_data']['new_columns_created']}")
        self.logger.info(f"  Flags added: {conversion_summary['cell_data']['flags_added']}")
        
        # Store conversion summary for later reference
        self.artifacts['conversion_summary'] = conversion_summary
        
        return cell_data_converted, ue_data_converted
    
    def _step3_generate_windows(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame) -> Dict:
        """Step 3: Generate 5-minute windows and save to disk."""
    
        # Generate windows
        windows = self.window_generator.generate_windows(cell_data, ue_data)
    
        # Get statistics before saving
        window_stats = self.window_generator.get_window_statistics(windows)
        
        # Log window generation results
        self.logger.info(f"Generated {window_stats['window_count']} valid windows")
        self.logger.info(f"Time span: {window_stats['time_span']['total_hours']:.1f} hours")
        self.logger.info(f"Mean completeness: {window_stats['completeness']['mean']:.3f}")
        #self.logger.info(f"Mean records per window: {window_stats['record_counts']['mean']:.1f}")
        
        #Save all windows to disk
        window_dir = self.paths['training']/'windows'
        self.window_generator.save_all_windows(windows, window_dir)

        # Store window statistics
        self.artifacts['window_statistics'] = window_stats
        
        # Store windows for later use
        #self.windows = windows
        
        return window_stats
    
    def _step4_create_baselines(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame, window_stats: Dict) -> Dict:
        """Step 4: Create all baseline artifacts"""

        # Load sample windows from disk for KS tests
        windows_dir = self.paths['training'] / 'windows'
        window_index_file = windows_dir / 'window_index.json'
    
        sample_windows = []
        if window_index_file.exists():
            with open(window_index_file, 'r') as f:
                index = json.load(f)
                total_windows = index['total_count']
            
                # Sample 100 windows evenly
                n_samples = min(100, total_windows)
                sample_indices = np.linspace(0, total_windows - 1, n_samples, dtype=int)
            
                for idx in sample_indices:                    
                    window_info = index['windows'][idx]
                    abs_path = Path(window_info['path'])
                    window_info['path'] = abs_path.relative_to(self.paths['processed'])
                    sample_windows.append(window_info)
        
        artifacts = {}
        
        # 1. Statistical Baselines
        self.logger.info("Creating statistical baselines...")
        artifacts['statistical_baselines'] = self._create_statistical_baselines(cell_data, ue_data)
        
        # 2. Historical PDFs
        self.logger.info("Creating histogram-based PDFs...")
        artifacts['historical_pdfs'] = self._create_histogram_pdfs(cell_data, ue_data)
        
        # 3. Correlation Matrix
        #self.logger.info("Creating correlation matrix...")
        #artifacts['correlation_matrix'] = self._create_correlation_matrix(cell_data, ue_data)

        #3. Field ranges baseline
        self.logger.info("Creating field ranges...")
        artifacts['field_ranges'] = self._create_field_ranges(cell_data, ue_data)
        
        # 4. Pattern Baselines
        self.logger.info("Creating pattern baselines...")
        artifacts['pattern_baselines'] = self._create_pattern_baselines(cell_data, ue_data)
        
        # 5. Temporal Templates
        self.logger.info("Creating temporal templates...")
        artifacts['temporal_templates'] = self._create_temporal_templates(cell_data, ue_data)
        
        # 6. Quality Thresholds
        #self.logger.info("Creating quality thresholds...")
        #artifacts['quality_thresholds'] = self._create_quality_thresholds()
        
        # 7. Sample Windows
        self.logger.info("Sampling historical windows...")
        #artifacts['sample_windows'] = self._create_sample_windows(windows)
        artifacts['sample_windows'] = self._create_sample_windows(sample_windows)

        
        # 8. Metadata Configuration
        self.logger.info("Creating metadata configuration...")
        artifacts['metadata_config'] = self._create_metadata_config()
        
        # 9. Placeholder for divergence array
        artifacts['divergence_array'] = {
            'status': 'placeholder',
            'note': 'Will be populated during quality scoring phase',
            'expected_size': '1000-2000 JSD values'
        }

        # 10. DQ Baseline
        self.logger.info("Creating dq_baseline...")
        artifacts['dq_baseline'] = self._create_dq_baseline(cell_data, ue_data)
        
        self.logger.info(f"Created {len([k for k, v in artifacts.items() if v.get('status') != 'placeholder'])} baseline artifacts")
        
        return artifacts
    
    def _create_statistical_baselines(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame) -> Dict:
        """Create statistical baselines for z-score calculations"""
        
        # Key metrics for statistical analysis
        cell_metrics = [
            'RRU.PrbUsedDl', 'RRU.PrbUsedUl', 'DRB.UEThpDl', 'DRB.UEThpUl','RRU.PrbAvailDl', 'RRU.PrbAvailUl','RRU.PrbTotDl_abs', 'RRU.PrbTotUl_abs', 'PEE.Energy_interval',
            'RRC.ConnMean', 'RRC.ConnMax', 'PEE.AvgPower','QosFlow.TotPdcpPduVolumeDl', 'QosFlow.TotPdcpPduVolumeUl', 'CARR.AverageLayersDl', 'RRU.MaxLayerDlMimo'
        ]
        
        ue_metrics = [
            'DRB.UEThpDl', 'DRB.UEThpUl', 'DRB.UECqiDl', 'DRB.UECqiUl',
            'RRU.PrbUsedDl', 'RRU.PrbUsedUl'
        ]
        
        baselines = {
            'cell_metrics': {},
            'ue_metrics': {},
            'metadata': {
                'creation_time': datetime.now().isoformat(),
                'sample_size_cells': len(cell_data),
                'sample_size_ues': len(ue_data)
            }
        }
        
        # Calculate cell metric statistics
        for metric in cell_metrics:
            if metric in cell_data.columns:
                data = cell_data[metric].dropna()
                if len(data) > 0:
                    baselines['cell_metrics'][metric] = {
                        'mean': float(data.mean()),
                        'std': float(data.std()),
                        'mad': float((data - data.median()).abs().median()),
                        'p5': float(data.quantile(0.05)),
                        'p25': float(data.quantile(0.25)),
                        'p50': float(data.quantile(0.50)),
                        'p75': float(data.quantile(0.75)),
                        'p95': float(data.quantile(0.95)),
                        'min': float(data.min()),
                        'max': float(data.max()),
                        'count': len(data)
                    }
        
        # Calculate UE metric statistics
        for metric in ue_metrics:
            if metric in ue_data.columns:
                data = ue_data[metric].dropna()
                if len(data) > 0:
                    baselines['ue_metrics'][metric] = {
                        'mean': float(data.mean()),
                        'std': float(data.std()),
                        'mad': float((data - data.median()).abs().median()),
                        'p5': float(data.quantile(0.05)),
                        'p25': float(data.quantile(0.25)),
                        'p50': float(data.quantile(0.50)),
                        'p75': float(data.quantile(0.75)),
                        'p95': float(data.quantile(0.95)),
                        'min': float(data.min()),
                        'max': float(data.max()),
                        'count': len(data)
                    }
        
        return baselines
    
    def _create_histogram_pdfs(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame) -> Dict:
        """Create histogram-based PDFs for drift detection"""
        
        cell_metrics = ['RRU.PrbUsedDl', 'RRU.PrbUsedUl', 'DRB.UEThpDl', 'DRB.UEThpUl', 'RRC.ConnMean', 'PEE.AvgPower']
        ue_metrics = ['DRB.UEThpDl', 'DRB.UEThpUl', 'DRB.UECqiDl', 'DRB.UECqiUl', 'RRU.PrbUsedDl', 'RRU.PrbUsedUl']
        
        pdfs = {
            'cell_distributions': {},
            'ue_distributions': {},
            'metadata': {
                'creation_time': datetime.now().isoformat(),
                'bin_count': 50,
                'total_metrics': len(cell_metrics) + len(ue_metrics)
            }
        }
        
        # Create cell metric PDFs
        for metric in cell_metrics:
            if metric in cell_data.columns:
                data = cell_data[metric].dropna()
                if len(data) > 0:
                    # Determine bin strategy
                    if metric == 'RRC.ConnMean':
                        bins = np.arange(data.min(), data.max() + 2) - 0.5
                    else:
                        bins = np.linspace(data.min(), data.max(), 51)
                    
                    hist, bin_edges = np.histogram(data, bins=bins)
                    pdf = hist / hist.sum()
                    
                    pdfs['cell_distributions'][metric] = {
                        'frequencies': pdf.tolist(),
                        'bin_edges': bin_edges.tolist(),
                        'total_samples': len(data),
                        'min_value': float(data.min()),
                        'max_value': float(data.max()),
                        'is_discrete': metric == 'RRC.ConnMean'
                    }
        
        # Create UE metric PDFs
        for metric in ue_metrics:
            if metric in ue_data.columns:
                data = ue_data[metric].dropna()
                if len(data) > 0:
                    # Determine bin strategy
                    if metric in ['DRB.UECqiDl', 'DRB.UECqiUl']:
                        bins = np.arange(0, 17) - 0.5
                    else:
                        bins = np.linspace(data.min(), data.max(), 51)
                    
                    hist, bin_edges = np.histogram(data, bins=bins)
                    pdf = hist / hist.sum()
                    
                    pdfs['ue_distributions'][metric] = {
                        'frequencies': pdf.tolist(),
                        'bin_edges': bin_edges.tolist(),
                        'total_samples': len(data),
                        'min_value': float(data.min()),
                        'max_value': float(data.max()),
                        'is_discrete': metric in ['DRB.UECqiDl', 'DRB.UECqiUl']
                    }
        
        return pdfs

    def _create_correlation_matrix(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame) -> Dict:
        """Create correlation matrix for consistency checking"""
        from common.constants import EXPECTED_CORRELATIONS
    
        correlations = {
            'expected_correlations': {},
            'actual_correlations': {},
            'metadata': {
                'creation_time': datetime.now().isoformat(),
                'sample_size_cells': len(cell_data),
                'tolerance': 0.30  # Default tolerance
            }
        }
    
        # Use expected correlations from config
        for metric1, metric2, expected_corr, tolerance in EXPECTED_CORRELATIONS:
            if metric1 in cell_data.columns and metric2 in cell_data.columns:
                valid_data = cell_data[[metric1, metric2]].dropna()
            
                if len(valid_data) > 10:
                    actual_corr = valid_data[metric1].corr(valid_data[metric2])
                
                    key = f'{metric1}_vs_{metric2}'
                    correlations['expected_correlations'][key] = expected_corr
                    correlations['actual_correlations'][key] = float(actual_corr)
                    correlations['tolerance'] = tolerance
    
        return correlations
    
    def _create_pattern_baselines(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame) -> Dict:
        """Create pattern baselines (GLOBAL learned bands: IQR across cells)"""
        patterns = {
            'learned_bands': {'global': {}},
            'metadata': {
                'creation_time': datetime.now().isoformat(),
                'note': 'Self-calibrated bands from training data (no config tolerances)'
            }
        }

        cell_col = self.column_names.get('cell_entity', 'Viavi.Cell.Name') # keep existing
        # --- 1) MIMO zero-rate (cell data)
        mimo_band = {"median": None, "q25": None, "q75": None, "n": 0}
        mimo_pref_col = 'CARR.AverageLayersDl'
        mimo_fallback_col = 'RRU.MaxLayerDlMimo'
        use_col = None
        if {mimo_pref_col, cell_col}.issubset(cell_data.columns):
            use_col = mimo_pref_col
        elif {mimo_fallback_col, cell_col}.issubset(cell_data.columns):
            use_col = mimo_fallback_col
        if use_col and not cell_data.empty:
            cdf = cell_data[[cell_col, use_col]].copy()
            cdf[use_col] = pd.to_numeric(cdf[use_col], errors='coerce')
            g = cdf.groupby(cell_col, dropna=True)[use_col]
            per_cell_rate = g.apply(lambda s: float((s == 0).mean()) if s.notna().any() else np.nan).dropna()
            if not per_cell_rate.empty:
                q25, q75 = per_cell_rate.quantile([0.25, 0.75])
                mimo_band = {
                    "median": float(per_cell_rate.median()),
                    "q25": float(q25),
                    "q75": float(q75),
                    "n": int(per_cell_rate.size)
                }
        patterns['learned_bands']['global']['mimo_zero_layers'] = mimo_band

        # --- 2) CQI no-measurement rate (UE data) — per-timestamp global fraction → IQR
        cqi_band = {"median": None, "q25": None, "q75": None, "n": 0}
        ts_col = self.column_names.get('timestamp', 'timestamp')
        if {'DRB.UECqiDl', ts_col}.issubset(ue_data.columns) and not ue_data.empty:
            udf = ue_data[[ts_col, 'DRB.UECqiDl']].copy()
            udf[ts_col] = pd.to_datetime(udf[ts_col], errors='coerce')
            udf['is_cqi0'] = pd.to_numeric(udf['DRB.UECqiDl'], errors='coerce').eq(0)
            # per-timestamp fraction of CQI==0 across all UEs
            frac_by_ts = (udf.groupby(ts_col, dropna=True)['is_cqi0']
                             .mean()
                             .dropna())
            if not frac_by_ts.empty:
                q25, q75 = frac_by_ts.quantile([0.25, 0.75])
                cqi_band = {
                    "median": float(frac_by_ts.median()),
                    "q25": float(q25),
                    "q75": float(q75),
                    "n": int(frac_by_ts.size)
                }
        patterns['learned_bands']['global']['cqi_no_report'] = cqi_band
        # (Optional) UL counterpart for future use — same logic on 'DRB.UECqiUl'
        if {'DRB.UECqiUl', ts_col}.issubset(ue_data.columns) and not ue_data.empty:
            udf_ul = ue_data[[ts_col, 'DRB.UECqiUl']].copy()
            udf_ul[ts_col] = pd.to_datetime(udf_ul[ts_col], errors='coerce')
            udf_ul['is_cqi0'] = pd.to_numeric(udf_ul['DRB.UECqiUl'], errors='coerce').eq(0)
            frac_by_ts_ul = (udf_ul.groupby(ts_col, dropna=True)['is_cqi0'].mean().dropna())
            if not frac_by_ts_ul.empty:
                q25, q75 = frac_by_ts_ul.quantile([0.25, 0.75])
                patterns['learned_bands']['global']['cqi_ul_no_report'] = {
                    "median": float(frac_by_ts_ul.median()),
                    "q25": float(q25),
                    "q75": float(q75),
                    "n": int(frac_by_ts_ul.size)
                }             

        # --- 3) UL-DL correlation (cell data) — Pearson r per cell
        corr_band = {"median": None, "q25": None, "q75": None, "n": 0}
        if {'DRB.UEThpDl', 'DRB.UEThpUl', cell_col}.issubset(cell_data.columns) and not cell_data.empty:
            tdf = cell_data[[cell_col, 'DRB.UEThpDl', 'DRB.UEThpUl']].copy()
            tdf['DRB.UEThpDl'] = pd.to_numeric(tdf['DRB.UEThpDl'], errors='coerce')
            tdf['DRB.UEThpUl'] = pd.to_numeric(tdf['DRB.UEThpUl'], errors='coerce')
            def _safe_corr(grp):
                dl = grp['DRB.UEThpDl']; ul = grp['DRB.UEThpUl']
                mask = dl.notna() & ul.notna()
                if mask.sum() >= 30:
                    try: return float(dl[mask].corr(ul[mask]))
                    except Exception: return np.nan
                return np.nan
            per_cell_corr = tdf.groupby(cell_col, dropna=True).apply(_safe_corr).dropna()
            if not per_cell_corr.empty:
                q25, q75 = per_cell_corr.quantile([0.25, 0.75])
                corr_band = {
                    "median": float(per_cell_corr.median()),
                    "q25": float(q25),
                    "q75": float(q75),
                    "n": int(per_cell_corr.size)
                }
        patterns['learned_bands']['global']['ul_dl_correlation'] = corr_band

        return patterns
    

    def _create_field_ranges(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame) -> Dict:
        """Create field valid ranges for validation check"""
    
        field_ranges = {
            'cell_metrics': {},
            'ue_metrics': {},
            'metadata': {
                'creation_time': datetime.now().isoformat(),
                'source': 'VIAVI training dataset observations',
                'purpose': 'Valid ranges for validation check'
            }
        }
    
        # Cell metrics to track ranges
        cell_metrics = [
            'DRB.UEThpDl', 'DRB.UEThpUl',
            'RRU.PrbUsedDl', 'RRU.PrbUsedUl',
            'RRU.PrbAvailDl', 'RRU.PrbAvailUl',
            'RRU.PrbTotDl', 'RRU.PrbTotUl',
            'PEE.AvgPower', 'PEE.Energy_interval',
            'RRC.ConnMean', 'RRC.ConnMax',
            'CARR.AverageLayersDl', 'RRU.MaxLayerDlMimo'
        ]
    
        # UE metrics to track ranges
        ue_metrics = [
            'DRB.UEThpDl', 'DRB.UEThpUl',
            'DRB.UECqiDl', 'DRB.UECqiUl',
            'RRU.PrbUsedDl', 'RRU.PrbUsedUl'
        ]
    
        # Calculate ranges for cell metrics
        for metric in cell_metrics:
            if metric in cell_data.columns:
                data = cell_data[metric].dropna()
                if len(data) > 0:
                    field_ranges['cell_metrics'][metric] = {
                        'min': float(data.min()),
                        'max': float(data.max()),
                        'p01': float(data.quantile(0.01)),
                        'p99': float(data.quantile(0.99)),
                        'median': float(data.median()),
                        'count': len(data)
                    }
    
        # Calculate ranges for UE metrics
        for metric in ue_metrics:
            if metric in ue_data.columns:
                data = ue_data[metric].dropna()
                if len(data) > 0:
                    field_ranges['ue_metrics'][metric] = {
                        'min': float(data.min()),
                        'max': float(data.max()),
                        'p01': float(data.quantile(0.01)),
                        'p99': float(data.quantile(0.99)),
                        'median': float(data.median()),
                        'count': len(data)
                    }
    
        self.logger.info(f"Created field ranges for {len(field_ranges['cell_metrics'])} cell metrics and "
                    f"{len(field_ranges['ue_metrics'])} UE metrics")
    
        return field_ranges

    def _create_temporal_templates(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame) -> Dict:
        """Create temporal pattern templates"""
        if 'timestamp' not in cell_data.columns:
            return {'error': 'No timestamp column found'}
    
        # Extract temporal features
        cell_data_temp = cell_data.copy()
        ue_data_temp = ue_data.copy()
        cell_data_temp['hour'] = cell_data_temp['timestamp'].dt.hour
        ue_data_temp['hour'] = ue_data_temp['timestamp'].dt.hour
    
        templates = {
            'cells': {
                'hod_median': [],
                'hod_iqr': []
            },
            'ues': {
                'hod_median': [],
                'hod_iqr': []
            },
            'metadata': {
                'creation_time': datetime.now().isoformat(),
                'sample_period': f"{cell_data['timestamp'].min()} to {cell_data['timestamp'].max()}",
                'purpose': 'Hour-of-day entity count baselines for C2 check'
            }
        }   
    
        # Calculate cell baselines per hour
        cell_entity_col = self.column_names.get('cell_entity', 'Viavi.Cell.Name')
        for hour in range(24):
            hour_data = cell_data_temp[cell_data_temp['hour'] == hour]
            if not hour_data.empty:
                # Count unique cells per timestamp in this hour
                counts = hour_data.groupby('timestamp')[cell_entity_col].nunique().values
                if len(counts) > 0:
                    q1, median, q3 = np.percentile(counts, [25, 50, 75])
                    templates['cells']['hod_median'].append(float(median))
                    templates['cells']['hod_iqr'].append(float(q3 - q1))
                else:
                    # Use expected value if no data
                    templates['cells']['hod_median'].append(float(self.expected_entities['cells']))
                    templates['cells']['hod_iqr'].append(5.0)
            else:
                templates['cells']['hod_median'].append(float(self.expected_entities['cells']))
                templates['cells']['hod_iqr'].append(5.0)
    
        # Calculate UE baselines per hour
        ue_entity_col = self.column_names.get('ue_entity', 'Viavi.UE.Name')
        for hour in range(24):
            hour_data = ue_data_temp[ue_data_temp['hour'] == hour]
            if not hour_data.empty:
                counts = hour_data.groupby('timestamp')[ue_entity_col].nunique().values
                if len(counts) > 0:
                    q1, median, q3 = np.percentile(counts, [25, 50, 75])
                    templates['ues']['hod_median'].append(float(median))
                    templates['ues']['hod_iqr'].append(float(q3 - q1))
                else:
                    templates['ues']['hod_median'].append(30.0)  # Default
                    templates['ues']['hod_iqr'].append(10.0)
            else:
                templates['ues']['hod_median'].append(30.0)
                templates['ues']['hod_iqr'].append(10.0)
    
        self.logger.info(f"Created HoD entity baselines: Cells [{min(templates['cells']['hod_median']):.0f}-{max(templates['cells']['hod_median']):.0f}], "
                        f"UEs [{min(templates['ues']['hod_median']):.0f}-{max(templates['ues']['hod_median']):.0f}]")
    
        return templates
    
 
    
    def _create_sample_windows(self, sample_window_info: List[Dict]) -> Dict:
        """Create sample windows for KS tests"""
        
        #n_samples = min(100, len(windows))
        
        if not sample_window_info:
            return {'error': 'No windows available for sampling'}
        
        # Sample evenly across time
        #indices = np.linspace(0, len(windows) - 1, n_samples, dtype=int)
        #sampled_windows = [windows[i] for i in indices]
        
        sample_data = {
            'sample_windows': sample_window_info,
            'metadata': {
                'creation_time': datetime.now().isoformat(),
                'total_available_windows': len(sample_window_info),
                'samples_taken': len(sample_window_info),
                'sampling_method': 'evenly_distributed'
            }
        }
        
        """for window in sampled_windows:
            sample_data['sample_windows'].append({
                'window_id': window['window_id'],
                'start_time': window['start_time'].isoformat(),
                'end_time': window['end_time'].isoformat(),
                'metadata': window['metadata']
            })"""
        
        return sample_data
    
    
    def _create_dq_baseline(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame) -> Dict:
        """
        Build self-calibrated DQ baseline for threshold-free checks.
        Returns a small dict saved as a separate artifact.
        Keys:
        - cadence_sec: Detected measurement interval (seconds)
        - ts_resolution_sec: Timestamp precision (seconds)
        - prb_pct_decimals: {'RRU.PrbTotDl': int, 'RRU.PrbTotUl': int}
        - energy_power_band: {'ratio_q1': float, 'ratio_q3': float, 'n': int}
        - ue_cell_thp_ratio: {'dl': {q25, q50, q75, n}, 'ul': {q25, q50, q75, n}}
        - ue_cell_prb_ratio: {'dl': {q25, q50, q75, n}, 'ul': {q25, q50, q75, n}} 
        """
        dq: Dict[str, Any] = {}

        # --- 1) Cadence & timestamp resolution (union of cell + UE timestamps)
        ts_col = self.column_names.get('timestamp', 'timestamp')
        ts_all: List[pd.Timestamp] = []
        if ts_col in cell_data.columns and not cell_data.empty:
            ts_all.extend(pd.to_datetime(cell_data[ts_col], errors='coerce').tolist())
        if ts_col in ue_data.columns and not ue_data.empty:
            ts_all.extend(pd.to_datetime(ue_data[ts_col], errors='coerce').tolist())

        ts_u = pd.Series(ts_all, dtype="datetime64[ns]").dropna().drop_duplicates().sort_values()
        if len(ts_u) >= 2:
            diffs = ts_u.diff().dropna().dt.total_seconds()
            if not diffs.empty:
                cadence = float(np.median(diffs))
                pos = diffs[diffs > 0]
                res = float(pos.min()) if not pos.empty else cadence
                dq['cadence_sec'] = cadence
                dq['ts_resolution_sec'] = res
        # Fallbacks if missing
        if 'cadence_sec' not in dq:
            dq['cadence_sec'] = float(getattr(self, 'measurement_interval', 60))
        if 'ts_resolution_sec' not in dq:
            dq['ts_resolution_sec'] = dq['cadence_sec']

        # --- 2) PRB% decimals (auto-detect): use MODE of decimal lengths
        prb_dec: Dict[str, int] = {}
        for prb_col in ['RRU.PrbTotDl', 'RRU.PrbTotUl']:
            if prb_col in cell_data.columns:
                s = pd.to_numeric(cell_data[prb_col], errors='coerce').dropna()
                if not s.empty:
                    # count decimal places by string repr
                    dec_len = s.map(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
                    # mode (most common) is more robust than max
                    prb_dec[prb_col] = int(dec_len.mode().iloc[0])
        if prb_dec:
            dq['prb_pct_decimals'] = prb_dec

        # --- 3) Energy vs Power ratio IQR (global; exclude resets/NaNs)
        # Expect: Energy_interval[kWh] / (AvgPower[kW] * Δt_hours)
        energy_col_int = 'PEE.Energy_interval'
        power_col = self.column_names.get('avg_power', 'PEE.AvgPower')
        entity_col = self.column_names.get('cell_entity', 'Viavi.Cell.Name')

        ratio = pd.Series(dtype='float64')
        required_cols = [entity_col, ts_col, energy_col_int, power_col]
        optional_cols = []
        if 'PEE.Energy_reset' in cell_data.columns:
            optional_cols.append('PEE.Energy_reset')
        if 'PEE.Energy_imputed' in cell_data.columns:
            optional_cols.append('PEE.Energy_imputed')

        select_cols = required_cols + optional_cols
        
        if set(required_cols).issubset(cell_data.columns) and not cell_data.empty:
            en = cell_data[select_cols].copy() 
            en[ts_col] = pd.to_datetime(en[ts_col], errors='coerce')
            en[energy_col_int] = pd.to_numeric(en[energy_col_int], errors='coerce')
            en[power_col] = pd.to_numeric(en[power_col], errors='coerce')

            en = en.sort_values([entity_col, ts_col])
            # per-entity Δt in hours with fallback to configured interval
            en['__dt_h'] = en.groupby(entity_col)[ts_col].diff().dt.total_seconds().div(3600.0)
            fallback_h = float(getattr(self, 'measurement_interval', 60)) / 3600.0
            en['__dt_h'] = en['__dt_h'].fillna(fallback_h).clip(lower=1e-9)

            # exclude resets/NaN intervals if flag exists
            base_mask = en[energy_col_int].notna() & en[power_col].notna() & en['__dt_h'].notna()
            extras = []
            if 'PEE.Energy_reset' in en.columns:
                #en = en.join(cell_data[['PEE.Energy_reset']], how='left')
                #en = cell_data[[entity_col, ts_col, energy_col_int, power_col, 'PEE.Energy_reset']].copy()
                #extras.append(en['PEE.Energy_reset'] != True)
                base_mask = base_mask & (en['PEE.Energy_reset'] != True)
            if 'PEE.Energy_imputed' in en.columns:
                #en = en.join(cell_data[['PEE.Energy_imputed']], how='left')
                #en = cell_data[[entity_col, ts_col, energy_col_int, power_col, 'PEE.Energy_imputed']].copy()
                #extras.append(en['PEE.Energy_imputed'] != True)
                base_mask = base_mask & (en['PEE.Energy_imputed'] != True)

            #base = en[energy_col_int].notna() & en[power_col].notna() & en['__dt_h'].notna()
            #mask_valid = base if not extras else base & np.logical_and.reduce(extras)
            mask_valid = base_mask
            #else:
                #mask_valid = en[energy_col_int].notna() & en[power_col].notna()

            # Power in W -> kW
            denom = (en.loc[mask_valid, power_col] / 1000.0) * en.loc[mask_valid, '__dt_h']
            #with np.errstate(divide='ignore', invalid='ignore'):
                #ratio = (en.loc[mask_valid, energy_col_int] / denom).replace([np.inf, -np.inf], np.nan)
            nonzero_mask = denom > 0
            if nonzero_mask.any():
            #idx = denom.index[pos]
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = (en.loc[mask_valid, energy_col_int][nonzero_mask] / denom[nonzero_mask]).replace([np.inf, -np.inf], np.nan)
                ratio = ratio.dropna()

        if not ratio.empty and len(ratio) >= 30:
            q1, q3 = np.quantile(ratio, [0.25, 0.75])
            dq['energy_power_band'] = {'ratio_q1': float(q1), 'ratio_q3': float(q3), 'n': int(ratio.size)}
        else:
            dq['energy_power_band'] = {'ratio_q1': None, 'ratio_q3': None, 'n': int(ratio.size if not ratio.empty else 0)}
        
        # --- 4) UE–Cell throughput ratio bands ---
        dq['ue_cell_thp_ratio'] = None  

        thp_dl = 'DRB.UEThpDl'
        thp_ul = 'DRB.UEThpUl'
        need_cell = {ts_col, thp_dl, thp_ul}.issubset(cell_data.columns)
        need_ue   = {ts_col, thp_dl, thp_ul}.issubset(ue_data.columns)

        if not (need_cell and need_ue):
            self.logger.info("Reconciliation bands: missing throughput cols; skipping.")
        else:
            c = cell_data[[ts_col, thp_dl, thp_ul]].copy()
            u = ue_data[[ts_col, thp_dl, thp_ul]].copy()
            c[ts_col] = pd.to_datetime(c[ts_col], errors='coerce'); c = c.dropna(subset=[ts_col])
            u[ts_col] = pd.to_datetime(u[ts_col], errors='coerce'); u = u.dropna(subset=[ts_col])

            cg = c.groupby(ts_col, as_index=True).agg({thp_dl:'sum', thp_ul:'sum'}).rename(
                columns={thp_dl:'cell_dl', thp_ul:'cell_ul'}
            )
            ug = u.groupby(ts_col, as_index=True).agg({thp_dl:'sum', thp_ul:'sum'}).rename(
                columns={thp_dl:'ue_dl', thp_ul:'ue_ul'}
            )
            g = cg.join(ug, how='inner')

            # positive denominators only
            g = g[(g['cell_dl'] > 0) | (g['cell_ul'] > 0)]
            if g.empty:
                self.logger.info("Reconciliation bands: no overlapping ts with positive denominators.")
            else:
                ratio_dl = (g.loc[g['cell_dl'] > 0, 'ue_dl'] / g.loc[g['cell_dl'] > 0, 'cell_dl']).replace([np.inf, -np.inf], np.nan).dropna()
                ratio_ul = (g.loc[g['cell_ul'] > 0, 'ue_ul'] / g.loc[g['cell_ul'] > 0, 'cell_ul']).replace([np.inf, -np.inf], np.nan).dropna()

                min_n = int(getattr(self, 'reconciliation_min_samples', 30))  # make config-driven if you want
                def iqr_band(s: pd.Series):
                    if s.size < min_n: return None
                    q25, q50, q75 = np.percentile(s, [25, 50, 75])
                    return {'q25': float(q25), 'q50': float(q50), 'q75': float(q75), 'n': int(s.size)}

                dl_band = iqr_band(ratio_dl)
                ul_band = iqr_band(ratio_ul)
                if dl_band or ul_band:
                    dq['ue_cell_thp_ratio'] = {'dl': dl_band, 'ul': ul_band}
                else:
                    self.logger.info(f"Reconciliation bands: insufficient samples (DL={ratio_dl.size}, UL={ratio_ul.size}, need {min_n})")

        # --- 5) UE–Cell PRB used ratio bands ---
        dq['ue_cell_prb_ratio'] = None  

        prb_dl = 'RRU.PrbUsedDl'
        prb_ul = 'RRU.PrbUsedUl'
        need_cell = {ts_col, prb_dl, prb_ul}.issubset(cell_data.columns)
        need_ue   = {ts_col, prb_dl, prb_ul}.issubset(ue_data.columns)

        if not (need_cell and need_ue):
            self.logger.info("PRB reconciliation: missing PRB columns; skipping.")
        else:
            c = cell_data[[ts_col, prb_dl, prb_ul]].copy()
            u = ue_data[[ts_col, prb_dl, prb_ul]].copy()
            c[ts_col] = pd.to_datetime(c[ts_col], errors='coerce'); c = c.dropna(subset=[ts_col])
            u[ts_col] = pd.to_datetime(u[ts_col], errors='coerce'); u = u.dropna(subset=[ts_col])

            cg = c.groupby(ts_col, as_index=True).agg({prb_dl:'sum', prb_ul:'sum'}).rename(
            columns={prb_dl:'cell_dl', prb_ul:'cell_ul'}
            )
            ug = u.groupby(ts_col, as_index=True).agg({prb_dl:'sum', prb_ul:'sum'}).rename(
            columns={prb_dl:'ue_dl', prb_ul:'ue_ul'}
            )
            g = cg.join(ug, how='inner')

            # positive denominators only
            g = g[(g['cell_dl'] > 0) | (g['cell_ul'] > 0)]
            if g.empty:
                self.logger.info("PRB reconciliation: no overlapping ts with positive denominators.")
            else:
                ratio_dl = (g.loc[g['cell_dl'] > 0, 'ue_dl'] / g.loc[g['cell_dl'] > 0, 'cell_dl']).replace([np.inf,-np.inf], np.nan).dropna()
                ratio_ul = (g.loc[g['cell_ul'] > 0, 'ue_ul'] / g.loc[g['cell_ul'] > 0, 'cell_ul']).replace([np.inf,-np.inf], np.nan).dropna()

                min_n = int(getattr(self, 'reconciliation_min_samples', 30))
                def iqr_band(s: pd.Series):
                    if s.size < min_n: return None
                    q25, q50, q75 = np.percentile(s, [25, 50, 75])
                    return {'q25': float(q25), 'q50': float(q50), 'q75': float(q75), 'n': int(s.size)}

                dl_band = iqr_band(ratio_dl)
                ul_band = iqr_band(ratio_ul)
                if dl_band or ul_band:
                    dq['ue_cell_prb_ratio'] = {'dl': dl_band, 'ul': ul_band}
                else:
                    self.logger.info(f"PRB reconciliation: insufficient samples (<{min_n}).")                    

        # Metadata
        dq['metadata'] = {
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'source': 'Phase 1 training (self-calibrated)'
        }
        return dq

    
    
    def _create_metadata_config(self) -> Dict:
        """Create metadata configuration"""
        
        from common.constants import EXPECTED_ENTITIES, BAND_LIMITS
        
        metadata = {
            'dataset_info': {
                'source': 'VIAVI O-RAN Simulated Dataset',
                'expected_entities': EXPECTED_ENTITIES,
                'band_configuration': BAND_LIMITS,
                'measurement_interval_seconds': 60
            },
            'known_issues': {
                'prb_tot_unit_mismatch': 'PrbTot reported as percentage, needs conversion',
                'energy_cumulative': 'Energy reported as cumulative, needs differencing',
                'tb_counters_unreliable': 'TB counters should be ignored',
                'qos_flow_semantics': 'QosFlow uses 1-second semantics despite 60s intervals',
                'cqi_zero_meaning': 'CQI=0 means no measurement, converted to NaN'
            },
            'expected_patterns': {
                'mimo_zero_rate': 0.86,
                'cqi_no_measurement_rate': 0.60,
                'ul_dl_symmetry': 0.87
            },
            'phase1_completion': {
                'timestamp': datetime.now().isoformat(),
                'artifacts_created': len(self.artifacts_to_create),
                'ready_for_phase2': True
            }
        }
        
        return metadata
    
    def _step5_save_and_prepare(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame, 
                               windows: List[Dict], artifacts: Dict) -> Dict:
        """Step 5: Save artifacts and prepare for Phase 2"""
        
        artifact_paths = {}
        
        # Save processed data
        processed_dir = self.paths['training']
        processed_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime(self.versioning['timestamp_format'])
        artifact_formats = {
            'statistical_baselines': 'pkl',
            'historical_pdfs': 'pkl',  # or 'npz' if converted to numpy
            #'correlation_matrix': 'json',
            'pattern_baselines': 'pkl',
            'temporal_templates': 'pkl',
            'field_ranges': 'json',
            #'quality_thresholds': 'json',
            'metadata_config': 'json',
            'sample_windows': 'json',
            'dq_baseline': 'json'
        }
        
        cell_data.to_parquet(processed_dir / 'cell_data_processed.parquet')
        ue_data.to_parquet(processed_dir / 'ue_data_processed.parquet')
        artifact_paths['processed_data'] = {
            'cell_data': str(processed_dir / 'cell_data_processed.parquet'),
            'ue_data': str(processed_dir / 'ue_data_processed.parquet')
        }
        
        # Save artifacts with versioning
        artifacts_dir = self.paths['artifacts']
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime(self.versioning['timestamp_format'])
        
        for artifact_name, artifact_data in artifacts.items():
            if artifact_data.get('status') == 'placeholder':
                continue
                
            artifact_subdir = artifacts_dir / artifact_name
            artifact_subdir.mkdir(parents=True, exist_ok=True)
            file_format = artifact_formats.get(artifact_name, 'pkl')
            artifact_file = artifact_subdir / f'{artifact_name}_v1_{timestamp}.{file_format}'
            if file_format == 'json':
                with open(artifact_file, 'w') as f:
                    json.dump(artifact_data, f, indent=2, default=str)
            else: 
                with open(artifact_file, 'wb') as f:
                    pickle.dump(artifact_data, f)

            artifact_paths[artifact_name] = str(artifact_file)
        
        # Save sample windows
        windows_dir = self.paths['historical_windows']
        windows_dir.mkdir(parents=True, exist_ok=True)
        
        #sample_windows = windows[:50] if len(windows) > 50 else windows
        #windows_file = windows_dir / f'sample_windows_{timestamp}.pkl'
        #with open(windows_file, 'wb') as f:
            #pickle.dump(sample_windows, f)
        #artifact_paths['sample_windows_file'] = str(windows_file)
        
        # Create artifact registry
        registry = {
            'creation_timestamp': timestamp,
            'phase1_artifacts': artifact_paths,
            'total_windows_generated': len(windows),
            'artifacts_ready_for_phase2': [k for k in artifacts.keys() if artifacts[k].get('status') != 'placeholder'],
            'next_steps': [
                'Run Phase 2: Quality scoring and model training',
                'Generate quality labels using dimension scores',
                'Train XGBoost model with features'
            ]
        }
        
        registry_file = artifacts_dir / f'artifact_registry_{timestamp}.json'
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2, default=str)
        artifact_paths['registry'] = str(registry_file)
        
        self.logger.info(f"Saved {len(artifact_paths)} artifact files")
        return artifact_paths
    
    def _step6_generate_report(self, results: Dict, artifacts: Dict, windows: List[Dict]) -> Dict:
        """Step 6: Generate comprehensive Phase 1 report"""
        
        report = {
            'phase1_summary': {
                'execution_timestamp': datetime.now().isoformat(),
                'total_steps_completed': len([k for k, v in results.items() if k.startswith('step') and v.get('status') == 'completed']),
                'ready_for_phase2': True
            },
            'data_processing_results': {
                'records_processed': {
                    'cells': results.get('step1', {}).get('records', {}).get('cells', 0),
                    'ues': results.get('step1', {}).get('records', {}).get('ues', 0)
                },
                'conversions_applied': results.get('step2', {}).get('conversions_applied', False),
                'windows_generated': results.get('step3', {}).get('windows_generated', 0)
            },
            'artifacts_created': {
                'total_artifacts': len(artifacts),
                'ready_artifacts': len([k for k, v in artifacts.items() if v.get('status') != 'placeholder']),
                'artifact_types': list(artifacts.keys())
            },
            'quality_indicators': {},
            'next_phase_requirements': [
                'Quality dimension calculators implementation',
                'PCA consolidation model training',
                'XGBoost model training with features',
                'Drift detection mechanism setup'
            ]
        }
        
        # Add quality indicators from artifacts
        if 'statistical_baselines' in artifacts:
            stats = artifacts['statistical_baselines']
            report['quality_indicators']['data_quality'] = {
                'cell_metrics_available': len(stats.get('cell_metrics', {})),
                'ue_metrics_available': len(stats.get('ue_metrics', {}))
            }
        
        if 'pattern_baselines' in artifacts:
            patterns = artifacts['pattern_baselines'].get('known_patterns', {})
            report['quality_indicators']['pattern_validation'] = {
                pattern_name: {
                    'actual': pattern_data.get('value', 0),
                    'expected': pattern_data.get('expected', 0),
                    'deviation': abs(pattern_data.get('value', 0) - pattern_data.get('expected', 0))
                }
                for pattern_name, pattern_data in patterns.items()
            }
        
        # Save report
        report_dir = Path('reports')
        report_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = report_dir / f'phase1_completion_report_{timestamp}.json'
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        report['report_path'] = str(report_file)
        
        return {'report': report, 'report_path': str(report_file)}


def main():
    """Main execution function"""
    
    print("\n" + "="*60)
    print("STARTING PHASE 1: DATA PREPARATION AND BASELINE CREATION")
    print("="*60)
    
    # Initialize and run Phase 1
    orchestrator = Phase1Orchestrator()
    
    try:
        results = orchestrator.run_phase1_complete()
        
        if results['execution_summary']['status'] == 'SUCCESS':
            print("\n" + "="*60)
            print("PHASE 1 COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"Windows generated: {results['step3']['windows_generated']}")
            print(f"Artifacts created: {results['step4']['artifacts_created']}")
            print(f"Execution time: {results['execution_summary']['total_time_seconds']:.1f} seconds")
            print(f"Report saved to: {results['step6']['report_path']}")
            print("\nReady for Phase 2: Quality Scoring and Model Training")
            print("="*60)
            
            return 0
        else:
            print(f"\nPhase 1 failed: {results['execution_summary']['error']}")
            return 1
            
    except Exception as e:
        print(f"\nPhase 1 execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())