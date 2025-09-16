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
from typing import Dict, List, Optional, Tuple
import warnings
import pickle
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from data_processing.data_loader import DataLoader
from data_processing.unit_converter import UnitConverter
from data_processing.window_generator import WindowGenerator
from common.logger import get_phase1_logger
from common.constants import PATHS, VERSIONING

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
            'correlation_matrix',
            'pattern_baselines',
            'temporal_templates',
            'quality_thresholds',
            'metadata_config',
            'sample_windows',
            'divergence_array_placeholder'
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
            #check if resuming
            if resume_from:
                self.logger.info(f"Attempting to resume from {resume_from}")
                checkpoint_data = self._load_checkpoint(resume_from)
                if checkpoint_data:
                    self.logger.info(f"Resumed from checkpoint: {resume_from}")
                # Set appropriate variables based on resume point

            # Step 1: Load and preprocess data
            if not resume_from or resume_from == 'step1':
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
                results['step3'] = {'status': 'completed', 'windows_generated': len(windows)}
            
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
    def _save_checkpoint(self, step_name: str, data: any):
        """Save checkpoint after successful step completion."""
        checkpoint_dir = self.paths['artifacts'] / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
        checkpoint = {
            'step': step_name,
            'timestamp': datetime.now().isoformat(),
            'data_type': type(data).__name__
        }
    
        # Save based on data type
        if isinstance(data, pd.DataFrame):
            data.to_parquet(checkpoint_dir / f'{step_name}_data.parquet')
            checkpoint['file'] = f'{step_name}_data.parquet'
        elif isinstance(data, tuple) and all(isinstance(d, pd.DataFrame) for d in data):
            ata[0].to_parquet(checkpoint_dir / f'{step_name}_cell.parquet')
            data[1].to_parquet(checkpoint_dir / f'{step_name}_ue.parquet')
            checkpoint['files'] = [f'{step_name}_cell.parquet', f'{step_name}_ue.parquet']
        else:
            with open(checkpoint_dir / f'{step_name}_data.pkl', 'wb') as f:
                pickle.dump(data, f)
            checkpoint['file'] = f'{step_name}_data.pkl'
    
        # Save checkpoint metadata
        with open(checkpoint_dir / 'checkpoint_status.json', 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
        self.logger.info(f"Checkpoint saved for {step_name}")

    def _load_checkpoint(self, step_name: str):
        """Load checkpoint if it exists."""
        checkpoint_dir = self.paths['artifacts'] / 'checkpoints'
        status_file = checkpoint_dir / 'checkpoint_status.json'
    
        if not status_file.exists():
            return None
    
        with open(status_file, 'r') as f:
            checkpoint = json.load(f)
    
        if checkpoint['step'] != step_name:
            return None
    
        # Load based on file type
        if 'files' in checkpoint:  # Tuple of DataFrames
            cell_data = pd.read_parquet(checkpoint_dir / checkpoint['files'][0])
            ue_data = pd.read_parquet(checkpoint_dir / checkpoint['files'][1])
            return (cell_data, ue_data)
        elif checkpoint['file'].endswith('.parquet'):
            return pd.read_parquet(checkpoint_dir / checkpoint['file'])
        else:
            with open(checkpoint_dir / checkpoint['file'], 'rb') as f:
                return pickle.load(f)
        

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
        
        # Log data alignment
        alignment = data_summary['data_alignment']
        if alignment['timestamp_overlap']:
            overlap_pct = alignment['timestamp_overlap'].get('overlap_percentage', 0)
            self.logger.info(f"Timestamp overlap: {overlap_pct:.1f}%")
        
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

        # Save checkpoint with just statistics (not the windows themselves)
        self._save_checkpoint('step3', window_stats)

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
                    sample_windows.append(window_info)
        
        artifacts = {}
        
        # 1. Statistical Baselines
        self.logger.info("Creating statistical baselines...")
        artifacts['statistical_baselines'] = self._create_statistical_baselines(cell_data, ue_data)
        
        # 2. Historical PDFs
        self.logger.info("Creating histogram-based PDFs...")
        artifacts['historical_pdfs'] = self._create_histogram_pdfs(cell_data, ue_data)
        
        # 3. Correlation Matrix
        self.logger.info("Creating correlation matrix...")
        artifacts['correlation_matrix'] = self._create_correlation_matrix(cell_data, ue_data)
        
        # 4. Pattern Baselines
        self.logger.info("Creating pattern baselines...")
        artifacts['pattern_baselines'] = self._create_pattern_baselines(cell_data, ue_data)
        
        # 5. Temporal Templates
        self.logger.info("Creating temporal templates...")
        artifacts['temporal_templates'] = self._create_temporal_templates(cell_data, ue_data)
        
        # 6. Quality Thresholds
        self.logger.info("Creating quality thresholds...")
        artifacts['quality_thresholds'] = self._create_quality_thresholds()
        
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
        
        self.logger.info(f"Created {len([k for k, v in artifacts.items() if v.get('status') != 'placeholder'])} baseline artifacts")
        
        return artifacts
    
    def _create_statistical_baselines(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame) -> Dict:
        """Create statistical baselines for z-score calculations"""
        
        # Key metrics for statistical analysis
        cell_metrics = [
            'RRU.PrbUsedDl', 'RRU.PrbUsedUl', 'DRB.UEThpDl', 'DRB.UEThpUl','RRU.PrbAvailDl', 'RRU.PrbAvailUl','RRU.PrbTotDl_abs', 'RRU.PrbTotUl_abs', 'PEE.Energy_interval',
            'RRC.ConnMean', 'RRC.ConnMax', 'PEE.AvgPower','QosFlow.TotPdcpPduVolumeDl', 'QosFlow.TotPdcpPduVolumeUl', 'CARR.AverageLayersDl'
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
        """Create pattern baselines for accuracy validation"""
        from common.constants import EXPECTED_PATTERNS
    
        patterns = {
            'known_patterns': {},
            'metadata': {
                'creation_time': datetime.now().isoformat(),
                'tolerance': EXPECTED_PATTERNS.get('mimo_tolerance', 0.10)
            }
        }
    
        # MIMO zero rate pattern
        if 'CARR.AverageLayersDl' in cell_data.columns:
            mimo_zero_rate = (cell_data['CARR.AverageLayersDl'] == 0).mean()
            patterns['known_patterns']['mimo_zero_rate'] = {
                'value': float(mimo_zero_rate),
                'expected': EXPECTED_PATTERNS['mimo_zero_rate'],
                'description': 'Fraction of records with MIMO disabled'
            }
    
        # CQI no measurement rate (now NaN after correction)
        if 'DRB.UECqiDl' in ue_data.columns:
            cqi_nan_rate = ue_data['DRB.UECqiDl'].isna().mean()
            patterns['known_patterns']['cqi_no_measurement_rate'] = {
                'value': float(cqi_nan_rate),
                'expected': EXPECTED_PATTERNS['cqi_no_measurement_rate'],
                'description': 'Fraction of UE records with no CQI measurement'
            }
    
        # UL/DL symmetry pattern
        if 'DRB.UEThpUl' in cell_data.columns and 'DRB.UEThpDl' in cell_data.columns:
            valid_data = cell_data[['DRB.UEThpUl', 'DRB.UEThpDl']].dropna()
            if len(valid_data) > 100:
                ul_dl_corr = valid_data['DRB.UEThpUl'].corr(valid_data['DRB.UEThpDl'])
                patterns['known_patterns']['ul_dl_symmetry'] = {
                    'value': float(ul_dl_corr),
                    'expected': EXPECTED_PATTERNS['ul_dl_symmetry'],
                    'description': 'Correlation between UL and DL throughput'
                }
    
        return patterns
    
    def _create_temporal_templates(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame) -> Dict:
        """Create temporal pattern templates"""
        from common.constants import TEMPORAL_COEFFICIENTS
    
        if 'timestamp' not in cell_data.columns:
            return {'error': 'No timestamp column found'}
    
        # Extract temporal features
        cell_data_temp = cell_data.copy()
        cell_data_temp['hour'] = cell_data_temp['timestamp'].dt.hour
        cell_data_temp['dayofweek'] = cell_data_temp['timestamp'].dt.dayofweek
    
        templates = {
            'hourly_patterns': {},
            'daily_patterns': {},
            'peak_hours': [],
            'temporal_variation': {},
            'metadata': {
                'creation_time': datetime.now().isoformat(),
                'sample_period': f"{cell_data['timestamp'].min()} to {cell_data['timestamp'].max()}"
            }
        }
    
        # Hourly connection patterns
        if 'RRC.ConnMean' in cell_data_temp.columns:
            hourly_connections = cell_data_temp.groupby('hour')['RRC.ConnMean'].mean()
            templates['hourly_patterns']['connections'] = {int(k): float(v) for k, v in hourly_connections.to_dict().items()}
    
        # Hourly load patterns and peak hour identification
        if 'RRU.PrbUsedDl' in cell_data_temp.columns:
            hourly_load = cell_data_temp.groupby('hour')['RRU.PrbUsedDl'].mean()
            templates['hourly_patterns']['load_dl'] = {int(k): float(v) for k, v in hourly_load.to_dict().items()}
        
            # Use configured percentile for peak hours
            peak_percentile = TEMPORAL_COEFFICIENTS.get('peak_hour_percentile', 0.70)
            peak_threshold = hourly_load.quantile(peak_percentile)
            peak_hours = hourly_load[hourly_load >= peak_threshold].index.tolist()
            templates['peak_hours'] = [int(h) for h in peak_hours]
    
        # Daily patterns (by day of week)
        metrics_for_daily = ['RRC.ConnMean', 'RRU.PrbUsedDl', 'DRB.UEThpDl']
        available_metrics = [m for m in metrics_for_daily if m in cell_data_temp.columns]
    
        if available_metrics:
            daily_patterns = cell_data_temp.groupby('dayofweek')[available_metrics].mean()
            templates['daily_patterns'] = {
                metric: {int(k): float(v) for k, v in daily_patterns[metric].to_dict().items()}
                for metric in available_metrics
            }
    
        # Use temporal coefficients from config
        templates['temporal_variation'] = {
            'minute_variation_coefficient': TEMPORAL_COEFFICIENTS.get('minute_variation_coefficient', 0.15),
            'hour_variation_coefficient': TEMPORAL_COEFFICIENTS.get('hour_variation_coefficient', 0.30)
        }
    
        return templates
    
    def _create_quality_thresholds(self) -> Dict:
        """Create quality thresholds for all dimensions"""
        
        from common.constants import BAND_LIMITS, EXPECTED_ENTITIES, WINDOW_SPECS
        
        thresholds = {
            'completeness': {
                'expected_cell_records_per_window': WINDOW_SPECS['expected_records']['cells_per_window'],
                'expected_ue_records_per_window': WINDOW_SPECS['expected_records']['ues_per_window'],
                'expected_total_records_per_window': WINDOW_SPECS['expected_records']['total_per_window'],
                'minimum_completeness_ratio': WINDOW_SPECS.get('min_completeness', 0.95)
            },
            'consistency': {
                'physical_rules': {
                    'prb_band_limits': BAND_LIMITS,
                    'mean_le_max_tolerance': 0.01,
                    'min_le_mean_tolerance': 0.01
                },
                'correlation_tolerances': {
                    'prb_throughput_tolerance': 0.30,
                    'power_load_tolerance': 0.30,
                    'connections_throughput_tolerance': 0.30
                }
            },
            'accuracy': {
                'z_score_threshold': 3.0,
                'spectral_efficiency_max': 30.0,
                'energy_validation_minor_threshold': 0.10,
                'energy_validation_major_threshold': 0.20,
                'pattern_deviation_threshold': 0.10
            },
            'timeliness': {
                'expected_update_interval_seconds': 60,
                'update_interval_tolerance_seconds': 5,
                'staleness_detection_threshold': 3,
                'ks_test_significance_level': 0.05
            },
            'skewness': {
                'jsd_normal_threshold': 0.1,
                'jsd_moderate_threshold': 0.3,
                'distribution_bins': 50
            },
            'metadata': {
                'creation_time': datetime.now().isoformat(),
                'source': 'Phase 1 baseline creation'
            }
        }
        
        return thresholds
    
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
                
            artifact_file = artifacts_dir / f'{artifact_name}_v1_{timestamp}.pkl'
            with open(artifact_file, 'wb') as f:
                pickle.dump(artifact_data, f)
            artifact_paths[artifact_name] = str(artifact_file)
        
        # Save sample windows
        windows_dir = self.paths['windows']
        windows_dir.mkdir(parents=True, exist_ok=True)
        
        sample_windows = windows[:50] if len(windows) > 50 else windows
        windows_file = windows_dir / f'sample_windows_{timestamp}.pkl'
        with open(windows_file, 'wb') as f:
            pickle.dump(sample_windows, f)
        artifact_paths['sample_windows_file'] = str(windows_file)
        
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