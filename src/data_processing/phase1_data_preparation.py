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

Based on detailed_implementation_plan.md
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
from typing import Dict, List, Tuple
import warnings

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent 
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))


from data_processing.data_loader import DataLoader
from data_processing.unit_converter import UnitConverter
from data_processing.window_generator import WindowGenerator

class Phase1Orchestrator:
    """
    Orchestrates Phase 1: Data Preparation and Baseline Creation
    
    Implements the complete workflow from raw VIAVI data to processed windows
    and baseline artifacts ready for Phase 2 quality scoring.
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
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
            'divergence_array_placeholder'  # Will be populated during training
        ]
        
        # Initialize processors
        self.data_loader = DataLoader(self.config.get('data_loader', {}))
        self.unit_converter = UnitConverter(self.config.get('band_limits', {}))
        self.window_generator = WindowGenerator(self.config.get('window_generator', {}))
        
        # Artifacts storage
        self.artifacts = {}
        self.windows = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration or use defaults"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'data_paths': {
                'cell_file': 'data/raw/CellReports_v0.csv',
                'ue_file': 'data/raw/UEReports_v0.csv'
            },
            'output_paths': {
                'artifacts_dir': 'data/artifacts',
                'processed_data_dir': 'data/processed/training/v0',
                'windows_dir': 'data/processed/windowed_data'
            },
            'phase1_settings': {
                'create_baselines': True,
                'validate_conversions': True,
                'save_windows': True,
                'generate_reports': True
            },
            'expected_entities': {
                'cells': 52,
                'ues': 48
            },
            'band_limits': {
                'B2': 100,
                'B13': 75,
                'N77': 273
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for Phase 1"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/phase1_preparation.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def run_phase1_complete(self) -> Dict:
        """
        Run complete Phase 1 workflow.
        
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
            self.logger.error(f"Phase 1 failed: {str(e)}")
            results['execution_summary'] = {
                'status': 'FAILED',
                'error': str(e),
                'ready_for_phase2': False
            }
            raise
        
        return results
    
    def _step1_load_and_preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Step 1: Load and preprocess raw VIAVI data"""
        
        # Update data loader config with file paths
        self.data_loader.config.update({
            'cell_file': self.config['data_paths']['cell_file'],
            'ue_file': self.config['data_paths']['ue_file'],
            'expected_cells': self.config['expected_entities']['cells'],
            'expected_ues': self.config['expected_entities']['ues']
        })
        
        # Load data with built-in preprocessing
        cell_data, ue_data = self.data_loader.load_data()
        
        # Log summary
        data_summary = self.data_loader.get_data_summary(cell_data, ue_data)
        self.logger.info(f"Loaded {data_summary['cell_data']['total_records']} cell records")
        self.logger.info(f"Loaded {data_summary['ue_data']['total_records']} UE records")
        self.logger.info(f"Date range: {data_summary['cell_data']['date_range'][0]} to {data_summary['cell_data']['date_range'][1]}")
        
        return cell_data, ue_data
    
    def _step2_apply_conversions(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Step 2: Apply critical unit conversions"""
        
        # Apply comprehensive unit conversions
        cell_data_converted, ue_data_converted = self.unit_converter.standardize_units_comprehensive(
            cell_data, ue_data
        )
        
        # Generate conversion report
        if self.config['phase1_settings']['validate_conversions']:
            #conversion_report = self.unit_converter.generate_conversion_report(
             #   cell_data_converted, ue_data_converted
            #)
            
            # Log key conversion results
            self.logger.info("Unit conversion validation:")
            """if 'energy_validation' in conversion_report['validation_results']:
                energy_val = conversion_report['validation_results']['energy_validation']
                self.logger.info(f"  Energy formula validation - Mean error: {energy_val['mean_error_percent']:.3f}")
                self.logger.info(f"  Major energy errors: {energy_val['major_errors_percent']:.1f}%")
            
            # Store conversion report for later use
            self.artifacts['conversion_report'] = conversion_report"""
        
        return cell_data_converted, ue_data_converted
    
    def _step3_generate_windows(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame) -> List[Dict]:
        """Step 3: Generate 5-minute windows"""
        
        # Generate windows
        windows = self.window_generator.generate_windows(cell_data, ue_data)
        
        # Get window statistics
        window_stats = self.window_generator.get_window_statistics(windows)
        
        # Log window generation results
        self.logger.info(f"Generated {window_stats['window_count']} valid windows")
        self.logger.info(f"Time span: {window_stats['time_span']['total_hours']:.1f} hours")
        self.logger.info(f"Mean completeness: {window_stats['completeness']['mean']:.3f}")
        self.logger.info(f"Mean records per window: {window_stats['record_counts']['mean']:.1f}")
        
        # Store window statistics
        self.artifacts['window_statistics'] = window_stats
        
        # Store windows for later use
        self.windows = windows
        
        return windows
    
    def _step4_create_baselines(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame, windows: List[Dict]) -> Dict:
        """Step 4: Create all baseline artifacts"""
        
        artifacts = {}
        
        # 1. Statistical Baselines (for z-score calculations)
        self.logger.info("Creating statistical baselines...")
        artifacts['statistical_baselines'] = self._create_statistical_baselines(cell_data, ue_data)
        
        # 2. Historical PDFs (for drift detection and skewness)
        self.logger.info("Creating histogram-based PDFs...")
        artifacts['historical_pdfs'] = self._create_histogram_pdfs(cell_data, ue_data)
        
        # 3. Correlation Matrix (for consistency checking)
        self.logger.info("Creating correlation matrix...")
        artifacts['correlation_matrix'] = self._create_correlation_matrix(cell_data, ue_data)
        
        # 4. Pattern Baselines (for accuracy validation)
        self.logger.info("Creating pattern baselines...")
        artifacts['pattern_baselines'] = self._create_pattern_baselines(cell_data, ue_data)
        
        # 5. Temporal Templates (for timeliness and cycle detection)
        self.logger.info("Creating temporal templates...")
        artifacts['temporal_templates'] = self._create_temporal_templates(cell_data, ue_data)
        
        # 6. Quality Thresholds (for all dimensions)
        self.logger.info("Creating quality thresholds...")
        artifacts['quality_thresholds'] = self._create_quality_thresholds()
        
        # 7. Sample Windows (for KS tests in timeliness)
        self.logger.info("Sampling historical windows...")
        artifacts['sample_windows'] = self._create_sample_windows(windows)
        
        # 8. Metadata Configuration
        self.logger.info("Creating metadata configuration...")
        artifacts['metadata_config'] = self._create_metadata_config()
        
        # 9. Placeholder for divergence array (populated during training)
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
            'RRU.PrbUsedDl', 'RRU.PrbUsedUl', 'DRB.UEThpDl', 'DRB.UEThpUl',
            'RRC.ConnMean', 'PEE.AvgPower'
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
        
        # Key metrics for distribution analysis (from baseline_metrics_specification.md)
        cell_metrics = ['RRU.PrbUsedDl', 'RRU.PrbUsedUl', 'DRB.UEThpDl', 'DRB.UEThpUl', 'RRC.ConnMean', 'PEE.AvgPower']
        ue_metrics = ['DRB.UEThpDl', 'DRB.UEThpUl', 'DRB.UECqiDl', 'DRB.UECqiUl', 'RRU.PrbUsedDl', 'RRU.PrbUsedUl']
        
        pdfs = {
            'cell_distributions': {},
            'ue_distributions': {},
            'metadata': {
                'creation_time': datetime.now().isoformat(),
                'bin_count': 50,  # Fixed bin count as per specification
                'total_metrics': len(cell_metrics) + len(ue_metrics)
            }
        }
        
        # Create cell metric PDFs
        for metric in cell_metrics:
            if metric in cell_data.columns:
                data = cell_data[metric].dropna()
                
                # Determine bin strategy
                if metric == 'RRC.ConnMean':
                    # Discrete values for connection count
                    bins = np.arange(data.min(), data.max() + 2) - 0.5
                else:
                    # Continuous values
                    bins = np.linspace(data.min(), data.max(), 51)  # 50 bins
                
                hist, bin_edges = np.histogram(data, bins=bins)
                pdf = hist / hist.sum()  # Normalize to probability
                
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
                
                # Determine bin strategy
                if metric in ['DRB.UECqiDl', 'DRB.UECqiUl']:
                    # Discrete CQI values (0-15)
                    bins = np.arange(0, 17) - 0.5
                else:
                    # Continuous values
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
        
        correlations = {
            'expected_correlations': {},
            'actual_correlations': {},
            'metadata': {
                'creation_time': datetime.now().isoformat(),
                'sample_size_cells': len(cell_data),
                'tolerance': 0.30  # ±30% tolerance as per documentation
            }
        }
        
        # Expected correlations from documentation
        expected_pairs = [
            ('RRU.PrbUsedDl', 'DRB.UEThpDl', 0.85),
            ('PEE.AvgPower', 'RRU.PrbUsedDl', 0.72),
            ('RRC.ConnMean', 'DRB.UEThpDl', 0.68)
        ]
        
        # Calculate actual correlations
        for metric1, metric2, expected_corr in expected_pairs:
            if metric1 in cell_data.columns and metric2 in cell_data.columns:
                # Remove null values
                valid_data = cell_data[[metric1, metric2]].dropna()
                
                if len(valid_data) > 10:  # Need minimum data for correlation
                    actual_corr = valid_data[metric1].corr(valid_data[metric2])
                    
                    correlations['expected_correlations'][f'{metric1}_vs_{metric2}'] = expected_corr
                    correlations['actual_correlations'][f'{metric1}_vs_{metric2}'] = float(actual_corr)
        
        return correlations
    
    def _create_pattern_baselines(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame) -> Dict:
        """Create pattern baselines for accuracy validation"""
        
        patterns = {
            'known_patterns': {},
            'metadata': {
                'creation_time': datetime.now().isoformat(),
                'tolerance': 0.10  # ±10% tolerance for pattern deviations
            }
        }
        
        # MIMO zero rate pattern (from documentation: 86% expected)
        if 'CARR.AverageLayersDl' in cell_data.columns:
            mimo_zero_rate = (cell_data['CARR.AverageLayersDl'] == 0).mean()
            patterns['known_patterns']['mimo_zero_rate'] = {
                'value': float(mimo_zero_rate),
                'expected': 0.86,
                'description': 'Fraction of records with MIMO disabled'
            }
        
        # CQI zero rate pattern (from documentation: 60% expected)
        if 'DRB.UECqiDl' in ue_data.columns:
            cqi_zero_rate = (ue_data['DRB.UECqiDl'] == 0).mean()
            patterns['known_patterns']['cqi_zero_rate'] = {
                'value': float(cqi_zero_rate),
                'expected': 0.60,
                'description': 'Fraction of UE records with no CQI measurement'
            }
        
        # UL/DL symmetry pattern (from documentation: 87% expected)
        if len(cell_data) > 100:
            ul_dl_corr = cell_data['DRB.UEThpUl'].corr(cell_data['DRB.UEThpDl'])
            patterns['known_patterns']['ul_dl_symmetry'] = {
                'value': float(ul_dl_corr),
                'expected': 0.87,
                'description': 'Correlation between UL and DL throughput'
            }
        
        return patterns
    
    def _create_temporal_templates(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame) -> Dict:
        """Create temporal pattern templates"""
        
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
        hourly_connections = cell_data_temp.groupby('hour')['RRC.ConnMean'].mean()
        templates['hourly_patterns']['connections'] = hourly_connections.to_dict()
        
        # Hourly load patterns
        hourly_load = cell_data_temp.groupby('hour')['RRU.PrbUsedDl'].mean()
        templates['hourly_patterns']['load_dl'] = hourly_load.to_dict()
        
        # Identify peak hours (top 30% of load)
        peak_threshold = hourly_load.quantile(0.70)
        peak_hours = hourly_load[hourly_load >= peak_threshold].index.tolist()
        templates['peak_hours'] = peak_hours
        
        # Daily patterns
        daily_patterns = cell_data_temp.groupby('dayofweek').agg({
            'RRC.ConnMean': 'mean',
            'RRU.PrbUsedDl': 'mean',
            'DRB.UEThpDl': 'mean'
        })
        templates['daily_patterns'] = daily_patterns.to_dict()
        
        # Temporal variation coefficients
        templates['temporal_variation'] = {
            'minute_variation_coefficient': 0.15,  # Expected variation per minute
            'hour_variation_coefficient': 0.30     # Expected variation per hour
        }
        
        return templates
    
    def _create_quality_thresholds(self) -> Dict:
        """Create quality thresholds for all dimensions"""
        
        thresholds = {
            'completeness': {
                'expected_cell_records_per_window': 260,  # 52 cells × 5 timestamps
                'expected_ue_records_per_window': 240,    # 48 UEs × 5 timestamps
                'expected_total_records_per_window': 500,
                'minimum_completeness_ratio': 0.95
            },
            'consistency': {
                'physical_rules': {
                    'prb_band_limits': self.config['band_limits'],
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
                'spectral_efficiency_max': 30.0,  # bps/Hz theoretical limit
                'energy_validation_minor_threshold': 0.10,
                'energy_validation_major_threshold': 0.20,
                'pattern_deviation_threshold': 0.10
            },
            'timeliness': {
                'expected_update_interval_seconds': 60,
                'update_interval_tolerance_seconds': 5,
                'staleness_detection_threshold': 3,  # consecutive unchanged values
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
    
    def _create_sample_windows(self, windows: List[Dict]) -> Dict:
        """Create sample windows for KS tests"""
        
        # Sample 100 windows evenly across the dataset (or all if fewer)
        n_samples = min(100, len(windows))
        
        if n_samples == 0:
            return {'error': 'No windows available for sampling'}
        
        # Sample evenly across time
        indices = np.linspace(0, len(windows) - 1, n_samples, dtype=int)
        sampled_windows = [windows[i] for i in indices]
        
        # Extract just the metadata and timestamps for storage efficiency
        sample_data = {
            'sample_windows': [],
            'metadata': {
                'creation_time': datetime.now().isoformat(),
                'total_available_windows': len(windows),
                'samples_taken': n_samples,
                'sampling_method': 'evenly_distributed'
            }
        }
        
        for window in sampled_windows:
            sample_data['sample_windows'].append({
                'window_id': window['window_id'],
                'start_time': window['start_time'].isoformat(),
                'end_time': window['end_time'].isoformat(),
                'metadata': window['metadata']
            })
        
        return sample_data
    
    def _create_metadata_config(self) -> Dict:
        """Create metadata configuration"""
        
        metadata = {
            'dataset_info': {
                'source': 'VIAVI O-RAN Simulated Dataset',
                'expected_entities': self.config['expected_entities'],
                'band_configuration': self.config['band_limits'],
                'measurement_interval_seconds': 60
            },
            'known_issues': {
                'prb_tot_unit_mismatch': 'PrbTot reported as percentage, needs conversion',
                'energy_cumulative': 'Energy reported as cumulative, needs differencing',
                'tb_counters_unreliable': 'TB counters should be ignored',
                'qos_flow_semantics': 'QosFlow uses 1-second semantics despite 60s intervals'
            },
            'expected_patterns': {
                'mimo_zero_rate': 0.86,
                'cqi_zero_rate': 0.60,
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
        
        # Ensure output directories exist
        output_paths = self.config['output_paths']
        for path_key, path_value in output_paths.items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
        
        artifact_paths = {}
        
        # Save processed data
        processed_dir = Path(output_paths['processed_data_dir'])
        cell_data.to_parquet(processed_dir / 'cell_data_processed.parquet')
        ue_data.to_parquet(processed_dir / 'ue_data_processed.parquet')
        artifact_paths['processed_data'] = {
            'cell_data': str(processed_dir / 'cell_data_processed.parquet'),
            'ue_data': str(processed_dir / 'ue_data_processed.parquet')
        }
        
        # Save artifacts with versioning
        artifacts_dir = Path(output_paths['artifacts_dir'])
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for artifact_name, artifact_data in artifacts.items():
            if artifact_data.get('status') == 'placeholder':
                continue
                
            artifact_file = artifacts_dir / f'{artifact_name}_v1_{timestamp}.pkl'
            with open(artifact_file, 'wb') as f:
                pickle.dump(artifact_data, f)
            artifact_paths[artifact_name] = str(artifact_file)
        
        # Save windows (sample for efficiency)
        if self.config['phase1_settings']['save_windows']:
            windows_dir = Path(output_paths['windows_dir'])
            
            # Save first 50 windows as examples
            sample_windows = windows[:50]
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
            json.dump(registry, f, indent=2)
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
                'cell_metrics_available': len(stats['cell_metrics']),
                'ue_metrics_available': len(stats['ue_metrics'])
            }
        
        if 'pattern_baselines' in artifacts:
            patterns = artifacts['pattern_baselines']['known_patterns']
            report['quality_indicators']['pattern_validation'] = {
                pattern_name: {
                    'actual': pattern_data['value'],
                    'expected': pattern_data['expected'],
                    'deviation': abs(pattern_data['value'] - pattern_data['expected'])
                }
                for pattern_name, pattern_data in patterns.items()
            }
        
        # Save report
        report_dir = Path('reports')
        report_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = report_dir / f'phase1_completion_report_{timestamp}.json'
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return {'report': report, 'report_path': str(report_file)}


def main():
    """Main execution function"""
    
    # Setup paths
    config_path = 'config/config.yaml' if Path('config/config.yaml').exists() else None
    
    # Initialize and run Phase 1
    orchestrator = Phase1Orchestrator(config_path)
    
    try:
        results = orchestrator.run_phase1_complete()
        
        if results['execution_summary']['status'] == 'SUCCESS':
            print("\n" + "="*60)
            print("PHASE 1 COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"Windows generated: {results['step3']['windows_generated']}")
            print(f"Artifacts created: {results['step4']['artifacts_created']}")
            print(f"Execution time: {results['execution_summary']['total_time_seconds']:.1f} seconds")
            print("\nReady for Phase 2: Quality Scoring and Model Training")
            print("="*60)
            
            return 0
        else:
            print(f"Phase 1 failed: {results['execution_summary']['error']}")
            return 1
            
    except Exception as e:
        print(f"Phase 1 execution failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())