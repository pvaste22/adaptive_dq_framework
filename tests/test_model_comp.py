"""
Test script to compare PCA-generated labels vs trained model predictions
Usage: python tests/test_model_comparison.py --windows_dir <path> --n_windows 10
"""

import argparse
import sys
from pathlib import Path
import time
from typing import Dict, List, Tuple
import json
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# Imports from your codebase
from scoring.traditional_scorer import score_window, _default_dimensions
from scoring.pca_consolidator import PCAConsolidator
from data_processing.feature_extractor import make_feature_row, load_feature_schema
from common.constants import PATHS, MAIN_CONFIG
from common.logger import get_phase2_logger

logger = get_phase2_logger('model_comparison_test')


class ModelComparisonTester:
    """Compare PCA labels vs trained model predictions on sample windows."""
    
    def __init__(self, model_dir: Path = None, pca_model_path: Path = None):
        """
        Initialize tester with model paths.
        
        Args:
            model_dir: Path to trained model directory (default: latest symlink)
            pca_model_path: Path to PCA consolidator JSON
        """
        self.logger = logger
        
        # Load PCA consolidator
        if pca_model_path is None:
            pca_model_path = Path("./data/artifacts/models/pca_consolidator.json")
        
        if not pca_model_path.exists():
            raise FileNotFoundError(f"PCA model not found: {pca_model_path}")
        
        with open(pca_model_path, 'r') as f:
            self.pca_consolidator = PCAConsolidator.from_dict(json.load(f))
        self.logger.info(f"Loaded PCA consolidator from {pca_model_path}")
        
        # Load trained XGBoost model
        if model_dir is None:
            # Use latest symlink
            latest_link = Path(MAIN_CONFIG.get("ML_Training", {}).get("out_root", 
                              "./data/artifacts/models")) / "latest"
            if latest_link.exists():
                model_dir = latest_link.resolve() 
            else:
                raise FileNotFoundError(f"Latest model symlink not found: {latest_link}")
        
        self.model_dir = Path(model_dir)
        self.logger.info(f"Loading model from {self.model_dir}")
        
        # Load model artifacts
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        self.meta = self._load_metadata()
        self.feature_schema = load_feature_schema()
        print(" feat cnt : ",len(self.feature_schema))
        if not self.feature_schema:
            raise RuntimeError("Feature schema not found. Run feature extraction first.")
        
        self.logger.info(f"Model loaded: {self.meta.get('final_val_mae', 'N/A')} MAE on validation")
    
    def _load_model(self) -> xgb.Booster:
        """Load XGBoost model."""
        model_path = self.model_dir / "model.joblib"
        if not model_path.exists():
            model_path = self.model_dir / "model.bin"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found in {self.model_dir}")
        
        if model_path.suffix == ".joblib":
            return joblib.load(model_path)
        else:
            model = xgb.Booster()
            model.load_model(str(model_path))
            return model
    
    def _load_scaler(self):
        """Load scaler if it exists."""
        scaler_path = self.model_dir / "scaler.joblib"
        if scaler_path.exists():
            self.logger.info("Loading scaler")
            return joblib.load(scaler_path)
        else:
            self.logger.info("No scaler found (features not scaled during training)")
            return None
    
    def _load_metadata(self) -> Dict:
        """Load model metadata."""
        meta_path = self.model_dir / "meta.json"
        if not meta_path.exists():
            self.logger.warning(f"Metadata not found: {meta_path}")
            return {}
        
        with open(meta_path, 'r') as f:
            return json.load(f)
    
    def score_single_window(self, window_path: Path) -> Dict:
        """
        Score a single window with both PCA and trained model.
        
        Returns dict with:
            - window_id
            - dimension_scores (dict)
            - pca_label (float)
            - model_prediction (float)
            - absolute_diff (float)
        """
        window_id = window_path.name
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Processing: {window_id}")
        self.logger.info(f"{'='*60}")
        
        # === STEP 1: Traditional Scoring (Dimensions) ===
        self.logger.info("Step 1: Calculating dimension scores...")
        dimensions = _default_dimensions()
        t_dim0 = time.perf_counter()
        dim_results = score_window(window_path, dimensions=dimensions)
        t_dim1 = time.perf_counter()
        baseline_dim_time_ms = int(round((t_dim1 - t_dim0) * 1000))
        #dim_results = score_window(window_path, dimensions=dimensions)
        
        # Extract MPR scores
        dim_scores = {}
        for dim_name, res in dim_results.items():
            mpr = res.get('mpr', res.get('score', 0.0))
            dim_scores[f"{dim_name}_mpr"] = float(mpr)
            self.logger.info(f"  {dim_name:15s}: Score={mpr:.4f}, ")
                           #f"APR={res.get('apr', 0.0):.4f}, "
                           #f"Coverage={res.get('coverage', 0.0):.4f}")
        
        # === STEP 2: PCA Consolidation ===
        self.logger.info("\nStep 2: Generating PCA label...")
        dim_df = pd.DataFrame([dim_scores])
        t_pca0 = time.perf_counter()
        pca_label = self.pca_consolidator.transform(dim_df)[0]
        t_pca1 = time.perf_counter()
        baseline_pca_time_ms = int(round((t_pca1 - t_pca0) * 1000))
        baseline_total_ms = baseline_dim_time_ms + baseline_pca_time_ms
        #pca_label = self.pca_consolidator.transform(dim_df)[0]
        self.logger.info(f"  PCA Label: {pca_label:.4f}")
        
        # === STEP 3: Feature Extraction ===
        self.logger.info("\nStep 3: Extracting features for model...")
        #from quality_dimensions.base_dimension import BaseDimension
        #base_dim = BaseDimension('temp')
        from scoring.traditional_scorer import load_window_from_disk
        window_data = load_window_from_disk(window_path)
        
        t_feat0 = time.perf_counter()

        feature_row = make_feature_row(window_data)
        print(" actually generated feats: ", len(feature_row))
        feature_row['window_id'] = window_id
        
        # Create feature DataFrame with proper column order
        feat_df = pd.DataFrame([feature_row])
        
        # Align features to training schema
        drop_cols = self.meta.get('drop_cols', ['window_id'])
        train_features = [f for f in self.meta.get('features', []) 
                         if f not in drop_cols]
        
        # Ensure all training features are present
        for col in train_features:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        
        X = feat_df[train_features].copy()
        
        # Handle inf/nan
        X = X.replace([np.inf, -np.inf], np.nan)
        median_vals = X.median()
        X = X.fillna(median_vals)
        
        self.logger.info(f"  Extracted {len(train_features)} features")
        
        # === STEP 4: Model Prediction ===
        self.logger.info("\nStep 4: Predicting with trained model...")
        
        # Apply scaler if used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            self.logger.info("  Applied feature scaling")
        else:
            X_scaled = X.values
        

        t_pred0 = time.perf_counter()
        dtest = xgb.DMatrix(X_scaled)
        model_pred = float(self.model.predict(dtest)[0])
        t_pred1 = time.perf_counter()

        model_predict_time_ms = int(round((t_pred1 - t_pred0) * 1000))
        model_total_ms = int(round((t_pred1 - t_feat0) * 1000))
        # Predict
        #dtest = xgb.DMatrix(X_scaled)
        #model_pred = float(self.model.predict(dtest)[0])
        self.logger.info(f"  Model Prediction: {model_pred:.4f}")
        
        # === STEP 5: Comparison ===
        diff = abs(pca_label - model_pred)
        self.logger.info(f"\n{'─'*60}")
        self.logger.info(f"COMPARISON:")
        self.logger.info(f"  PCA Label:         {pca_label:.4f}")
        self.logger.info(f"  Model Prediction:  {model_pred:.4f}")
        self.logger.info(f"  Absolute Diff:     {diff:.4f}")
        self.logger.info(f"{'─'*60}")
        
        return {
            'window_id': window_id,
            'window_path': str(window_path),
            'dimension_scores': dim_scores,
            'pca_label': pca_label,
            'model_prediction': model_pred,
            'absolute_diff': diff,
            'dim_details': dim_results,
            # NEW fields:
            'baseline_dim_time_ms': baseline_dim_time_ms,
            'baseline_pca_time_ms': baseline_pca_time_ms,
            'baseline_total_ms': baseline_total_ms,
            'model_predict_time_ms': model_predict_time_ms,
            'model_total_ms': model_total_ms
        }

    
    def test_multiple_windows(self, windows_dir: Path, n_windows: int = 10) -> pd.DataFrame:
        """
        Test multiple windows and return summary DataFrame.
        
        Args:
            windows_dir: Directory containing window subdirectories
            n_windows: Number of windows to test
            
        Returns:
            DataFrame with comparison results
        """
        windows_dir = Path(windows_dir)
        if not windows_dir.exists():
            raise FileNotFoundError(f"Windows directory not found: {windows_dir}")
        
        # Get window directories
        window_paths = sorted([p for p in windows_dir.iterdir() 
                              if p.is_dir() and (p / 'cell_data.parquet').exists()])
        
        if not window_paths:
            raise RuntimeError(f"No valid windows found in {windows_dir}")
        
        # Sample windows
        if len(window_paths) > n_windows:
            step = len(window_paths) // n_windows
            window_paths = window_paths[::step][:n_windows]
        
        self.logger.info(f"\n{'#'*60}")
        self.logger.info(f"Testing {len(window_paths)} windows")
        self.logger.info(f"{'#'*60}\n")
        
        results = []
        for i, window_path in enumerate(window_paths, 1):
            self.logger.info(f"\n[{i}/{len(window_paths)}]")
            try:
                result = self.score_single_window(window_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing {window_path.name}: {e}", 
                                exc_info=True)
                results.append({
                    'window_id': window_path.name,
                    'window_path': str(window_path),
                    'error': str(e)
                })
        
        # Create summary DataFrame
        summary_data = []
        for r in results:
            if 'error' in r:
                summary_data.append({
                    'window_id': r['window_id'],
                    'pca_label': None,
                    'model_pred': None,
                    'abs_diff': None,
                    'error': r['error']
                })
            else:
                row = {
                    'window_id': r['window_id'],
                    'pca_label': r['pca_label'],
                    'model_pred': r['model_prediction'],
                    'abs_diff': r['absolute_diff'],
                    # timings:
                    'baseline_dim_time_ms': r['baseline_dim_time_ms'],
                    'baseline_pca_time_ms': r['baseline_pca_time_ms'],
                    'baseline_total_ms': r['baseline_total_ms'],
                    'model_predict_time_ms': r['model_predict_time_ms'],
                    'model_total_ms': r['model_total_ms'],
                }
                # Add dimension scores
                row.update(r['dimension_scores'])
                summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        
        # Print summary statistics
        self.print_summary(df)
        
        return df
    
    def print_summary(self, df: pd.DataFrame):
        """Print summary statistics."""
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        if 'pca_label' in df.columns and 'model_pred' in df.columns:
            valid_df = df.dropna(subset=['pca_label', 'model_pred'])
            
            if len(valid_df) > 0:
                print(f"\nWindows Tested: {len(valid_df)}")
                print(f"\nPCA Labels:")
                print(f"  Mean:   {valid_df['pca_label'].mean():.4f}")
                print(f"  Std:    {valid_df['pca_label'].std():.4f}")
                print(f"  Min:    {valid_df['pca_label'].min():.4f}")
                print(f"  Max:    {valid_df['pca_label'].max():.4f}")
                
                print(f"\nModel Predictions:")
                print(f"  Mean:   {valid_df['model_pred'].mean():.4f}")
                print(f"  Std:    {valid_df['model_pred'].std():.4f}")
                print(f"  Min:    {valid_df['model_pred'].min():.4f}")
                print(f"  Max:    {valid_df['model_pred'].max():.4f}")
                
                print(f"\nAbsolute Differences:")
                print(f"  Mean:   {valid_df['abs_diff'].mean():.4f}")
                print(f"  Std:    {valid_df['abs_diff'].std():.4f}")
                print(f"  Min:    {valid_df['abs_diff'].min():.4f}")
                print(f"  Max:    {valid_df['abs_diff'].max():.4f}")
                
                # Correlation
                corr = valid_df['pca_label'].corr(valid_df['model_pred'])
                print(f"\nCorrelation (PCA vs Model): {corr:.4f}")
                
                # MAE between PCA and Model
                mae = (valid_df['pca_label'] - valid_df['model_pred']).abs().mean()
                print(f"Mean Absolute Error: {mae:.4f}")
        
        errors = df[df['error'].notna()] if 'error' in df.columns else pd.DataFrame()
        if len(errors) > 0:
            print(f"\nErrors: {len(errors)} windows failed")
            for _, row in errors.iterrows():
                print(f"  - {row['window_id']}: {row['error']}")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Compare PCA labels vs trained model predictions'
    )
    parser.add_argument('--windows_dir', type=str, 
                       default='./data/processed/training/v0/windows',
                       help='Directory containing window subdirectories')
    parser.add_argument('--n_windows', type=int, default=10,
                       help='Number of windows to test')
    parser.add_argument('--model_dir', type=str, default=None,
                       help='Path to model directory (default: latest symlink)')
    parser.add_argument('--pca_path', type=str, 
                       default='./data/artifacts/models/pca_consolidator.json',
                       help='Path to PCA consolidator JSON')
    parser.add_argument('--output_csv', type=str, default=None,
                       help='Optional: Save results to CSV')
    parser.add_argument('--single_window', type=str, default=None,
                       help='Test single window by name')
    
    args = parser.parse_args()
    
    try:
        # Initialize tester
        tester = ModelComparisonTester(
            model_dir=Path(args.model_dir) if args.model_dir else None,
            pca_model_path=Path(args.pca_path)
        )
        
        if args.single_window:
            # Test single window
            windows_dir = Path(args.windows_dir)
            window_path = windows_dir / args.single_window
            
            if not window_path.exists():
                print(f"Error: Window not found: {window_path}")
                return 1
            
            result = tester.score_single_window(window_path)
            
            # Print detailed results
            print("\n" + "="*80)
            print("DETAILED RESULTS")
            print("="*80)
            print(f"\nWindow: {result['window_id']}")
            print(f"\nDimension Scores:")
            for dim, score in result['dimension_scores'].items():
                print(f"  {dim:25s}: {score:.4f}")
            print(f"\nPCA Label:        {result['pca_label']:.4f}")
            print(f"Model Prediction: {result['model_prediction']:.4f}")
            print(f"Absolute Diff:    {result['absolute_diff']:.4f}")
            
        else:
            # Test multiple windows
            df = tester.test_multiple_windows(
                Path(args.windows_dir),
                n_windows=args.n_windows
            )
            
            # Save to CSV if requested
            if args.output_csv:
                output_path = Path(args.output_csv)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_path, index=False)
                print(f"\nResults saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

"""

# 300s
python ./tests/test_model_comp.py \
  --windows_dir ./data/processed/training/v0/windows_300s \
  --model_dir  ./data/artifacts/models/v1.3.0_300s/20251021_235803_6907daa9a7 \
  --pca_path   ./data/artifacts/models/pca_consolidator_300s.json \
  --n_windows  50 \
  --output_csv ./data/results/300s/model_vs_pca.csv

# 600s
python ./tests/test_model_comp.py \
  --windows_dir ./data/processed/training/v0/windows_600s \
  --model_dir  ./data/artifacts/models/v1.3.0_600s/20251027_161325_1ded779ab5 \
  --pca_path   ./data/artifacts/models/pca_consolidator_600s.json \
  --n_windows  50 \
  --output_csv ./data/results/600s/model_vs_pca.csv

# 180s
python ./tests/test_model_comp.py \
  --windows_dir ./data/processed/training/v0/windows_180s \
  --model_dir  ./data/artifacts/models/v1.3.0_180s/20251027_154044_b37b884647 \
  --pca_path   ./data/artifacts/models/pca_consolidator_180s.json \
  --n_windows  50 \
  --output_csv ./data/results/180s/model_vs_pca.csv  


for 300s windows:
================================================================================
SUMMARY STATISTICS
================================================================================

Windows Tested: 50

PCA Labels:
  Mean:   0.5002
  Std:    0.1606
  Min:    0.2000
  Max:    0.7700

Model Predictions:
  Mean:   0.4999
  Std:    0.1611
  Min:    0.2025
  Max:    0.7819

Absolute Differences:
  Mean:   0.0149
  Std:    0.0144
  Min:    0.0006
  Max:    0.0663

Correlation (PCA vs Model): 0.9917
Mean Absolute Error: 0.0149


================================================================================
SUMMARY STATISTICS latest
================================================================================

Windows Tested: 50

PCA Labels:
  Mean:   0.5002
  Std:    0.1606
  Min:    0.2000
  Max:    0.7700

Model Predictions:
  Mean:   0.5012
  Std:    0.1594
  Min:    0.2082
  Max:    0.7728

Absolute Differences:
  Mean:   0.0124
  Std:    0.0120
  Min:    0.0000
  Max:    0.0604

Correlation (PCA vs Model): 0.9941
Mean Absolute Error: 0.0124

for 600s windows:
================================================================================
SUMMARY STATISTICS
================================================================================

Windows Tested: 50

PCA Labels:
  Mean:   0.5084
  Std:    0.1876
  Min:    0.0900
  Max:    0.7900

Model Predictions:
  Mean:   0.5096
  Std:    0.1861
  Min:    0.0903
  Max:    0.7894

Absolute Differences:
  Mean:   0.0036
  Std:    0.0087
  Min:    0.0000
  Max:    0.0427

Correlation (PCA vs Model): 0.9988
Mean Absolute Error: 0.0036

================================================================================
SUMMARY STATISTICS latest
================================================================================

Windows Tested: 50

PCA Labels:
  Mean:   0.5084
  Std:    0.1876
  Min:    0.0900
  Max:    0.7900

Model Predictions:
  Mean:   0.5061
  Std:    0.1891
  Min:    0.0922
  Max:    0.7880

Absolute Differences:
  Mean:   0.0079
  Std:    0.0125
  Min:    0.0000
  Max:    0.0502

Correlation (PCA vs Model): 0.9970
Mean Absolute Error: 0.0079




================================================================================
SUMMARY STATISTICS
================================================================================

Windows Tested: 50

PCA Labels:
  Mean:   0.5186
  Std:    0.1580
  Min:    0.1600
  Max:    0.7800

Model Predictions:
  Mean:   0.5185
  Std:    0.1573
  Min:    0.1590
  Max:    0.7749

Absolute Differences:
  Mean:   0.0081
  Std:    0.0086
  Min:    0.0004
  Max:    0.0499

Correlation (PCA vs Model): 0.9972
Mean Absolute Error: 0.0081
================================================================================


for 180s windows:

================================================================================
SUMMARY STATISTICS
================================================================================

Windows Tested: 50

PCA Labels:
  Mean:   0.5116
  Std:    0.1705
  Min:    0.0300
  Max:    0.7700

Model Predictions:
  Mean:   0.5092
  Std:    0.1673
  Min:    0.0395
  Max:    0.7695

Absolute Differences:
  Mean:   0.0144
  Std:    0.0109
  Min:    0.0001
  Max:    0.0456

Correlation (PCA vs Model): 0.9944
Mean Absolute Error: 0.0144
================================================================================


================================================================================
SUMMARY STATISTICS
================================================================================

Windows Tested: 50

PCA Labels:
  Mean:   0.5176
  Std:    0.1744
  Min:    0.0200
  Max:    0.8200

Model Predictions:
  Mean:   0.5106
  Std:    0.1293
  Min:    0.2166
  Max:    0.7308

Absolute Differences:
  Mean:   0.0435
  Std:    0.0357
  Min:    0.0073
  Max:    0.1966

Correlation (PCA vs Model): 0.9751
Mean Absolute Error: 0.0435
================================================================================

"""    