#!/usr/bin/env python3
"""
Ablation Study Script - Remove one dimension at a time
Evaluates impact of each quality dimension on final model performance
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import hashlib

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Your existing modules
from scoring.traditional_scorer import score_windows_in_directory
from scoring.pca_consolidator import PCAConsolidator
from data_processing.feature_extractor import make_feature_row, load_feature_schema

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'windows_dir': './data/processed/training/v0/windows_300s',
    'output_dir': './results/ablation_study',
    'base_labels_path': './data/processed/training/v0/train_labels_300s.parquet',  # Full 5-dimension labels
    
    # XGBoost training params (keep consistent across all runs)
    'xgb_params': {
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
        'max_depth': 4,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'lambda': 0.01,
        'alpha': 0.001,
        'seed': 42,
        'random_state': 42
    },
    'num_boost_rounds': 300,
    'early_stopping_rounds': 50,
    'test_size': 0.2,
    'use_scaler': True,
}

ALL_DIMENSIONS = ['completeness', 'timeliness', 'validity', 'consistency', 'accuracy']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _flatten_dimension_results(dim_results: dict, include_dims: List[str]) -> dict:
    """Extract MPR scores only for specified dimensions"""
    flat = {}
    for dim in include_dims:
        if dim not in dim_results:
            flat[f"{dim}_mpr"] = float("nan")
            continue
        res = dim_results[dim]
        val = res.get("mpr", res.get("score"))
        try:
            flat[f"{dim}_mpr"] = float(val) if val is not None else float("nan")
        except Exception:
            flat[f"{dim}_mpr"] = float("nan")
    return flat


def train_xgboost_model(X_train, y_train, X_val, y_val, params: dict, 
                        num_rounds: int, early_stop: int) -> tuple:
    """Train XGBoost and return model + metrics"""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=[(dval, 'validation')],
        early_stopping_rounds=early_stop,
        verbose_eval=False
    )
    
    # Predict on validation
    y_pred = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    
    metrics = {
        'mae': float(mean_absolute_error(y_val, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_val, y_pred))),
        'r2': float(r2_score(y_val, y_pred)),
        'best_iteration': int(model.best_iteration)
    }
    
    return model, metrics


# ============================================================================
# MAIN ABLATION LOGIC
# ============================================================================

def run_ablation_study():
    """
    Main ablation study pipeline:
    1. Score all windows with 5 dimensions
    2. For each dimension to remove:
       - Create 4-dimension PCA
       - Generate labels
       - Train XGBoost
       - Record metrics
    3. Save results and visualizations
    """
    
    print("="*70)
    print("ABLATION STUDY: Quality Dimension Impact Analysis")
    print("="*70)
    
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ------------------------------------------------------------------------
    # STEP 1: Score all windows with traditional scorer (5 dimensions)
    # ------------------------------------------------------------------------
    print("\n[1/4] Scoring windows with all 5 dimensions...")
    windows_dir = Path(CONFIG['windows_dir'])
    
    if not windows_dir.exists():
        raise FileNotFoundError(f"Windows directory not found: {windows_dir}")
    
    scored = score_windows_in_directory(windows_dir)
    
    if not scored:
        raise RuntimeError(f"No windows scored in {windows_dir}")
    
    window_ids = list(scored.keys())
    print(f"   ✓ Scored {len(window_ids)} windows")
    
    # Extract paths for feature extraction
    paths = [Path(scored[wid]["path"]) for wid in window_ids]
    
    # ------------------------------------------------------------------------
    # STEP 2: Run ablation experiments
    # ------------------------------------------------------------------------
    print(f"\n[2/4] Running {len(ALL_DIMENSIONS) + 1} experiments...")
    
    results = {}
    
    # Baseline: All 5 dimensions
    print(f"\n  → Experiment 0: BASELINE (all 5 dimensions)")
    baseline_result = run_single_experiment(
        experiment_name="baseline_all_5",
        dimensions_to_use=ALL_DIMENSIONS,
        scored_data=scored,
        window_ids=window_ids,
        paths=paths,
        output_dir=output_dir
    )
    results['baseline'] = baseline_result
    
    # Ablation: Remove each dimension one at a time
    for i, dim_to_remove in enumerate(ALL_DIMENSIONS, 1):
        remaining_dims = [d for d in ALL_DIMENSIONS if d != dim_to_remove]
        
        print(f"\n  → Experiment {i}: WITHOUT {dim_to_remove.upper()}")
        print(f"     Using: {remaining_dims}")
        
        exp_result = run_single_experiment(
            experiment_name=f"without_{dim_to_remove}",
            dimensions_to_use=remaining_dims,
            scored_data=scored,
            window_ids=window_ids,
            paths=paths,
            output_dir=output_dir
        )
        results[f'without_{dim_to_remove}'] = exp_result
    
    # ------------------------------------------------------------------------
    # STEP 3: Generate comparison results
    # ------------------------------------------------------------------------
    print(f"\n[3/4] Generating comparison results...")
    comparison_df = generate_comparison_table(results)
    
    # Save results
    results_file = output_dir / 'ablation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   ✓ Saved: {results_file}")
    
    comparison_file = output_dir / 'ablation_comparison.csv'
    comparison_df.to_csv(comparison_file, index=False)
    print(f"   ✓ Saved: {comparison_file}")
    
    # ------------------------------------------------------------------------
    # STEP 4: Generate visualizations
    # ------------------------------------------------------------------------
    print(f"\n[4/4] Generating visualizations...")
    generate_visualizations(comparison_df, output_dir)
    
    print("\n" + "="*70)
    print("✓ ABLATION STUDY COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir.resolve()}")
    print("\nGenerated files:")
    print("  - ablation_results.json          (full results)")
    print("  - ablation_comparison.csv        (comparison table)")
    print("  - ablation_mae_comparison.png    (MAE bar chart)")
    print("  - ablation_metrics_heatmap.png   (metrics heatmap)")
    print("  - ablation_delta_plot.png        (delta from baseline)")
    
    return results, comparison_df


def run_single_experiment(experiment_name: str, dimensions_to_use: List[str],
                          scored_data: dict, window_ids: List[str], 
                          paths: List[Path], output_dir: Path) -> dict:
    """
    Run a single ablation experiment:
    1. Build dimension dataframe with specified dimensions
    2. Fit PCA consolidator
    3. Generate labels
    4. Extract features
    5. Train XGBoost model
    6. Return metrics
    """
    
    start_time = time.time()
    
    # Step A: Build dimension dataframe
    dim_rows = []
    for wid in window_ids:
        payload = scored_data[wid]
        dim_rows.append(_flatten_dimension_results(payload["result"], dimensions_to_use))
    
    dim_df = pd.DataFrame(dim_rows, index=window_ids)
    
    # Step B: Fit PCA on this configuration
    pca = PCAConsolidator(good_is_high=True, round_ndecimals=2)
    pca.fit(dim_df)
    labels = pca.transform(dim_df)
    
    # Save PCA model
    pca_file = output_dir / f'{experiment_name}_pca.json'
    with open(pca_file, 'w') as f:
        json.dump(pca.to_dict(), f, indent=2)
    
    # Step C: Extract features (same for all experiments)
    feature_rows = []
    for idx, wid in enumerate(window_ids):
        from scoring.traditional_scorer import load_window_from_disk
        window_data = load_window_from_disk(paths[idx])
        feat_row = make_feature_row(window_data)
        feat_row['label'] = float(labels[idx])
        feat_row['window_id'] = wid
        feature_rows.append(feat_row)
    
    dataset = pd.DataFrame(feature_rows)
    
    # Enforce feature schema if available
    schema = load_feature_schema()
    if schema:
        dataset = dataset.reindex(columns=schema, fill_value=0.0)
    
    # Save labeled dataset
    labels_file = output_dir / f'{experiment_name}_labels.parquet'
    dataset.to_parquet(labels_file, index=False)
    
    # Step D: Prepare training data
    y = dataset['label'].astype(float)
    X = dataset.drop(columns=['label', 'window_id'], errors='ignore')
    X = X.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    
    # Drop bad columns
    bad_cols = [c for c in X.columns if X[c].isna().all() or X[c].nunique(dropna=True) <= 1]
    if bad_cols:
        X = X.drop(columns=bad_cols)
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=CONFIG['test_size'], random_state=42
    )
    
    # Optional scaling
    if CONFIG['use_scaler']:
        median_vals = X_train.median(numeric_only=True)
        X_train = X_train.fillna(median_vals)
        X_val = X_val.fillna(median_vals)
        
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
    
    # Step E: Train XGBoost
    model, metrics = train_xgboost_model(
        X_train, y_train, X_val, y_val,
        params=CONFIG['xgb_params'],
        num_rounds=CONFIG['num_boost_rounds'],
        early_stop=CONFIG['early_stopping_rounds']
    )
    
    elapsed = time.time() - start_time
    
    # Return results
    result = {
        'experiment': experiment_name,
        'dimensions_used': dimensions_to_use,
        'num_dimensions': len(dimensions_to_use),
        'metrics': metrics,
        'pca_explained_variance': float(pca.explained_variance_ratio_) if hasattr(pca, 'explained_variance_ratio_') else None,
        'num_features': int(X.shape[1]),
        'num_windows': int(len(dataset)),
        'elapsed_seconds': float(elapsed),
        'files': {
            'pca_model': str(pca_file),
            'labels': str(labels_file)
        }
    }
    
    print(f"     MAE: {metrics['mae']:.4f} | R²: {metrics['r2']:.4f} | Time: {elapsed:.1f}s")
    
    return result


def generate_comparison_table(results: dict) -> pd.DataFrame:
    """Generate comparison table from ablation results"""
    
    rows = []
    baseline_mae = results['baseline']['metrics']['mae']
    
    for exp_name, exp_data in results.items():
        metrics = exp_data['metrics']
        
        # Calculate delta from baseline
        delta_mae = metrics['mae'] - baseline_mae
        delta_r2 = metrics['r2'] - results['baseline']['metrics']['r2']
        
        # Determine which dimension was removed
        if exp_name == 'baseline':
            removed_dim = 'None'
            config = 'All 5 dimensions'
        else:
            removed_dim = exp_name.replace('without_', '').upper()
            config = f'w/o {removed_dim}'
        
        rows.append({
            'Configuration': config,
            'Removed_Dimension': removed_dim,
            'Num_Dimensions': exp_data['num_dimensions'],
            'MAE': metrics['mae'],
            'R²': metrics['r2'],
            'RMSE': metrics['rmse'],
            'Delta_MAE': delta_mae,
            'Delta_R²': delta_r2,
            'Best_Iteration': metrics['best_iteration'],
            'Elapsed_Sec': exp_data['elapsed_seconds']
        })
    
    df = pd.DataFrame(rows)
    
    # Sort by MAE (best to worst)
    df = df.sort_values('MAE')
    
    return df


def generate_visualizations(comparison_df: pd.DataFrame, output_dir: Path):
    """Generate visualization plots"""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # 1. MAE Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = comparison_df.sort_values('MAE', ascending=False)
    colors = ['green' if x == 'All 5 dimensions' else 'orange' for x in data['Configuration']]
    
    bars = ax.barh(data['Configuration'], data['MAE'], color=colors, alpha=0.7)
    ax.set_xlabel('Mean Absolute Error (MAE)', fontsize=12)
    ax.set_title('Ablation Study: MAE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax.axvline(comparison_df[comparison_df['Configuration'] == 'All 5 dimensions']['MAE'].values[0], 
               color='red', linestyle='--', linewidth=2, label='Baseline')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontsize=10)
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_mae_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Delta from Baseline Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ablation_data = comparison_df[comparison_df['Configuration'] != 'All 5 dimensions'].copy()
    ablation_data = ablation_data.sort_values('Delta_MAE', ascending=False)
    
    colors = ['red' if x > 0 else 'green' for x in ablation_data['Delta_MAE']]
    bars = ax.barh(ablation_data['Removed_Dimension'], ablation_data['Delta_MAE'], color=colors, alpha=0.7)
    
    ax.set_xlabel('Δ MAE from Baseline (Higher = Worse Performance)', fontsize=12)
    ax.set_title('Impact of Removing Each Dimension', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:+.4f}', ha='left' if width > 0 else 'right', 
                va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_delta_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Metrics Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    heatmap_data = comparison_df[['Configuration', 'MAE', 'R²', 'RMSE']].set_index('Configuration')
    
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Metric Value'}, ax=ax)
    ax.set_title('Ablation Study: All Metrics Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Generated 3 visualization plots")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ablation study on quality dimensions')
    parser.add_argument('--windows_dir', type=str, help='Path to windows directory')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    
    args = parser.parse_args()
    
    # Override config if arguments provided
    if args.windows_dir:
        CONFIG['windows_dir'] = args.windows_dir
    if args.output_dir:
        CONFIG['output_dir'] = args.output_dir
    
    # Run ablation study
    results, comparison = run_ablation_study()
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(comparison.to_string(index=False))
    print("="*70)