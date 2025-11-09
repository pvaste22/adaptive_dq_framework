#!/usr/bin/env python3
"""
Complete Per-Dimension Analysis Script
=====================================
Single script to extract all per-dimension metrics, confusion matrices, 
MAE, reason codes WITHOUT modifying existing code.

Usage: python confusion_dim.py
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

# Add src to path
#PROJECT_ROOT = Path(__file__).resolve().parent
#sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from scoring.traditional_scorer import score_window
from quality_dimensions.completeness import CompletenessDimension
from quality_dimensions.timeliness import TimelinessDimension
from quality_dimensions.validity import ValidityDimension
from quality_dimensions.consistency import ConsistencyDimension
from quality_dimensions.accuracy import AccuracyDimension

# ============================================================================
# CONFIGURATION - Modify these paths as needed
# ============================================================================

CONFIG = {
    'labeled_data_path': './data/processed/training/v0/train_labels_300s.parquet',
    'windows_dir': './data/processed/training/v0/windows_300s',
    'output_dir': './results/dimension_analysis',
    'quality_threshold': 0.5,  # Binary classification threshold
    'dimensions': ['timeliness', 'completeness', 'validity', 'consistency', 'accuracy']
}

# ============================================================================
# STEP 1: Load Labeled Dataset and Window Mappings
# ============================================================================

def load_labeled_data():
    """Load the labeled dataset with PCA-based ground truth labels."""
    print("\n" + "="*70)
    print("STEP 1: Loading Labeled Dataset")
    print("="*70)
    
    labeled_path = Path(CONFIG['labeled_data_path'])
    if not labeled_path.exists():
        raise FileNotFoundError(f"Labeled data not found: {labeled_path}")
    
    df = pd.read_parquet(labeled_path)
    print(f"✓ Loaded {len(df)} labeled windows")
    print(f"✓ Columns: {df.columns.tolist()[:10]}...")
    print(f"✓ Label stats: min={df['label'].min():.3f}, max={df['label'].max():.3f}, mean={df['label'].mean():.3f}")
    
    return df


# ============================================================================
# STEP 2: Score All Windows with Traditional Scorer
# ============================================================================

def score_all_windows(labeled_df: pd.DataFrame):
    """Re-score all windows to get per-dimension MPR values."""
    print("\n" + "="*70)
    print("STEP 2: Scoring Windows with Traditional Scorer")
    print("="*70)
    
    windows_dir = Path(CONFIG['windows_dir'])
    if not windows_dir.exists():
        raise FileNotFoundError(f"Windows directory not found: {windows_dir}")
    
    # Initialize dimensions (reused across windows to cache baselines)
    dimensions = [
        TimelinessDimension(),
        CompletenessDimension(),
        ValidityDimension(),
        ConsistencyDimension(),
        AccuracyDimension()
    ]
    
    results = []
    failed_windows = []
    
    print(f"Processing {len(labeled_df)} windows...")
    
    for idx, row in tqdm(labeled_df.iterrows(), total=len(labeled_df), desc="Scoring"):
        window_id = row['window_id']
        pca_label = row['label']
        
        # Find window directory
        window_path = windows_dir / window_id
        if not window_path.exists():
            failed_windows.append((window_id, "path_not_found"))
            continue
        
        try:
            # Score window
            dim_results = score_window(window_path, dimensions=dimensions)
            
            # Extract MPR scores
            record = {
                'window_id': window_id,
                'pca_label': pca_label
            }
            
            for dim_name, dim_result in dim_results.items():
                mpr = dim_result.get('score', dim_result.get('mpr', np.nan))
                apr = dim_result.get('apr', np.nan)
                coverage = dim_result.get('coverage', np.nan)
                
                record[f'{dim_name}_mpr'] = mpr
                record[f'{dim_name}_apr'] = apr
                record[f'{dim_name}_coverage'] = coverage
                record[f'{dim_name}_details'] = dim_result.get('details', {})
            
            results.append(record)
            
        except Exception as e:
            failed_windows.append((window_id, str(e)))
            print(f"\n✗ Failed to score {window_id}: {e}")
    
    print(f"\n✓ Successfully scored: {len(results)}/{len(labeled_df)} windows")
    if failed_windows:
        print(f"✗ Failed windows: {len(failed_windows)}")
        print("  First 5 failures:")
        for wid, err in failed_windows[:5]:
            print(f"    - {wid}: {err}")
    
    return pd.DataFrame(results), failed_windows


# ============================================================================
# STEP 3: Calculate Per-Dimension Metrics
# ============================================================================

def calculate_dimension_metrics(scored_df: pd.DataFrame):
    """Calculate confusion matrix, MAE, RMSE, R² for each dimension."""
    print("\n" + "="*70)
    print("STEP 3: Calculating Per-Dimension Metrics")
    print("="*70)
    
    threshold = CONFIG['quality_threshold']
    dimensions = CONFIG['dimensions']
    
    metrics_summary = {}
    
    for dim in dimensions:
        mpr_col = f'{dim}_mpr'
        
        if mpr_col not in scored_df.columns:
            print(f"✗ {dim}: MPR column not found")
            continue
        
        # Get dimension MPRs and PCA labels
        dim_scores = scored_df[mpr_col].values
        pca_labels = scored_df['pca_label'].values
        
        # Remove NaN values
        valid_mask = ~np.isnan(dim_scores) & ~np.isnan(pca_labels)
        dim_scores_clean = dim_scores[valid_mask]
        pca_labels_clean = pca_labels[valid_mask]
        
        if len(dim_scores_clean) == 0:
            print(f"✗ {dim}: No valid scores")
            continue
        
        # Regression metrics (MPR as continuous predictor of PCA label)
        mae = mean_absolute_error(pca_labels_clean, dim_scores_clean)
        rmse = np.sqrt(mean_squared_error(pca_labels_clean, dim_scores_clean))
        r2 = r2_score(pca_labels_clean, dim_scores_clean)
        
        # Binary classification (at threshold)
        dim_binary = (dim_scores_clean >= threshold).astype(int)
        pca_binary = (pca_labels_clean >= threshold).astype(int)
        
        cm = confusion_matrix(pca_binary, dim_binary)
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_summary[dim] = {
            'dimension': dim,
            'n_windows': int(len(dim_scores_clean)),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'confusion_matrix': cm.tolist(),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'mean_mpr': float(dim_scores_clean.mean()),
            'std_mpr': float(dim_scores_clean.std()),
            'correlation_with_pca': float(np.corrcoef(dim_scores_clean, pca_labels_clean)[0, 1])
        }
        
        print(f"\n{dim.upper()}:")
        print(f"  MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
        print(f"  Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
        print(f"  Confusion Matrix:\n{cm}")
    
    return metrics_summary


# ============================================================================
# STEP 4: Extract Reason Codes
# ============================================================================

def extract_reason_codes(scored_df: pd.DataFrame):
    """Extract detailed reason codes from dimension details."""
    print("\n" + "="*70)
    print("STEP 4: Extracting Reason Codes")
    print("="*70)
    
    dimensions = CONFIG['dimensions']
    threshold = CONFIG['quality_threshold']
    
    reason_code_stats = defaultdict(lambda: defaultdict(int))
    
    for dim in dimensions:
        mpr_col = f'{dim}_mpr'
        details_col = f'{dim}_details'
        
        if details_col not in scored_df.columns:
            continue
        
        # Find windows where this dimension failed (MPR < threshold)
        failed_mask = scored_df[mpr_col] < threshold
        failed_windows = scored_df[failed_mask]
        
        print(f"\n{dim.upper()}: {len(failed_windows)} failures")
        
        for idx, row in failed_windows.iterrows():
            details = row[details_col]
            if not isinstance(details, dict):
                continue
            
            # Extract sub-check failures
            for check_name, check_data in details.items():
                if not isinstance(check_data, dict):
                    continue
                
                # Check for failures
                passed = check_data.get('passed', 0)
                applicable = check_data.get('applicable', 0)
                
                if applicable > 0 and passed < applicable:
                    fail_rate = (applicable - passed) / applicable
                    reason_code = f"{dim}_{check_name}"
                    reason_code_stats[dim][reason_code] += 1
                    reason_code_stats['overall'][reason_code] += 1
        
        # Print top 5 reasons for this dimension
        if reason_code_stats[dim]:
            top_reasons = sorted(reason_code_stats[dim].items(), 
                               key=lambda x: x[1], reverse=True)[:5]
            for reason, count in top_reasons:
                print(f"  - {reason}: {count} windows")
    
    return dict(reason_code_stats)


# ============================================================================
# STEP 5: Generate Visualizations
# ============================================================================

def generate_visualizations(scored_df: pd.DataFrame, metrics_summary: Dict, output_dir: Path):
    """Generate all plots and confusion matrices."""
    print("\n" + "="*70)
    print("STEP 5: Generating Visualizations")
    print("="*70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    dimensions = CONFIG['dimensions']
    threshold = CONFIG['quality_threshold']
    
    # 1. Confusion Matrices Grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, dim in enumerate(dimensions):
        if dim not in metrics_summary:
            continue
        
        cm = np.array(metrics_summary[dim]['confusion_matrix'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Poor', 'Good'],
                   yticklabels=['Poor', 'Good'],
                   ax=axes[idx], cbar=True)
        
        axes[idx].set_title(f'{dim.upper()}\n(Acc: {metrics_summary[dim]["accuracy"]:.3f})')
        axes[idx].set_xlabel('Predicted (Dimension MPR)')
        axes[idx].set_ylabel('Ground Truth (PCA Label)')
    
    # Hide extra subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices_grid.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: confusion_matrices_grid.png")
    plt.close()
    
    # 2. MAE Comparison Bar Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dims_with_metrics = [d for d in dimensions if d in metrics_summary]
    maes = [metrics_summary[d]['mae'] for d in dims_with_metrics]
    colors = ['#2ecc71' if mae < 0.05 else '#e74c3c' for mae in maes]
    
    bars = ax.bar(dims_with_metrics, maes, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='Target MAE = 0.05')
    
    ax.set_xlabel('Quality Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Dimension Prediction Error (vs PCA Label)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mae_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: mae_comparison.png")
    plt.close()
    
    # 3. Scatter Plots: Dimension MPR vs PCA Label
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, dim in enumerate(dimensions):
        mpr_col = f'{dim}_mpr'
        
        if mpr_col not in scored_df.columns:
            continue
        
        valid_mask = scored_df[mpr_col].notna() & scored_df['pca_label'].notna()
        x = scored_df.loc[valid_mask, mpr_col]
        y = scored_df.loc[valid_mask, 'pca_label']
        
        axes[idx].scatter(x, y, alpha=0.5, s=20)
        axes[idx].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect correlation')
        
        # Add regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        axes[idx].plot(x.sort_values(), p(x.sort_values()), "b-", linewidth=2, alpha=0.8, label='Fit')
        
        r2 = metrics_summary[dim]['r2']
        axes[idx].set_title(f'{dim.upper()} (R²={r2:.3f})')
        axes[idx].set_xlabel('Dimension MPR')
        axes[idx].set_ylabel('PCA Label')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
    
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_plots.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: scatter_plots.png")
    plt.close()
    
    # 4. Metrics Summary Table Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for dim in dimensions:
        if dim not in metrics_summary:
            continue
        m = metrics_summary[dim]
        table_data.append([
            dim.upper(),
            f"{m['mae']:.4f}",
            f"{m['r2']:.3f}",
            f"{m['accuracy']:.3f}",
            f"{m['precision']:.3f}",
            f"{m['recall']:.3f}",
            f"{m['f1_score']:.3f}"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Dimension', 'MAE', 'R²', 'Accuracy', 'Precision', 'Recall', 'F1'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code MAE cells
    for i in range(1, len(table_data) + 1):
        mae_val = float(table_data[i-1][1])
        cell = table[(i, 1)]
        cell.set_facecolor('#d4edda' if mae_val < 0.05 else '#f8d7da')
    
    plt.savefig(output_dir / 'metrics_summary_table.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: metrics_summary_table.png")
    plt.close()


# ============================================================================
# STEP 6: Save All Results
# ============================================================================

def save_results(scored_df: pd.DataFrame, metrics_summary: Dict, 
                reason_codes: Dict, output_dir: Path):
    """Save all results to JSON and CSV files."""
    print("\n" + "="*70)
    print("STEP 6: Saving Results")
    print("="*70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save scored dataframe
    scored_csv = output_dir / 'dimension_scores.csv'
    # Drop details columns (too complex for CSV)
    scored_df_clean = scored_df.drop(columns=[c for c in scored_df.columns if '_details' in c])
    scored_df_clean.to_csv(scored_csv, index=False)
    print(f"✓ Saved: dimension_scores.csv ({len(scored_df_clean)} windows)")
    
    # Save FULL scored_df with details as pickle
    scored_pkl = output_dir / 'dimension_scores_with_details.pkl'
    with open(scored_pkl, 'wb') as f:
        pickle.dump(scored_df, f)
    print(f"✓ Saved with details: dimension_scores_with_details.pkl")

    # 2. Save metrics summary
    metrics_json = output_dir / 'metrics_summary.json'
    with open(metrics_json, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"✓ Saved: metrics_summary.json")
    
    # 3. Save reason codes
    reason_codes_json = output_dir / 'reason_codes.json'
    with open(reason_codes_json, 'w') as f:
        json.dump(reason_codes, f, indent=2)
    print(f"✓ Saved: reason_codes.json")
    
    # 4. Create human-readable summary report
    report_txt = output_dir / 'analysis_report.txt'
    with open(report_txt, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PER-DIMENSION QUALITY ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total Windows Analyzed: {len(scored_df)}\n")
        f.write(f"Quality Threshold: {CONFIG['quality_threshold']}\n\n")
        
        f.write("="*70 + "\n")
        f.write("DIMENSION-WISE METRICS\n")
        f.write("="*70 + "\n\n")
        
        for dim in CONFIG['dimensions']:
            if dim not in metrics_summary:
                continue
            
            m = metrics_summary[dim]
            f.write(f"\n{dim.upper()}\n")
            f.write("-" * 50 + "\n")
            f.write(f"  Regression Metrics:\n")
            f.write(f"    MAE:  {m['mae']:.4f}\n")
            f.write(f"    RMSE: {m['rmse']:.4f}\n")
            f.write(f"    R²:   {m['r2']:.3f}\n\n")
            
            f.write(f"  Classification Metrics (threshold={CONFIG['quality_threshold']}):\n")
            f.write(f"    Accuracy:  {m['accuracy']:.3f}\n")
            f.write(f"    Precision: {m['precision']:.3f}\n")
            f.write(f"    Recall:    {m['recall']:.3f}\n")
            f.write(f"    F1-Score:  {m['f1_score']:.3f}\n\n")
            
            f.write(f"  Confusion Matrix:\n")
            f.write(f"    TN: {m['tn']}  FP: {m['fp']}\n")
            f.write(f"    FN: {m['fn']}  TP: {m['tp']}\n\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("TOP FAILURE REASONS (Overall)\n")
        f.write("="*70 + "\n\n")
        
        if 'overall' in reason_codes:
            top_reasons = sorted(reason_codes['overall'].items(), 
                               key=lambda x: x[1], reverse=True)[:10]
            for reason, count in top_reasons:
                f.write(f"  {reason}: {count} windows\n")
    
    print(f"✓ Saved: analysis_report.txt")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    print("\n" + "="*70)
    print("PER-DIMENSION QUALITY ANALYSIS")
    print("="*70)
    print(f"Configuration:")
    print(f"  Labeled data: {CONFIG['labeled_data_path']}")
    print(f"  Windows dir:  {CONFIG['windows_dir']}")
    print(f"  Output dir:   {CONFIG['output_dir']}")
    print(f"  Threshold:    {CONFIG['quality_threshold']}")
    
    output_dir = Path(CONFIG['output_dir'])
    
    try:
        # Step 1: Load labeled data
        labeled_df = load_labeled_data()
        
        # Step 2: Score all windows
        scored_df, failed_windows = score_all_windows(labeled_df)
        
        if scored_df.empty:
            raise RuntimeError("No windows were successfully scored!")
        
        # Step 3: Calculate metrics
        metrics_summary = calculate_dimension_metrics(scored_df)
        
        # Step 4: Extract reason codes
        reason_codes = extract_reason_codes(scored_df)
        
        # Step 5: Generate visualizations
        generate_visualizations(scored_df, metrics_summary, output_dir)
        
        # Step 6: Save results
        save_results(scored_df, metrics_summary, reason_codes, output_dir)
        
        print("\n" + "="*70)
        print("✓ ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\nResults saved to: {output_dir.resolve()}")
        print("\nGenerated files:")
        print("  - dimension_scores.csv          (per-window scores)")
        print("  - metrics_summary.json          (all metrics)")
        print("  - reason_codes.json             (failure reasons)")
        print("  - analysis_report.txt           (human-readable)")
        print("  - confusion_matrices_grid.png   (visual)")
        print("  - mae_comparison.png            (visual)")
        print("  - scatter_plots.png             (visual)")
        print("  - metrics_summary_table.png     (visual)")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())