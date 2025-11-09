#!/usr/bin/env python3
"""
FIXED Per-Dimension Analysis with Normalization
==============================================
This script fixes the scale mismatch problem by normalizing
dimension MPR scores to [0,1] before comparison with PCA labels.

Usage: python dimension_analysis_fixed.py
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'input_csv': './results/dimension_analysis/dimension_scores.csv',
    'output_dir': './results/dimension_analysis_fixed',
    'quality_threshold': 0.5,
    'dimensions': ['timeliness', 'completeness', 'validity', 'consistency', 'accuracy']
}

# ============================================================================
# STEP 1: Load and Normalize Data
# ============================================================================

def load_and_normalize():
    """Load dimension scores and normalize to [0, 1]."""
    print("\n" + "="*70)
    print("FIXED PER-DIMENSION ANALYSIS (WITH NORMALIZATION)")
    print("="*70)
    
    input_path = Path('./results/dimension_analysis/dimension_scores_with_details.pkl')    
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    
    df = pd.read_pickle(input_path)
    print(f"✓ Loaded {len(df)} windows")
    details_cols = [c for c in df.columns if '_details' in c]
    print(f"✓ Details columns: {len(details_cols)}")
    
    # Normalize each dimension to [0, 1]
    for dim in CONFIG['dimensions']:
        mpr_col = f'{dim}_mpr'
        norm_col = f'{dim}_norm'
        
        if mpr_col not in df.columns:
            print(f"✗ {dim}: MPR column not found")
            continue
        
        # Min-max normalization
        min_val = df[mpr_col].min()
        max_val = df[mpr_col].max()
        df[norm_col] = (df[mpr_col] - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        
        print(f"✓ {dim.upper()}: normalized [{min_val:.3f}, {max_val:.3f}] → [0.0, 1.0]")
    
    return df


# ============================================================================
# STEP 2: Calculate Metrics on Normalized Scores
# ============================================================================

def calculate_normalized_metrics(df):
    """Calculate metrics using normalized dimension scores."""
    print("\n" + "="*70)
    print("CALCULATING METRICS (NORMALIZED SCORES)")
    print("="*70)
    
    threshold = CONFIG['quality_threshold']
    dimensions = CONFIG['dimensions']
    
    metrics_summary = {}
    
    for dim in dimensions:
        norm_col = f'{dim}_norm'
        
        if norm_col not in df.columns:
            print(f"✗ {dim}: Normalized column not found")
            continue
        
        # Get normalized scores and PCA labels
        dim_norm = df[norm_col].values
        pca_labels = df['pca_label'].values
        
        # Remove NaN values
        valid_mask = ~np.isnan(dim_norm) & ~np.isnan(pca_labels)
        dim_norm_clean = dim_norm[valid_mask]
        pca_labels_clean = pca_labels[valid_mask]
        
        if len(dim_norm_clean) == 0:
            print(f"✗ {dim}: No valid scores")
            continue
        
        # Regression metrics
        mae = mean_absolute_error(pca_labels_clean, dim_norm_clean)
        rmse = np.sqrt(mean_squared_error(pca_labels_clean, dim_norm_clean))
        r2 = r2_score(pca_labels_clean, dim_norm_clean)
        correlation = np.corrcoef(pca_labels_clean, dim_norm_clean)[0, 1]
        
        # Binary classification
        dim_binary = (dim_norm_clean >= threshold).astype(int)
        pca_binary = (pca_labels_clean >= threshold).astype(int)
        
        cm = confusion_matrix(pca_binary, dim_binary)
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_summary[dim] = {
            'dimension': dim,
            'n_windows': int(len(dim_norm_clean)),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'correlation': float(correlation),
            'confusion_matrix': cm.tolist(),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
        print(f"\n{dim.upper()}:")
        print(f"  MAE: {mae:.4f} | R²: {r2:.3f} | Corr: {correlation:.3f}")
        print(f"  Accuracy: {accuracy:.3f} | F1: {f1:.3f}")
        print(f"  Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    return metrics_summary


# ============================================================================
# STEP 3: Generate Visualizations
# ============================================================================

def generate_visualizations(df, metrics_summary, output_dir):
    """Generate comparison plots."""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dimensions = CONFIG['dimensions']
    
    # 1. Confusion Matrices Grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, dim in enumerate(dimensions):
        if dim not in metrics_summary:
            axes[idx].axis('off')
            continue
        
        cm = np.array(metrics_summary[dim]['confusion_matrix'])
        acc = metrics_summary[dim]['accuracy']
        
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], 
                   cmap='Blues', cbar=False,
                   xticklabels=['Poor', 'Good'],
                   yticklabels=['Poor', 'Good'])
        
        axes[idx].set_title(f'{dim.upper()}\n(Acc: {acc:.3f})')
        axes[idx].set_xlabel('Predicted (Normalized Dim)')
        axes[idx].set_ylabel('Ground Truth (PCA Label)')
    
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices_normalized.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: confusion_matrices_normalized.png")
    plt.close()
    
    # 2. MAE Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dims_sorted = sorted(metrics_summary.keys(), 
                        key=lambda x: metrics_summary[x]['mae'])
    
    mae_values = [metrics_summary[d]['mae'] for d in dims_sorted]
    colors = ['#d4edda' if mae < 0.15 else '#f8d7da' for mae in mae_values]
    
    ax.barh(range(len(dims_sorted)), mae_values, color=colors)
    ax.set_yticks(range(len(dims_sorted)))
    ax.set_yticklabels([d.upper() for d in dims_sorted])
    ax.set_xlabel('MAE (Lower is Better)')
    ax.set_title('Per-Dimension MAE Comparison (Normalized Scores)')
    ax.axvline(x=0.15, color='red', linestyle='--', linewidth=2, label='Good threshold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mae_comparison_normalized.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: mae_comparison_normalized.png")
    plt.close()
    
    # 3. Scatter Plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, dim in enumerate(dimensions):
        if dim not in metrics_summary:
            axes[idx].axis('off')
            continue
        
        norm_col = f'{dim}_norm'
        x = df[norm_col].dropna()
        y = df.loc[x.index, 'pca_label']
        
        axes[idx].scatter(x, y, alpha=0.5, s=20)
        axes[idx].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect correlation')
        
        # Add regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        axes[idx].plot(x.sort_values(), p(x.sort_values()), "b-", linewidth=2, alpha=0.8, label='Fit')
        
        r2 = metrics_summary[dim]['r2']
        corr = metrics_summary[dim]['correlation']
        axes[idx].set_title(f'{dim.upper()}\n(R²={r2:.3f}, Corr={corr:.3f})')
        axes[idx].set_xlabel('Normalized Dimension Score')
        axes[idx].set_ylabel('PCA Label')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
    
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_plots_normalized.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: scatter_plots_normalized.png")
    plt.close()
    
    # 4. Metrics Summary Table
    fig, ax = plt.subplots(figsize=(14, 6))
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
            f"{m['correlation']:.3f}",
            f"{m['accuracy']:.3f}",
            f"{m['precision']:.3f}",
            f"{m['recall']:.3f}",
            f"{m['f1_score']:.3f}"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Dimension', 'MAE', 'R²', 'Correlation', 'Accuracy', 'Precision', 'Recall', 'F1'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.12, 0.10, 0.10, 0.12, 0.12, 0.12, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code cells
    for i in range(1, len(table_data) + 1):
        # MAE column
        mae_val = float(table_data[i-1][1])
        mae_cell = table[(i, 1)]
        mae_cell.set_facecolor('#d4edda' if mae_val < 0.15 else '#fff3cd' if mae_val < 0.25 else '#f8d7da')
        
        # R² column
        r2_val = float(table_data[i-1][2])
        r2_cell = table[(i, 2)]
        r2_cell.set_facecolor('#d4edda' if r2_val > 0.7 else '#fff3cd' if r2_val > 0.5 else '#f8d7da')
    
    plt.savefig(output_dir / 'metrics_summary_table_normalized.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: metrics_summary_table_normalized.png")
    plt.close()


# ============================================================================
# STEP 4: Extract Reason Codes
# ============================================================================

def extract_reason_codes(scored_df: pd.DataFrame):
    """Extract failure reason codes from dimension details."""
    print("\n" + "="*70)
    print("EXTRACTING REASON CODES")
    print("="*70)
    
    dimensions = CONFIG['dimensions']
    threshold = CONFIG['quality_threshold']
    
    reason_codes = {'overall': {}}
    for dim in dimensions:
        reason_codes[dim] = {}
    
    print(f"Analyzing {len(scored_df)} windows...")
    
    for idx, row in scored_df.iterrows():
        pca_label = row['pca_label']
        
        # Only extract from low-quality windows
        if pca_label >= threshold:
            continue
        
        for dim in dimensions:
            mpr_col = f'{dim}_mpr'
            details_col = f'{dim}_details'
            print(f"DEBUG: Details columns found: {details_col}")
            
            if mpr_col not in row or details_col not in row:
                continue
            
            dim_mpr = row[mpr_col]
            
            # Low dimension score = failure
            if dim_mpr <= 0.95:
                try:
                    details = row[details_col]
                    
                    # Parse if string (CSV stores as string)
                    if isinstance(details, str):
                        import ast
                        details = ast.literal_eval(details)
                    
                    # Extract specific reasons
                    reasons = _parse_dimension_failures(dim, details)
                    
                    for reason in reasons:
                        if reason:
                            reason_codes[dim][reason] = reason_codes[dim].get(reason, 0) + 1
                            reason_codes['overall'][f'{dim}:{reason}'] = reason_codes['overall'].get(f'{dim}:{reason}', 0) + 1
                
                except Exception as e:
                    continue
    
    print(f"✓ Extracted {sum(len(v) for v in reason_codes.values())} failure patterns")
    
    return reason_codes

def _parse_dimension_failures(dimension: str, details: dict) -> list:
    """Parse dimension details to extract specific failure reasons."""
    reasons = []
    
    if not isinstance(details, dict):
        return reasons
    
    try:
        if dimension == 'timeliness':
            # T1: Inter-arrival
            t1 = details.get('T1_interarrival', {})
            if isinstance(t1, dict):
                app = t1.get('applicable', 0)
                passed = t1.get('passed', 0)
                if app > 0 and passed < app * 0.80:
                    reasons.append('T1_interarrival_issues')
            
            # T2: Grid alignment  
            t2 = details.get('T2_grid_alignment', {})
            if isinstance(t2, dict):
                exp = t2.get('expected_grid_points', 0)
                aligned = t2.get('aligned_slots', 0)
                if exp > 0 and aligned < exp * 0.70:
                    reasons.append('T2_poor_grid_coverage')
            
            # T3: Monotonicity
            t3 = details.get('T3_monotonicity', {})
            if isinstance(t3, dict):
                app = t3.get('applicable', 0)
                passed = t3.get('passed', 0)
                if app > 0 and passed < app:
                    reasons.append('T3_timestamp_disorder')
        
        elif dimension == 'completeness':
            # C2: Entity presence
            c2 = details.get('entity_presence', {})
            if isinstance(c2, dict):
                ratio_cells = c2.get('ratio_cells', 1.0)
                ratio_ues = c2.get('ratio_ues', 1.0)
                if ratio_cells < 0.7:
                    reasons.append('C2_insufficient_cells')
                if ratio_ues < 0.6:
                    reasons.append('C2_low_ue_count')
            
            # C4: Field completeness
            c4 = details.get('field_completeness', {})
            if isinstance(c4, dict):
                total = c4.get('total_cells', 0)
                non_null = c4.get('non_null_cells', 0)
                if total > 0 and non_null < total * 0.6:
                    reasons.append('C4_sparse_data')
        
        elif dimension == 'validity':
            # V2: Range violations
            v2 = details.get('V2_ranges', {})
            if isinstance(v2, dict):
                app = v2.get('applicable', 0)
                passed = v2.get('passed', 0)
                if app > 0 and passed < app * 0.80:
                    reasons.append('V2_out_of_range')
            
            # V4: Business rules
            v4 = details.get('V4_business_rules', {})
            if isinstance(v4, dict):
                app = v4.get('applicable', 0)
                passed = v4.get('passed', 0)
                if app > 0 and passed < app * 0.90:
                    reasons.append('V4_prb_violations')
        
        elif dimension == 'consistency':
            # CS1: Intra-record
            cs1 = details.get('CS1_intra_record', {})
            if isinstance(cs1, dict):
                app = cs1.get('applicable', 0)
                passed = cs1.get('passed', 0)
                if app > 0 and passed < app * 0.90:
                    reasons.append('CS1_intra_inconsistency')
            
            # CS2: Energy identity
            cs2 = details.get('CS2_energy_identity', {})
            if isinstance(cs2, dict) and cs2:
                ratio = cs2.get('ratio')
                if ratio is not None and (ratio < 0.6 or ratio > 1.8):
                    reasons.append('CS2_energy_mismatch')
        
        elif dimension == 'accuracy':
            # A1: Efficiency bands
            a1 = details.get('A1_efficiency_bands', {})
            if isinstance(a1, dict):
                for side in ['dl', 'ul']:
                    sd = a1.get(side, {})
                    if isinstance(sd, dict) and sd.get('passed', 1) == 0:
                        reasons.append(f'A1_inefficient_{side}')
            
            # A2: Spectral efficiency
            a2 = details.get('A2_spectral_efficiency', {})
            if isinstance(a2, dict):
                for side in ['dl', 'ul']:
                    sd = a2.get(side, {})
                    if isinstance(sd, dict):
                        app = sd.get('applicable', 0)
                        passed = sd.get('passed', 0)
                        if app > 0 and passed < app * 0.90:
                            reasons.append(f'A2_spectral_cap_{side}')
    
    except Exception:
        pass
    
    return reasons

# ============================================================================
# STEP 4: Save Results
# ============================================================================

def save_results(df, metrics_summary,reason_codes ,output_dir):
    """Save results to files."""
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save normalized dataframe
    df.to_csv(output_dir / 'dimension_scores_normalized.csv', index=False)
    print(f"✓ Saved: dimension_scores_normalized.csv")
    
    # Save metrics
    with open(output_dir / 'metrics_summary_normalized.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"✓ Saved: metrics_summary_normalized.json")

    # Save reason codes
    with open(output_dir / 'reason_codes.json', 'w') as f:
        json.dump(reason_codes, f, indent=2)
    print(f"✓ Saved: reason_codes.json")
    
    # Generate report
    with open(output_dir / 'analysis_report_normalized.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("FIXED PER-DIMENSION ANALYSIS REPORT\n")
        f.write("(Normalized Scores: Scale-Matched Comparison)\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total Windows: {len(df)}\n")
        f.write(f"Quality Threshold: {CONFIG['quality_threshold']}\n\n")
        
        f.write("KEY INSIGHT:\n")
        f.write("Dimension MPR scores have been normalized to [0,1] to match\n")
        f.write("the PCA label scale. This enables fair comparison and meaningful\n")
        f.write("confusion matrices.\n\n")
        
        f.write("="*70 + "\n")
        f.write("DIMENSION-WISE METRICS (NORMALIZED)\n")
        f.write("="*70 + "\n\n")
        
        for dim in CONFIG['dimensions']:
            if dim not in metrics_summary:
                continue
            
            m = metrics_summary[dim]
            f.write(f"\n{dim.upper()}\n")
            f.write("-" * 50 + "\n")
            f.write(f"  Regression Metrics:\n")
            f.write(f"    MAE:         {m['mae']:.4f}\n")
            f.write(f"    RMSE:        {m['rmse']:.4f}\n")
            f.write(f"    R²:          {m['r2']:.3f}\n")
            f.write(f"    Correlation: {m['correlation']:.3f}\n\n")
            
            f.write(f"  Classification Metrics:\n")
            f.write(f"    Accuracy:  {m['accuracy']:.3f}\n")
            f.write(f"    Precision: {m['precision']:.3f}\n")
            f.write(f"    Recall:    {m['recall']:.3f}\n")
            f.write(f"    F1-Score:  {m['f1_score']:.3f}\n\n")
            
            f.write(f"  Confusion Matrix:\n")
            f.write(f"    TN: {m['tn']}  FP: {m['fp']}\n")
            f.write(f"    FN: {m['fn']}  TP: {m['tp']}\n\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("INTERPRETATION GUIDELINES\n")
        f.write("="*70 + "\n\n")
        f.write("MAE < 0.15: Excellent dimension predictor\n")
        f.write("MAE < 0.25: Good dimension predictor\n")
        f.write("MAE > 0.25: Weak dimension predictor\n\n")
        f.write("R² > 0.7: Strong linear relationship\n")
        f.write("R² > 0.5: Moderate relationship\n")
        f.write("R² < 0.5: Weak relationship\n\n")
        if 'overall' in reason_codes:
            top_reasons = sorted(reason_codes['overall'].items(), 
                               key=lambda x: x[1], reverse=True)[:10]
            for reason, count in top_reasons:
                f.write(f"  {reason}: {count} windows\n")
    
    print(f"✓ Saved: analysis_report_normalized.txt")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution."""
    try:
        # Step 1: Load and normalize
        df = load_and_normalize()
        
        # Step 2: Calculate metrics
        metrics_summary = calculate_normalized_metrics(df)
        
        if not metrics_summary:
            raise RuntimeError("No metrics calculated!")
        
        # Step 4: Extract reason codes
        reason_codes = extract_reason_codes(df)
        
        # Step 3: Generate visualizations
        output_dir = Path(CONFIG['output_dir'])
        generate_visualizations(df, metrics_summary, output_dir)

        #reason_codes = extract_reason_codes(df) 
        
        # Step 4: Save results
        save_results(df, metrics_summary,reason_codes, output_dir)
        
        print("\n" + "="*70)
        print("✓ FIXED ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\nResults saved to: {output_dir.resolve()}")
        print("\nGenerated files:")
        print("  - dimension_scores_normalized.csv")
        print("  - metrics_summary_normalized.json")
        print("  - analysis_report_normalized.txt")
        print("  - confusion_matrices_normalized.png")
        print("  - mae_comparison_normalized.png")
        print("  - scatter_plots_normalized.png")
        print("  - metrics_summary_table_normalized.png")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())