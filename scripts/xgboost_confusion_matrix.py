#!/usr/bin/env python3
"""
Generate XGBoost Model Confusion Matrix
========================================
Compares XGBoost predictions vs PCA ground truth labels.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import joblib
from pathlib import Path
import json

# Configuration
CONFIG = {
    'model_path': './data/artifacts/models/v1.3.0_300s/20251021_235803_6907daa9a7/model.bin',
    'model_dir': './data/artifacts/models/v1.3.0_300s/20251021_235803_6907daa9a7',
    'test_data_path': './data/processed/training/v0/train_labels_300s.parquet',
    'output_dir': './results/xgboost_confusion',
    'threshold': 0.5
}

def main():
    print("="*70)
    print("XGBOOST CONFUSION MATRIX GENERATION")
    print("="*70)
    
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model metadata to get feature list
    print("\n1. Loading model metadata...")
    model_dir = Path(CONFIG['model_path']).parent
    meta_path = model_dir / 'meta.json'
    
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    drop_cols = meta.get('drop_cols', ['window_id'])
    train_features = [f for f in meta.get('features', []) 
                     if f not in drop_cols]
    
    print(f"   ✓ Expected features: {len(train_features)}")
    

    # Load test data
    print("\n1. Loading test data...")   
    test_df = pd.read_parquet(CONFIG['test_data_path'])
    n = len(test_df)
    split_idx = int(n * 0.80)     
    test_df = test_df.iloc[split_idx:].reset_index(drop=True)
    print(f"   ✓ Loaded {len(test_df)} test windows")
    
    # Separate features and labels
    for col in train_features:
        if col not in test_df.columns:
            test_df[col] = 0.0
    
    X_test = test_df[train_features].copy() 
    y_test = test_df['label']
    
    print(f"   ✓ Aligned features: {len(X_test.columns)}")
    
    #print(f"   ✓ Features: {len(feature_cols)}")
    print(f"   ✓ Labels: min={y_test.min():.3f}, max={y_test.max():.3f}")
    
    # Load model
    print("\n2. Loading XGBoost model...")
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model(CONFIG['model_path'])
    print(f"   ✓ Model loaded")
    
    # Make predictions
    print("\n3. Making predictions...")
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    print(f"   ✓ Predictions: min={y_pred.min():.3f}, max={y_pred.max():.3f}")
    
    # Binarize at threshold
    threshold = CONFIG['threshold']
    y_test_binary = (y_test >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Confusion matrix
    print(f"\n4. Generating confusion matrix (threshold={threshold})...")
    cm = confusion_matrix(y_test_binary, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"   ✓ Confusion Matrix:")
    print(f"      TN={tn}, FP={fp}")
    print(f"      FN={fn}, TP={tp}")
    print(f"   ✓ Accuracy:  {accuracy:.3f}")
    print(f"   ✓ Precision: {precision:.3f}")
    print(f"   ✓ Recall:    {recall:.3f}")
    print(f"   ✓ F1-Score:  {f1:.3f}")
    
    # Save metrics
    metrics = {
        'confusion_matrix': cm.tolist(),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'threshold': threshold,
        'n_test': len(y_test)
    }
    
    with open(output_dir / 'xgboost_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n   ✓ Saved: xgboost_metrics.json")
    
    # Visualize confusion matrix
    print("\n5. Creating visualizations...")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Poor', 'Good'],
                yticklabels=['Poor', 'Good'],
                cbar=False, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth (PCA Label)')
    ax.set_title(f'XGBoost Confusion Matrix\n(Acc: {accuracy:.3f}, F1: {f1:.3f})')
    plt.tight_layout()
    plt.savefig(output_dir / 'xgboost_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: xgboost_confusion_matrix.png")
    plt.close()
    
    # Generate report
    with open(output_dir / 'xgboost_report.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("XGBOOST MODEL CONFUSION MATRIX REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Test Windows: {len(y_test)}\n")
        f.write(f"Threshold: {threshold}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  TN: {tn}  FP: {fp}\n")
        f.write(f"  FN: {fn}  TP: {tp}\n\n")
        f.write("Classification Metrics:\n")
        f.write(f"  Accuracy:  {accuracy:.3f}\n")
        f.write(f"  Precision: {precision:.3f}\n")
        f.write(f"  Recall:    {recall:.3f}\n")
        f.write(f"  F1-Score:  {f1:.3f}\n\n")
        f.write("Interpretation:\n")
        f.write(f"  False Positive Rate: {fp/(fp+tn)*100:.1f}%\n")
        f.write(f"  False Negative Rate: {fn/(fn+tp)*100:.1f}%\n")
    
    print(f"   ✓ Saved: xgboost_report.txt")
    
    print("\n" + "="*70)
    print("✓ XGBOOST CONFUSION MATRIX GENERATED!")
    print("="*70)
    print(f"\nResults in: {output_dir.resolve()}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())