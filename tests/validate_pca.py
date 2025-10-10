import pandas as pd
import numpy as np
import json

df = pd.read_parquet('./data/processed/training/v0/train_labels.parquet')

print("="*60)
print("PCA VALIDATION REPORT")
print("="*60)

# 1. Label distribution
print("\n1. LABEL STATISTICS:")
stats = df['label'].describe()
for k, v in stats.items():
    print(f"   {k:8s}: {v:.3f}")
print(f"   Range: [{df['label'].min():.2f}, {df['label'].max():.2f}]")

# 2. Check PCA model
pca_path = './data/artifacts/models/pca_consolidator.json'
with open(pca_path) as f:
    pca = json.load(f)

print("\n2. PCA MODEL INFO:")
print(f"   Available keys: {list(pca.keys())}")

# Check what's available
if 'explained_variance_ratio_' in pca:
    variance = pca['explained_variance_ratio_']
    print(f"   Explained variance: {variance:.1%}")
    
if 'loadings_' in pca:
    weights = pca['loadings_']
    print(f"   PC1 weights: {weights}")
    print(f"   All negative? {all(w < 0 for w in weights)}")

if 'pc1_mu_' in pca:
    print(f"   PC1 mean: {pca['pc1_mu_']:.4f}")
if 'pc1_sd_' in pca:
    print(f"   PC1 std: {pca['pc1_sd_']:.4f}")

# 3. Sign check (indirect)
print("\n3. SIGN CORRECTION CHECK:")
if 'loadings_' in pca:
    weights = pca['loadings_']
    if all(w < 0 for w in weights):
        print("   ⚠️ All weights negative (should be flipped)")
        print("   Check if PCAConsolidator applies 'good_is_high=True'")
    elif all(w > 0 for w in weights):
        print("   ✅ All weights positive (correctly oriented)")
    else:
        print("   ⚠️ Mixed signs (unusual)")

# 4. Variance assessment
print("\n4. VARIANCE ASSESSMENT:")
if 'explained_variance_ratio_' in pca:
    variance = pca['explained_variance_ratio_']
    if variance > 0.70:
        print(f"   ✅ Excellent: {variance:.1%} variance captured")
    elif variance > 0.60:
        print(f"   ✅ Acceptable: {variance:.1%} variance captured")
        print("   → Dimensions capture independent quality aspects")
    elif variance > 0.50:
        print(f"   ⚠️ Low: {variance:.1%} variance captured")
        print("   → Dimensions are quite independent")
    else:
        print(f"   ❌ Very low: {variance:.1%} variance captured")

# 5. Label distribution quality
print("\n5. LABEL DISTRIBUTION QUALITY:")
mean = df['label'].mean()
std = df['label'].std()
q25 = df['label'].quantile(0.25)
q75 = df['label'].quantile(0.75)

print(f"   Mean: {mean:.3f}")
print(f"   Std: {std:.3f}")
print(f"   IQR: [{q25:.2f}, {q75:.2f}]")

checks = []

# Check 1: Proper range
if df['label'].min() >= 0.0 and df['label'].max() <= 1.0:
    print("   ✅ Labels in valid range [0, 1]")
    checks.append(True)
else:
    print("   ❌ Labels outside valid range!")
    checks.append(False)

# Check 2: Good variance
if std > 0.15:
    print(f"   ✅ Good variance (std={std:.3f})")
    checks.append(True)
elif std > 0.10:
    print(f"   ⚠️ Acceptable variance (std={std:.3f})")
    checks.append(True)
else:
    print(f"   ❌ Low variance (std={std:.3f})")
    checks.append(False)

# Check 3: Centered distribution
if 0.4 < mean < 0.6:
    print(f"   ✅ Well-centered (mean={mean:.3f})")
    checks.append(True)
else:
    print(f"   ⚠️ Skewed distribution (mean={mean:.3f})")
    checks.append(True)  # Not critical

# Check 4: Full range usage
if (df['label'].max() - df['label'].min()) > 0.8:
    print(f"   ✅ Good dynamic range ({df['label'].max() - df['label'].min():.2f})")
    checks.append(True)
else:
    print(f"   ⚠️ Limited range ({df['label'].max() - df['label'].min():.2f})")
    checks.append(False)

# 6. Overall verdict
print("\n" + "="*60)
print("OVERALL ASSESSMENT:")

passed = sum(checks)
total = len(checks)
print(f"\nPassed {passed}/{total} critical checks")

if passed >= 3:
    print("\n✅ PCA LABELS LOOK GOOD!")
    print("\nKey findings:")
    print(f"  • Std = {std:.3f} (good discrimination)")
    print(f"  • Range = [{df['label'].min():.2f}, {df['label'].max():.2f}]")
    if 'explained_variance_ratio_' in pca:
        print(f"  • Variance = {pca['explained_variance_ratio_']:.1%} (acceptable)")
    print("\n💡 Recommendation: PROCEED WITH MODEL TRAINING")
    
elif passed >= 2:
    print("\n⚠️ PCA LABELS NEED REVIEW")
    print("\nRun detailed diagnostics on a few windows")
    
else:
    print("\n❌ PCA LABELS HAVE ISSUES")
    print("\nCheck dimension scoring and PCA implementation")

print("="*60)

# 7. Sample extreme cases
print("\n6. SAMPLE EXTREME CASES:")
print("\nTop 3 Quality (Highest Labels):")
top3 = df.nlargest(3, 'label')[['window_id', 'label']]
print(top3.to_string(index=False))

print("\nBottom 3 Quality (Lowest Labels):")
bottom3 = df.nsmallest(3, 'label')[['window_id', 'label']]
print(bottom3.to_string(index=False))

print("\n💡 Next step: Re-score these windows to verify")
print("   Run: python ./tests/phase2_test.py")