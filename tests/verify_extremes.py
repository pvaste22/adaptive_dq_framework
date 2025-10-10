from pathlib import Path
from scoring.traditional_scorer import score_window
from quality_dimensions.timeliness import TimelinessDimension
from quality_dimensions.completeness import CompletenessDimension
from quality_dimensions.validity import ValidityDimension
from quality_dimensions.consistency import ConsistencyDimension
from quality_dimensions.accuracy import AccuracyDimension

# Best windows (label=1.0)
best = ['window_000013_20221231_161000', 
        'window_004496_20230103_185300',
        'window_004497_20230103_185400']

# Worst windows (label=0.0)
worst = ['window_001478_20230101_163500',
         'window_003642_20230103_043900', 
         'window_004280_20230103_151700']

base_dir = Path('./data/processed/training/v0/windows')

print("="*60)
print("EXTREME CASES VERIFICATION")
print("="*60)

dims = [TimelinessDimension(), CompletenessDimension(), 
        ValidityDimension(), ConsistencyDimension(), AccuracyDimension()]

print("\n✨ BEST WINDOWS (Label = 1.0):")
best_scores = []
for wid in best:
    path = base_dir / wid
    if path.exists():
        result = score_window(path, dims)
        print(f"\n{wid}:")
        window_scores = []
        for dim_name, scores in result.items():
            # 'score' is the MPR
            mpr = scores.get('score', scores.get('mpr', 0.0))
            print(f"  {dim_name:15s}: {mpr:.3f}")
            window_scores.append(mpr)
        avg = sum(window_scores) / len(window_scores) if window_scores else 0
        print(f"  {'AVERAGE':15s}: {avg:.3f}")
        best_scores.append(avg)
    else:
        print(f"\n{wid}: NOT FOUND")

print("\n" + "-"*60)
print("\n💀 WORST WINDOWS (Label = 0.0):")
worst_scores = []
for wid in worst:
    path = base_dir / wid
    if path.exists():
        result = score_window(path, dims)
        print(f"\n{wid}:")
        window_scores = []
        for dim_name, scores in result.items():
            mpr = scores.get('score', scores.get('mpr', 0.0))
            print(f"  {dim_name:15s}: {mpr:.3f}")
            window_scores.append(mpr)
        avg = sum(window_scores) / len(window_scores) if window_scores else 0
        print(f"  {'AVERAGE':15s}: {avg:.3f}")
        worst_scores.append(avg)
    else:
        print(f"\n{wid}: NOT FOUND")

print("\n" + "="*60)
print("VERDICT:")
print("="*60)

if best_scores and worst_scores:
    best_avg = sum(best_scores) / len(best_scores)
    worst_avg = sum(worst_scores) / len(worst_scores)
    
    print(f"\nBEST windows average:  {best_avg:.3f}")
    print(f"WORST windows average: {worst_avg:.3f}")
    print(f"Gap: {best_avg - worst_avg:.3f}")
    
    if best_avg > 0.90 and worst_avg < 0.80:
        print("\n✅ PCA IS CORRECT!")
        print("   High labels = high quality ✓")
        print("   Low labels = low quality ✓")
        print("   Gap is significant ✓")
    elif best_avg > 0.85 and worst_avg < 0.85:
        print("\n✅ PCA IS MOSTLY CORRECT")
        print("   Labels properly distinguish quality levels")
    else:
        print("\n⚠️ UNEXPECTED PATTERN")
        print(f"   Expected: BEST > 0.90, WORST < 0.80")
        print(f"   Actual: BEST = {best_avg:.3f}, WORST = {worst_avg:.3f}")
else:
    print("\n⚠️ Could not verify - windows not found")

print("="*60)