

from quality_dimensions.completeness import CompletenessDimension
from quality_dimensions.timeliness import TimelinessDimension
from quality_dimensions.consistency import ConsistencyDimension
from quality_dimensions.validity import ValidityDimension
from quality_dimensions.accuracy import AccuracyDimension
from scoring.traditional_scorer import score_window, score_windows_in_directory
from pathlib import Path
import csv

# Load one window
windows_dir = Path('data/processed/training/v0/windows')
window_path = windows_dir / 'window_000000_20221231_160000'   
# First window  window_009908_20230107_130800  window_010037_20230107_151700  window_000000_20221231_160000  window_009813_20230107_113300 
#  window_010057_20230107_153700 window_000345_20221231_214500
#  window_002562_20230102_104200   window_005090_20230104_045000  window_008359_20230106_111900 window_000121_20221231_180100 
#  window_005483_20230104_112300  window_005290_20230104_081000  window_005178_20230104_061800  window_005084_20230104_044400 (perfect score)
# window_005042_20230104_040200 window_000471_20221231_235100
#window_000097_20221231_173700  window_003659_20230103_045900  window_009935_20230107_133500  window_001156_20230101_111600  
# window_000471_20221231_235100 window_005084_20230104_044400 window_005042_20230104_040200 window_000471_20221231_235100, window_010014_20230107_145400
# Initialize
comp = CompletenessDimension()
timely = TimelinessDimension()
cons = ConsistencyDimension()
val = ValidityDimension()
acc = AccuracyDimension()

# Load window
window_data = timely.load_window_from_disk(window_path)


"""timeli_res = timely.calculate_score(window_data)

print("\nScoring timeliness :\n" + "-" * 40)
print(f"Score: {timeli_res['score']:.3f}")
print(f"APR: {timeli_res['apr']:.3f}")
print(f"MPR: {timeli_res['score']:.3f}")
print(f"Coverage: {timeli_res['coverage']:.3f}")
print(f"Fails: {timeli_res['details']['fail_counts']}")"""

"""
scores= score_window(window_path)
for dim_name, res in scores.items():
    print(f"  {dim_name}: score={res.get('score'):.4f}, "
            f"apr={res.get('apr'):.4f}, "
            f"coverage={res.get('coverage'):.4f}")
  
    
print("\nScoring all windows in directory:\n" + "-" * 40)
all_scores = score_windows_in_directory(windows_dir)
for wid, metrics in all_scores.items():
    print(f"Window {wid} summary:")
    for dim_name, res in metrics.items():
        print(f"  {dim_name}: score={res.get('score'):.4f}, "
            f"apr={res.get('apr'):.4f}, coverage={res.get('coverage'):.4f}")
"""
# Score it


result = comp.calculate_score(window_data)
print("\nScoring completeness :\n" + "-" * 40)
print(f"Score: {result['score']:.3f}")
print(f"APR: {result['apr']:.3f}")
print(f"MPR: {result['score']:.3f}")
print(f"Coverage: {result['coverage']:.3f}")
print(f"Fails: {result['details']['fail_counts']}")

timeli_res = timely.calculate_score(window_data)

print("\nScoring timeliness :\n" + "-" * 40)
print(f"Score: {timeli_res['score']:.3f}")
print(f"APR: {timeli_res['apr']:.3f}")
print(f"MPR: {timeli_res['score']:.3f}")
print(f"Coverage: {timeli_res['coverage']:.3f}")
print(f"Fails: {timeli_res['details']['fail_counts']}")


con_res = cons.calculate_score(window_data)

print("\nScoring cnsistency :\n" + "-" * 40)
print(f"Score: {con_res['score']:.3f}")
print(f"APR: {con_res['apr']:.3f}")
print(f"MPR: {con_res['score']:.3f}")
print(f"Coverage: {con_res['coverage']:.3f}")
print(f"Fails: {con_res['details']['fail_counts']}")

val_res = val.calculate_score(window_data)

print("\nScoring validity :\n" + "-" * 40)
print(f"Score: {val_res['score']:.3f}")
print(f"APR: {val_res['apr']:.3f}")
print(f"MPR: {val_res['score']:.3f}")
print(f"Coverage: {val_res['coverage']:.3f}")
print(f"Fails: {val_res['details']['fail_counts']}")


acc_res = acc.calculate_score(window_data)

print("\nScoring accuracy :\n" + "-" * 40)
print(f"Score: {acc_res['score']:.3f}")
print(f"APR: {acc_res['apr']:.3f}")
print(f"MPR: {acc_res['score']:.3f}")
print(f"Coverage: {acc_res['coverage']:.3f}")
print(f"Fails: {acc_res['details']['fail_counts']}")

"""
SAVE_CSV = True  # CSV likhna ho to True rakhein

# ---- Init ----
timely = TimelinessDimension()

all_results = []  # aggregate ke liye

# ---- Iterate over all window_* folders ----
# sirf directories pick karne ke liye filter; sorted for stable order
window_dirs = sorted([p for p in windows_dir.iterdir() if p.is_dir() and p.name.startswith("window_")])

if not window_dirs:
    print(f"No window_* folders found under: {windows_dir.resolve()}")

for idx, window_path in enumerate(window_dirs, start=1):
    try:
        # Load
        window_data = timely.load_window_from_disk(window_path)
        # Score
        timeli_res = timely.calculate_score(window_data)

        # fields (APR/MPR may not exist in some impls—graceful fallback)
        score = timeli_res.get("score")
        apr = timeli_res.get("apr")
        mpr = timeli_res.get("mpr")  # aapke print me galti se score dubara print ho raha tha

        coverage = timeli_res.get("coverage")
        fails = timeli_res.get("details", {}).get("fail_counts")

        # Print nicely
        print(f"\n[{idx}/{len(window_dirs)}] Scoring timeliness for: {window_path.name}")
        print("-" * 50)
        if score is not None:
            print(f"Score:    {score:.3f}")
        if apr is not None:
            print(f"APR:      {apr:.3f}")
        if mpr is not None:
            print(f"MPR:      {mpr:.3f}")
        if coverage is not None:
            print(f"Coverage: {coverage:.3f}")
        print(f"Fails:    {fails}")

        # Collect for CSV
        all_results.append({
            "window": window_path.name,
            "score": score,
            "apr": apr,
            "mpr": mpr,
            "coverage": coverage,
            "fails": fails,
        })

    except Exception as e:
        # koi window fail ho jaye to baaki continue kare
        print(f"\n[{idx}/{len(window_dirs)}] ERROR in {window_path.name}: {e}")
        all_results.append({
            "window": window_path.name,
            "score": None,
            "apr": None,
            "mpr": None,
            "coverage": None,
            "fails": f"ERROR: {e}",
        })

# ---- Optional: write a CSV summary ----
if SAVE_CSV and all_results:
    out_csv = windows_dir / "timeliness_scores_summary.csv"
    fieldnames = ["window", "score", "apr", "mpr", "coverage", "fails"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    print(f"\nSaved summary: {out_csv.resolve()}")

print(f"\nDone. Processed {len(window_dirs)} window folders.")
"""

