

from quality_dimensions.completeness import CompletenessDimension
from quality_dimensions.timeliness import TimelinessDimension
from quality_dimensions.consistency import ConsistencyDimension
from quality_dimensions.validity import ValidityDimension
from quality_dimensions.accuracy import AccuracyDimension
from pathlib import Path

# Load one window
windows_dir = Path('data/processed/training/v0/windows')
window_path = windows_dir / 'window_000121_20221231_180100'   # First window  window_000000_20221231_160000  window_009813_20230107_113300  window_010057_20230107_153700
                             #  window_002562_20230102_104200   window_005090_20230104_045000  window_008359_20230106_111900 window_000121_20221231_180100

# Initialize
comp = CompletenessDimension()
timely = TimelinessDimension()
cons = ConsistencyDimension()
val = ValidityDimension()
acc = AccuracyDimension()

# Load window
window_data = comp.load_window_from_disk(window_path)

# Score it
"""result = comp.calculate_score(window_data)

print(f"Score: {result['score']:.3f}")
print(f"APR: {result['apr']:.3f}")
print(f"MPR: {result['score']:.3f}")
print(f"Coverage: {result['coverage']:.3f}")
print(f"Fails: {result['details']['fail_counts']}")

timeli_res = timely.calculate_score(window_data)


print(f"Score: {timeli_res['score']:.3f}")
print(f"APR: {timeli_res['apr']:.3f}")
print(f"MPR: {timeli_res['score']:.3f}")
print(f"Coverage: {timeli_res['coverage']:.3f}")
print(f"Fails: {timeli_res['details']['fail_counts']}")


con_res = cons.calculate_score(window_data)


print(f"Score: {con_res['score']:.3f}")
print(f"APR: {con_res['apr']:.3f}")
print(f"MPR: {con_res['score']:.3f}")
print(f"Coverage: {con_res['coverage']:.3f}")
print(f"Fails: {con_res['details']['fail_counts']}")

val_res = val.calculate_score(window_data)


print(f"Score: {val_res['score']:.3f}")
print(f"APR: {val_res['apr']:.3f}")
print(f"MPR: {val_res['score']:.3f}")
print(f"Coverage: {val_res['coverage']:.3f}")
print(f"Fails: {val_res['details']['fail_counts']}")"""


acc_res = acc.calculate_score(window_data)


print(f"Score: {acc_res['score']:.3f}")
print(f"APR: {acc_res['apr']:.3f}")
print(f"MPR: {acc_res['score']:.3f}")
print(f"Coverage: {acc_res['coverage']:.3f}")
print(f"Fails: {acc_res['details']['fail_counts']}")