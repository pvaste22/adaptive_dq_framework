

from quality_dimensions.completeness import CompletenessDimension
from quality_dimensions.timeliness import TimelinessDimension
from pathlib import Path

# Load one window
windows_dir = Path('data/processed/training/v0/windows')
window_path = windows_dir / 'window_000000_20221231_160000'  # First window

# Initialize
comp = CompletenessDimension()
timely = TimelinessDimension()

# Load window
window_data = comp.load_window_from_disk(window_path)

# Score it
result = comp.calculate_score(window_data)

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