import pandas as pd
from pathlib import Path
from scoring.traditional_scorer import score_window, score_windows_in_directory
from training.label_generator import generate_labeled_dataset

WINDOWS_DIR = Path("./data/processed/sample_windows")
OUTPUT_DIR  = Path("./data/processed/sample_windows/labelled")

def main() -> None:
    windows_dir = WINDOWS_DIR
    if not windows_dir.exists():
        raise FileNotFoundError(f"Expected directory {windows_dir} not found."
                                " Run this script from the project root.")

    """print("\nScoring individual windows:\n" + "-" * 40)
    for window_path in sorted(windows_dir.iterdir()):
        if window_path.is_dir():
            scores = score_window(window_path)
            print(f"Window {window_path.name} scores:")
            for dim_name, res in scores.items():
                s  = res.get('score', res.get('mpr', 0.0))
                ap = res.get('apr', 0.0)
                cv = res.get('coverage', res.get('coverage_ratio', 0.0))
                print(f"  {dim_name}: score={s:.4f}, apr={ap:.4f}, coverage={cv:.4f}")
            print()"""

    print("\nScoring all windows in directory:\n" + "-" * 40)
    all_scores = score_windows_in_directory(windows_dir)  # <-- fixed
    for wid, payload in all_scores.items():
        print(f"Window {wid} summary:")
        for dim_name, res in payload["result"].items():
            s  = res.get('score', res.get('mpr', 0.0))
            ap = res.get('apr', 0.0)
            cv = res.get('coverage', res.get('coverage_ratio', 0.0))
            print(f"  {dim_name}: score={s:.4f}, apr={ap:.4f}, coverage={cv:.4f}")
        print()

    print("\nGenerating labelled dataset...\n", end=' ', flush=True)
    dataset = generate_labeled_dataset(windows_dir, OUTPUT_DIR)  # use constants
    print("done.")
    print("\nFirst few rows of the labelled dataset:\n" + "-" * 40)
    print(dataset.head())
    print("\nLabel distribution: min={:.4f}, max={:.4f}".format(
        float(dataset['label'].min()), float(dataset['label'].max())))

if __name__ == '__main__':
    main()
