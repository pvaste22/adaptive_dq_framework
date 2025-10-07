
import pandas as pd
from pathlib import Path
import sys
import types





def main() -> None:
    """Execute a simple end-to-end test of the scoring pipeline."""
    # ---------------------------------------------------------------------



    from scoring.traditional_scorer import score_window, score_windows_in_directory
    from training.label_generator import generate_labeled_dataset

    # Directory containing synthetic test windows
    windows_dir = Path('data/processed/training/v0/windows')
    if not windows_dir.exists():
        raise FileNotFoundError(f"Expected directory {windows_dir} not found."
                                " Run this script from the project root.")

    # Score each window individually
    print("\nScoring individual windows:\n" + "-" * 40)
    for window_path in sorted(windows_dir.iterdir()):
        if window_path.is_dir():
            scores = score_window(window_path)
            print(f"Window {window_path.name} scores:")
            for dim_name, res in scores.items():
                print(f"  {dim_name}: score={res.get('score'):.4f}, "
                      f"apr={res.get('apr'):.4f}, "
                      f"coverage={res.get('coverage'):.4f}")
            print()

    # Score all windows at once
    print("\nScoring all windows in directory:\n" + "-" * 40)
    all_scores = score_windows_in_directory(windows_dir)
    for wid, metrics in all_scores.items():
        print(f"Window {wid} summary:")
        for dim_name, res in metrics.items():
            print(f"  {dim_name}: score={res.get('score'):.4f}, "
                  f"apr={res.get('apr'):.4f}, coverage={res.get('coverage'):.4f}")
        print()

    # Generate labelled dataset (features + PCA label)
    output_dir = Path('test_output')
    print("\nGenerating labelled dataset...", end=' ', flush=True)
    dataset = generate_labeled_dataset(windows_dir, output_dir)
    print("done.")
    print("\nFirst few rows of the labelled dataset:\n" + "-" * 40)
    print(dataset.head())
    print("\nLabel distribution: min={:.4f}, max={:.4f}".format(
        dataset['label'].min(), dataset['label'].max()))


if __name__ == '__main__':
    main()