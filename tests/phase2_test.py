
import pandas as pd
from pathlib import Path
import sys
import types

# -----------------------------------------------------------------------------
# Monkeypatch Pandas Parquet reader/writer for testing
#
# In this environment, PyArrow/FastParquet may not be installed, so reading
# Parquet files would raise an error. For the synthetic test windows
# included in this repository (under `test_windows_dir`), the Parquet files
# actually contain CSV data. We monkeypatch Pandas.read_parquet to read
# these files as CSV instead. This allows the test to run without
# installing additional dependencies.

def _read_parquet_as_csv(path, *args, **kwargs):
    """Fallback reader that treats a .parquet file as CSV."""
    return pd.read_csv(path, *args, **kwargs)

pd.read_parquet = _read_parquet_as_csv  # type: ignore

# Monkeypatch DataFrame.to_parquet to write CSV when Parquet engines are missing.
# Without pyarrow/fastparquet, calling to_parquet will raise ImportError. We
# override it to call DataFrame.to_csv instead. The extension will still
# reflect '.parquet', but the contents will be CSV; this is acceptable for
# testing because our reader patch reads '.parquet' files as CSV.
def _df_to_parquet_as_csv(self: pd.DataFrame, path, *args, **kwargs):
    return self.to_csv(path, index=kwargs.get('index', True))

pd.DataFrame.to_parquet = _df_to_parquet_as_csv  # type: ignore


def main() -> None:
    """Execute a simple end-to-end test of the scoring pipeline."""
    # ---------------------------------------------------------------------
    # Inject a minimal constants module into sys.modules before importing
    # the dimension calculators. Many dimension modules import
    # `common.constants`, which in turn imports the root-level
    # ``constants.py``. The default ``constants.py`` attempts to
    # create directories under the user's home directory, which is
    # prohibited in this environment. To avoid PermissionError, we
    # provide a lightweight substitute with only the attributes
    # required for testing. Additional attributes can be added as
    # needed to satisfy imports in other modules.
    #
    # This injection must occur before any import that eventually
    # imports `common.constants` (e.g. timeliness, completeness).
    dummy_constants = types.SimpleNamespace(
        COLUMN_NAMES={
            'timestamp': 'timestamp',
            'cell_entity': 'Viavi.Cell.Name',
            'ue_entity': 'Viavi.UE.Name',
        },
        MEAS_INTERVAL_SEC=60,
        EXPECTED_ENTITIES={'cells': 1, 'ues': 1},
        EXPECTED_PATTERNS={
            'cqi_no_measurement_rate': 0.60,
            'mimo_zero_rate': 0.86,
        },
        DATA_QUIRKS={},
        PATHS=None,
        SCORING_LEVELS=None,
        NEAR_ZERO_THRESHOLDS={},
    )
    sys.modules['constants'] = dummy_constants

    # Create a bridge module for common.constants. Many modules import
    # `from common.constants import ...` which would normally execute the
    # code in ``common/constants.py``. To avoid executing that bridge file (and
    # its root-level imports), we register a dummy module under this name
    # that simply forwards attribute lookups to our injected constants.
    sys.modules['common.constants'] = dummy_constants

    # Stub out common.utils and common.logger before importing the scorer.
    # Without these stubs, the bridging modules under common/ would attempt
    # to import root-level `utils.py` and `logger.py`, potentially causing
    # ImportErrors or permission issues. The dimension calculators only
    # require a handful of functions from these modules; we provide minimal
    # implementations here. Expand the API as needed.
    import logging
    dummy_common_utils = types.SimpleNamespace(
        load_artifact=lambda *args, **kwargs: None,
        save_artifact=lambda *args, **kwargs: None,
        extract_band_from_cell_name=lambda *args, **kwargs: '',
        validate_window_completeness=lambda *args, **kwargs: None,
        create_timestamp_range=lambda start, end, freq='1T': [],
    )
    dummy_common_logger = types.SimpleNamespace(
        get_phase2_logger=lambda name: logging.getLogger(name)
    )
    sys.modules['common.utils'] = dummy_common_utils
    sys.modules['common.logger'] = dummy_common_logger

    # Ensure the project root (directory containing this script) is on sys.path.
    # Some modules (e.g. common.utils) attempt to import root-level files
    # using absolute imports. Without the root directory in sys.path, these
    # imports will fail. We prepend the parent directory of this script to
    # sys.path so that ``import utils`` and other top-level modules resolve.

    # Import modules after monkeypatching
    from scoring.traditional_scorer import score_window, score_windows_in_directory
    from training.label_generator import generate_labeled_dataset

    # Directory containing synthetic test windows
    windows_dir = Path('test_windows_dir')
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