
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

# Import quality dimension classes. If additional dimensions are
# introduced (e.g. uniqueness or custom checks), add them to this list.
try:
    from quality_dimensions.timeliness import TimelinessDimension
    from quality_dimensions.completeness import CompletenessDimension
    from quality_dimensions.validity import ValidityDimension
    from quality_dimensions.consistency import ConsistencyDimension
    from quality_dimensions.accuracy import AccuracyDimension
    
except ImportError as e:
    # Provide a clear error message if imports fail. This may happen
    # during partial deployments or when the module names change. The
    # traditional scorer will not function without these classes.
    raise ImportError(
        "Failed to import one or more dimension calculators: {}. "
        "Ensure that timeliness.py, completeness.py, validity.py, "
        "consistency.py, and accuracy.py are importable and present on "
        "the PYTHONPATH.".format(e)
    )


logger = logging.getLogger(__name__)


def load_window_from_disk(window_path: Path) -> Dict[str, object]:
    """Load the contents of a single window from disk.

    Parameters
    ----------
    window_path: Path
        Path to the directory that contains ``cell_data.parquet``,
        ``ue_data.parquet``, and ``metadata.json``.

    Returns
    -------
    Dict[str, object]
        A dictionary with keys ``cell_data``, ``ue_data`` and
        ``metadata``. Missing files result in empty data frames or an
        empty metadata dictionary.
    """
    result: Dict[str, object] = {}
    # Parquet files
    cell_file = window_path / 'cell_data.parquet'
    if cell_file.exists():
        try:
            result['cell_data'] = pd.read_parquet(cell_file)
        except Exception as exc:
            logger.error(f"Failed to read cell_data from {cell_file}: {exc}")
            result['cell_data'] = pd.DataFrame()
    else:
        logger.debug(f"cell_data file not found at {cell_file}")
        result['cell_data'] = pd.DataFrame()

    ue_file = window_path / 'ue_data.parquet'
    if ue_file.exists():
        try:
            result['ue_data'] = pd.read_parquet(ue_file)
        except Exception as exc:
            logger.error(f"Failed to read ue_data from {ue_file}: {exc}")
            result['ue_data'] = pd.DataFrame()
    else:
        logger.debug(f"ue_data file not found at {ue_file}")
        result['ue_data'] = pd.DataFrame()

    # Metadata JSON
    meta_file = window_path / 'metadata.json'
    if meta_file.exists():
        try:
            with open(meta_file, 'r', encoding='utf-8') as fh:
                result['metadata'] = json.load(fh)
        except Exception as exc:
            logger.error(f"Failed to read metadata from {meta_file}: {exc}")
            result['metadata'] = {}
    else:
        logger.debug(f"metadata file not found at {meta_file}")
        result['metadata'] = {}

    return result


def _default_dimensions() -> Iterable:
    """Create a new list of dimension calculator instances.

    Returns
    -------
    Iterable
        A list containing one instance of each dimension calculator. A
        fresh instance is created on every call to ensure that
        potential internal state (e.g., score histories) does not
        accumulate between windows.
    """
    return [
        TimelinessDimension(),
        CompletenessDimension(),
        ValidityDimension(),
        ConsistencyDimension(),
        AccuracyDimension(),
    ]


def score_window(window_path: Path,
                 dimensions: Optional[Iterable] = None,
                 baselines: Optional[Dict] = None) -> Dict[str, Dict]:
    """Score a single data window using multiple quality dimensions.

    Parameters
    ----------
    window_path: Path
        Path to the window directory on disk.
    dimensions: Optional[Iterable]
        An iterable of dimension calculator instances. If omitted, a
        default set consisting of timeliness, completeness, validity,
        consistency and accuracy will be created. Pass a list of
        pre‐constructed dimensions to reuse baseline caches across
        windows.
    baselines: Optional[Dict]
        Optional dictionary containing baseline overrides. If
        provided, this dictionary will be passed to each dimension
        calculator's ``calculate_score`` method. Otherwise, the
        dimension's internal baselines will be used.

    Returns
    -------
    Dict[str, Dict]
        A dictionary keyed by dimension name (e.g. 'timeliness') with
        the result of each dimension's ``calculate_score`` call. If
        window loading fails or validation errors occur, the returned
        entries will contain error information in the 'details'
        sub-dictionary.
    """
    if dimensions is None:
        dimensions = _default_dimensions()
    # Load the window. Use a local helper instead of BaseDimension.load_window_from_disk
    window_data = load_window_from_disk(window_path)
    results: Dict[str, Dict] = {}
    for dim in dimensions:
        try:
            # Note: pass baselines explicitly if provided. Many
            # dimension classes will ignore the argument if None.
            res = dim.calculate_score(window_data) if baselines is None else dim.calculate_score(window_data, baselines)
        except Exception as exc:
            logger.exception(f"Error calculating score for dimension {dim.name} on window {window_path}: {exc}")
            res = {
                'score': 0.0,
                'apr': 0.0,
                'coverage': 0.0,
                'status': 'ERROR',
                'details': {'error': str(exc)},
            }
        results[dim.name] = res
    return results


def score_windows_in_directory(directory: Path,
                              dimensions: Optional[Iterable] = None,
                              baselines: Optional[Dict] = None) -> Dict[str, Dict[str, Dict]]:
    """Score all windows in a given directory.

    Parameters
    ----------
    directory: Path
        Path to a directory containing multiple window subdirectories.
        Each subdirectory will be scored individually. Only
        subdirectories that contain ``cell_data.parquet`` or
        ``ue_data.parquet`` will be processed; other files are
        ignored.
    dimensions: Optional[Iterable]
        As in :func:`score_window`, an iterable of dimension
        calculator instances to use. If omitted, a fresh list will be
        created for every window.
    baselines: Optional[Dict]
        Baseline overrides to pass to each dimension.

    Returns
    -------
    Dict[str, Dict[str, Dict]]
        A dictionary keyed by ``window_id`` (if available) or the
        directory name otherwise. Each entry contains the per‐dimension
        scoring result for that window. Errors during scoring are
        captured in the corresponding entry's dimension results.
    """
    results: Dict[str, Dict[str, Dict]] = {}
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a directory")

    for path in sorted(directory.iterdir()):
        if not path.is_dir():
            continue
        # Heuristically check that the subdirectory contains window data
        cell_file = path / 'cell_data.parquet'
        ue_file = path / 'ue_data.parquet'
        if not (cell_file.exists() or ue_file.exists()):
            continue
        # Determine an identifier for the window
        meta_file = path / 'metadata.json'
        if meta_file.exists():
            try:
                with open(meta_file, 'r', encoding='utf-8') as fh:
                    meta = json.load(fh)
                window_id = meta.get('window_id', path.name)
            except Exception:
                window_id = path.name
        else:
            window_id = path.name
        # Score the window. Always create fresh dimension instances if none provided.
        dims_to_use = dimensions if dimensions is not None else _default_dimensions()
        results[window_id] = score_window(path, dims_to_use, baselines)
    return results


__all__ = [
    'load_window_from_disk',
    'score_window',
    'score_windows_in_directory',
]