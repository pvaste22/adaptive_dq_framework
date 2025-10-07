
import argparse
import random
from pathlib import Path
from typing import Tuple

import pandas as pd


def _inject_cell_faults(df: pd.DataFrame, fault_fraction: float) -> pd.DataFrame:
    """Return a DataFrame containing a subset of rows with injected faults.

    Parameters
    ----------
    df : pd.DataFrame
        Original cell reports data.
    fault_fraction : float
        Fraction of rows to duplicate and corrupt (between 0 and 1).

    Returns
    -------
    pd.DataFrame
        New DataFrame containing only the faulty rows.
    """
    n_rows = len(df)
    n_faults = max(1, int(n_rows * fault_fraction))
    # Sample without replacement to avoid duplicate indices
    faulty_rows = df.sample(n=n_faults, random_state=42).copy(deep=True)

    # Inject various faults
    for idx, row in faulty_rows.iterrows():
        # Randomly pick a fault type
        fault_type = random.choice(["missing", "out_of_range", "inconsistent"])
        if fault_type == "missing":
            # Drop a mandatory field such as PRB availability
            if "RRU.PrbAvailDl" in row:
                row["RRU.PrbAvailDl"] = pd.NA
            if "RRU.PrbAvailUl" in row:
                row["RRU.PrbAvailUl"] = pd.NA
        elif fault_type == "out_of_range":
            # Inflate PRB total percentage beyond 100 and throughput beyond typical limits
            if "RRU.PrbTotDl" in row:
                row["RRU.PrbTotDl"] = 120.0  # percentage >100% invalid
            if "RRU.PrbTotUl" in row:
                row["RRU.PrbTotUl"] = 120.0
            # Throughput unrealistic high
            for col in ["DRB.UEThpDl", "DRB.UEThpUl"]:
                if col in row and pd.notna(row[col]):
                    row[col] = row[col] * 10.0
        elif fault_type == "inconsistent":
            # Set PRB used greater than available
            if "RRU.PrbUsedDl" in row and "RRU.PrbAvailDl" in row:
                row["RRU.PrbUsedDl"] = row["RRU.PrbAvailDl"] + 10
            if "RRU.PrbUsedUl" in row and "RRU.PrbAvailUl" in row:
                row["RRU.PrbUsedUl"] = row["RRU.PrbAvailUl"] + 10
            # Zero throughput but non‑zero connections
            if "RRC.ConnMean" in row and "DRB.UEThpDl" in row:
                row["RRC.ConnMean"] = 5
                row["DRB.UEThpDl"] = 0.0
        # Assign the modified row back
        faulty_rows.loc[idx] = row
    return faulty_rows


def _inject_ue_faults(df: pd.DataFrame, fault_fraction: float) -> pd.DataFrame:
    """Return a DataFrame containing a subset of UE rows with injected faults.

    Parameters
    ----------
    df : pd.DataFrame
        Original UE reports data.
    fault_fraction : float
        Fraction of rows to duplicate and corrupt (between 0 and 1).

    Returns
    -------
    pd.DataFrame
        New DataFrame containing only the faulty rows.
    """
    n_rows = len(df)
    n_faults = max(1, int(n_rows * fault_fraction))
    faulty_rows = df.sample(n=n_faults, random_state=42).copy(deep=True)
    for idx, row in faulty_rows.iterrows():
        fault_type = random.choice(["missing", "out_of_range", "inconsistent"])
        if fault_type == "missing":
            # Missing CQI values
            if "DRB.UECqiDl" in row:
                row["DRB.UECqiDl"] = pd.NA
            if "DRB.UECqiUl" in row:
                row["DRB.UECqiUl"] = pd.NA
        elif fault_type == "out_of_range":
            # CQI out of valid range
            if "DRB.UECqiDl" in row:
                row["DRB.UECqiDl"] = 20  # >15 invalid
            if "DRB.UECqiUl" in row:
                row["DRB.UECqiUl"] = 20
            # Inflate PRB used beyond typical counts
            for col in ["RRU.PrbUsedDl", "RRU.PrbUsedUl"]:
                if col in row and pd.notna(row[col]):
                    row[col] = row[col] * 5.0
        elif fault_type == "inconsistent":
            # Set throughput but zero PRB used (inefficient scenario)
            for col in ["RRU.PrbUsedDl", "RRU.PrbUsedUl"]:
                if col in row:
                    row[col] = 0
            for col in ["DRB.UEThpDl", "DRB.UEThpUl"]:
                if col in row:
                    row[col] = 100.0  # some positive throughput
        faulty_rows.loc[idx] = row
    return faulty_rows


def inject_faults(
    cell_input: Path,
    ue_input: Path,
    cell_output: Path,
    ue_output: Path,
    fault_fraction: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Inject faults into cell and UE datasets and write output files.

    Parameters
    ----------
    cell_input : Path
        Path to the input cell report CSV.
    ue_input : Path
        Path to the input UE report CSV.
    cell_output : Path
        Path to write the faulty cell report CSV.
    ue_output : Path
        Path to write the faulty UE report CSV.
    fault_fraction : float, optional
        Fraction of rows to duplicate and corrupt (default 0.1).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The augmented cell and UE DataFrames.
    """
    # Read the datasets
    cell_df = pd.read_csv(cell_input)
    ue_df = pd.read_csv(ue_input)

    faulty_cell = _inject_cell_faults(cell_df, fault_fraction)
    faulty_ue = _inject_ue_faults(ue_df, fault_fraction)

    # Append faulty rows to original datasets
    augmented_cell = pd.concat([cell_df, faulty_cell], ignore_index=True)
    augmented_ue = pd.concat([ue_df, faulty_ue], ignore_index=True)

    # Write out the new datasets
    augmented_cell.to_csv(cell_output, index=False)
    augmented_ue.to_csv(ue_output, index=False)

    return augmented_cell, augmented_ue


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject synthetic faults into O-RAN datasets.")
    parser.add_argument("--cell-input", type=Path, required=True, help="Path to input cell report CSV")
    parser.add_argument("--ue-input", type=Path, required=True, help="Path to input UE report CSV")
    parser.add_argument("--cell-output", type=Path, required=True, help="Path to output cell CSV with faults")
    parser.add_argument("--ue-output", type=Path, required=True, help="Path to output UE CSV with faults")
    parser.add_argument(
        "--fault-fraction",
        type=float,
        default=0.1,
        help="Fraction of rows to duplicate and corrupt (0 < f ≤ 1; default=0.1)",
    )
    args = parser.parse_args()
    inject_faults(args.cell_input, args.ue_input, args.cell_output, args.ue_output, args.fault_fraction)


if __name__ == "__main__":
    main()