"""Assign future-label session files to the saved chronological data split."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from src.evaluation.splits import DEFAULT_OUTPUT_PATH, PARTS
from src.labeling.future_conflicts import (
    DEFAULT_OUTPUT_DIR as FUTURE_LABEL_DIR,
    FUTURE_LABEL_COLUMN,
    LABEL_STATUS_COLUMN,
    prepare_future_model_inputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABEL_AUDIT_PATH = FUTURE_LABEL_DIR / "future_label_audit.csv"
DEFAULT_MANIFEST_PATH = FUTURE_LABEL_DIR / "future_label_split_manifest.csv"
DEFAULT_SCALER_PATH = FUTURE_LABEL_DIR / "future_feature_scaler.joblib"
DEFAULT_CHUNK_SIZE = 500_000
KNOWN_LABEL_STATUSES = ("positive", "negative")
STATUS_COUNT_COLUMNS = (
    "positive",
    "negative",
    "ongoing_excluded",
    "unknown_current_separation",
    "unknown_incomplete_followup",
)


def build_future_label_split_manifest(
    label_audit_path: Path = DEFAULT_LABEL_AUDIT_PATH,
    assignment_path: Path = DEFAULT_OUTPUT_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    chunksize: int = DEFAULT_CHUNK_SIZE,
) -> pd.DataFrame:
    """Map each labeled session to one split, requiring every frame to be assigned."""
    if chunksize < 1:
        raise ValueError("chunksize must be positive.")

    assignment_frame = pd.read_csv(
        assignment_path,
        usecols=["snapshot_time", "part"],
    )
    if assignment_frame["snapshot_time"].duplicated().any():
        raise ValueError(f"{assignment_path} contains duplicate snapshot assignments.")
    if not assignment_frame["part"].isin(PARTS).all():
        raise ValueError(f"{assignment_path} contains an invalid split part.")
    assignment = assignment_frame.set_index("snapshot_time")["part"]

    label_audit = pd.read_csv(label_audit_path)
    records = []
    for row in label_audit.to_dict(orient="records"):
        label_path = Path(str(row["output_file"]))
        if not label_path.is_absolute():
            label_path = PROJECT_ROOT / label_path
        if not label_path.is_file():
            raise FileNotFoundError(f"Future-label session file does not exist: {label_path}")

        observed_parts: set[str] = set()
        missing_timestamps: set[float] = set()
        for frame in pd.read_csv(label_path, usecols=["snapshot_time"], chunksize=chunksize):
            timestamps = pd.to_numeric(frame["snapshot_time"], errors="coerce")
            assigned_parts = timestamps.map(assignment)
            if assigned_parts.isna().any():
                missing_timestamps.update(
                    timestamps[assigned_parts.isna()].dropna().unique().tolist()
                )
            observed_parts.update(assigned_parts.dropna().astype(str).unique())
        if missing_timestamps:
            raise ValueError(
                f"{label_path.name} has {len(missing_timestamps)} snapshot(s) missing "
                f"from {assignment_path.name}."
            )
        if len(observed_parts) != 1:
            raise ValueError(
                f"{label_path.name} spans multiple split parts: {sorted(observed_parts)}."
            )

        split = observed_parts.pop()
        try:
            manifest_label_path = str(label_path.relative_to(PROJECT_ROOT))
        except ValueError:
            manifest_label_path = str(label_path)
        record = {
            "split": split,
            "session_file": label_path.name,
            "label_file": manifest_label_path,
            "rows": int(row["rows"]),
        }
        for column in STATUS_COUNT_COLUMNS:
            record[column] = int(row[column])
        records.append(record)

    manifest = pd.DataFrame(records)
    if set(manifest["split"]) != set(PARTS):
        raise ValueError(
            f"Expected all split parts {PARTS}; found {sorted(manifest['split'].unique())}."
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def iter_labeled_split(
    split: str,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    chunksize: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """Yield known future-conflict examples for one split, excluding unknown labels."""
    if split not in PARTS:
        raise ValueError(f"split must be one of {PARTS}, not {split!r}.")
    if chunksize < 1:
        raise ValueError("chunksize must be positive.")

    manifest = pd.read_csv(manifest_path)
    selected = manifest.loc[manifest["split"].eq(split)]
    if selected.empty:
        raise ValueError(f"No future-label session files are assigned to {split}.")

    for relative_path in selected["label_file"]:
        path = Path(relative_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Future-label session file does not exist: {path}")
        for chunk in pd.read_csv(path, chunksize=chunksize):
            known = chunk[LABEL_STATUS_COLUMN].isin(KNOWN_LABEL_STATUSES)
            yield chunk.loc[known].copy()


def fit_future_feature_scaler(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    chunksize: int = DEFAULT_CHUNK_SIZE,
) -> StandardScaler:
    """Fit feature scaling on known training examples only."""
    scaler = StandardScaler()
    fitted = False
    for chunk in iter_labeled_split("train", manifest_path, chunksize):
        features, _, _ = prepare_future_model_inputs(chunk)
        if not features.empty:
            scaler.partial_fit(features)
            fitted = True
    if not fitted:
        raise ValueError("No usable training examples were found to fit feature scaling.")
    return scaler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-audit", type=Path, default=DEFAULT_LABEL_AUDIT_PATH)
    parser.add_argument("--assignment", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--scaler-output", type=Path, default=DEFAULT_SCALER_PATH)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()

    manifest = build_future_label_split_manifest(
        label_audit_path=args.label_audit,
        assignment_path=args.assignment,
        manifest_path=args.manifest,
        chunksize=args.chunksize,
    )
    summary = manifest.groupby("split", sort=False)[
        ["rows", *STATUS_COUNT_COLUMNS]
    ].sum()
    print(summary.to_string())
    print(f"Wrote split manifest: {args.manifest}")
    scaler = fit_future_feature_scaler(args.manifest, args.chunksize)
    args.scaler_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, args.scaler_output)
    print(f"Fitted feature scaler on {scaler.n_samples_seen_} training examples.")
    print(f"Wrote feature scaler: {args.scaler_output}")


if __name__ == "__main__":
    main()
