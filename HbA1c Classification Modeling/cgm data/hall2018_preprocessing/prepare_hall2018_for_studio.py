#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare Hall et al. 2018 (PLOS Biology glucotypes) CGM data for CGM Multimode Studio.

Inputs
------
1) S1 Data file, e.g. pbio.2005143.s010
   Tab-separated columns:
   DisplayTime, GlucoseValue, subjectId, InternalTime

2) Clinical SQLite database, e.g. pbio.2005143.s014.db
   Table: clinical
   Key columns:
   userID, diagnosis, A1C, glucotype, ...

Outputs
-------
output_root/
├── raw/
│   └── train/
│       ├── 1636-69-001.csv
│       └── ...
├── labels/
│   └── labels.csv
└── metadata/
    ├── processing_manifest.csv
    ├── clinical_metadata.csv
    └── preprocessing_config.json

Notes
-----
- GlucoseValue is treated as mg/dL and converted to mmol/L by division by 18.0156.
- "Low" sensor values are converted to NaN rather than assigned a numeric glucose value.
- DisplayTime is used as the main Timestamp.
- The official diagnosis column is used by default:
    non-diabetic -> Healthy
    pre-diabetic -> Prediabetes
    diabetic -> T2D
- A1C is retained as an optional continuous target, but is not required for classification.
- No smoothing, resampling, interpolation, peak detection, or feature extraction is performed here.
"""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

MGDL_PER_MMOLL = 18.0156

DIAGNOSIS_MAP = {
    "non-diabetic": "Healthy",
    "pre-diabetic": "Prediabetes",
    "diabetic": "T2D",
}


def normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_glucose(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Return numeric mg/dL values and a flag for non-numeric sensor codes."""
    raw = series.astype("string").str.strip()
    numeric = pd.to_numeric(raw, errors="coerce")
    sensor_code = raw.where(numeric.isna() & raw.notna() & raw.ne(""), pd.NA)
    return numeric.astype(float), sensor_code


def load_cgm(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    required = {"DisplayTime", "GlucoseValue", "subjectId"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("CGM file is missing required columns: {}".format(sorted(missing)))

    df["subjectId"] = df["subjectId"].map(normalize_id)
    df["Timestamp"] = pd.to_datetime(df["DisplayTime"], errors="coerce")
    df["InternalTime_parsed"] = pd.to_datetime(
        df.get("InternalTime", pd.Series(index=df.index, dtype="object")),
        errors="coerce",
    )
    glucose_mgdl, sensor_code = parse_glucose(df["GlucoseValue"])
    df["glucose_original_mgdl"] = glucose_mgdl
    df["sensor_code"] = sensor_code
    df["CGM"] = df["glucose_original_mgdl"] / MGDL_PER_MMOLL
    return df


def load_clinical(path: Path) -> pd.DataFrame:
    with sqlite3.connect(str(path)) as conn:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )["name"].tolist()
        if "clinical" not in tables:
            raise ValueError(
                "SQLite database does not contain a 'clinical' table. Found: {}".format(tables)
            )
        clinical = pd.read_sql_query("SELECT * FROM clinical", conn)

    if "userID" not in clinical.columns:
        raise ValueError("Clinical table is missing userID")
    if "diagnosis" not in clinical.columns:
        raise ValueError("Clinical table is missing diagnosis")

    clinical["userID"] = clinical["userID"].map(normalize_id)
    clinical["diagnosis_original"] = (
        clinical["diagnosis"].astype("string").str.strip().str.lower()
    )
    clinical["label"] = clinical["diagnosis_original"].map(DIAGNOSIS_MAP)
    return clinical


def prepare_dataset(
    cgm_path: Path,
    clinical_db_path: Path,
    output_root: Path,
    drop_unlabeled: bool = True,
    keep_all_columns: bool = True,
) -> None:
    raw_dir = output_root / "raw" / "train"
    labels_dir = output_root / "labels"
    metadata_dir = output_root / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    cgm = load_cgm(cgm_path)
    clinical = load_clinical(clinical_db_path)

    cgm_ids = set(cgm["subjectId"].dropna().unique())
    clinical_ids = set(clinical["userID"].dropna().unique())

    missing_clinical = sorted(cgm_ids.difference(clinical_ids))
    missing_cgm = sorted(clinical_ids.difference(cgm_ids))
    if missing_clinical:
        print("Warning: CGM subjects without clinical rows:", missing_clinical)
    if missing_cgm:
        print("Warning: clinical subjects without CGM rows:", missing_cgm)

    clinical_index = clinical.set_index("userID", drop=False)
    manifest_rows: List[Dict] = []
    label_rows: List[Dict] = []

    for subject_id, subject_df in cgm.groupby("subjectId", sort=True):
        if not subject_id:
            continue

        if subject_id not in clinical_index.index:
            manifest_rows.append({
                "subject_id": subject_id,
                "status": "skipped_no_clinical",
                "rows_input": len(subject_df),
                "rows_output": 0,
                "valid_cgm": 0,
                "label": "",
                "error": "No matching clinical row",
            })
            continue

        clinical_row = clinical_index.loc[subject_id]
        if isinstance(clinical_row, pd.DataFrame):
            clinical_row = clinical_row.iloc[0]

        label = clinical_row.get("label", pd.NA)
        if drop_unlabeled and pd.isna(label):
            manifest_rows.append({
                "subject_id": subject_id,
                "status": "skipped_no_label",
                "rows_input": len(subject_df),
                "rows_output": 0,
                "valid_cgm": int(subject_df["CGM"].notna().sum()),
                "label": "",
                "error": "Diagnosis could not be mapped",
            })
            continue

        out = subject_df.copy()
        out = out.dropna(subset=["Timestamp"])
        out = out.sort_values("Timestamp")

        # Aggregate exact duplicate timestamps deterministically.
        # Numeric glucose values are averaged; sensor codes and auxiliary values use first non-null.
        numeric_cols = ["glucose_original_mgdl", "CGM"]
        first_cols = [
            col for col in [
                "subjectId", "DisplayTime", "GlucoseValue", "InternalTime",
                "InternalTime_parsed", "sensor_code"
            ] if col in out.columns
        ]
        agg = {col: "mean" for col in numeric_cols if col in out.columns}
        agg.update({col: "first" for col in first_cols})
        out = out.groupby("Timestamp", as_index=False).agg(agg)

        out.insert(0, "subject_id", subject_id)
        out["label"] = label
        if "A1C" in clinical_row.index:
            out["hba1c"] = clinical_row.get("A1C")
        out["diagnosis_original"] = clinical_row.get("diagnosis_original")
        if "glucotype" in clinical_row.index:
            out["glucotype"] = clinical_row.get("glucotype")
        out["CGM_unit"] = "mmol/L"
        out["source_dataset"] = "Hall2018_Glucotypes"

        # Studio-critical columns first.
        preferred = [
            "subject_id", "Timestamp", "CGM", "CGM_unit", "label", "hba1c",
            "diagnosis_original", "glucotype", "glucose_original_mgdl",
            "sensor_code", "source_dataset", "DisplayTime", "InternalTime",
            "InternalTime_parsed", "GlucoseValue", "subjectId"
        ]
        ordered = [c for c in preferred if c in out.columns]
        ordered += [c for c in out.columns if c not in ordered]
        out = out[ordered]

        output_file = raw_dir / "{}.csv".format(subject_id)
        out.to_csv(output_file, index=False)

        label_rows.append({
            "subject_id": subject_id,
            "label": label,
            "hba1c": clinical_row.get("A1C", pd.NA),
            "diagnosis_original": clinical_row.get("diagnosis_original", pd.NA),
            "glucotype": clinical_row.get("glucotype", pd.NA),
            "dataset": "Hall2018_Glucotypes",
            "label_source": "official_composite_diagnosis",
        })

        ts = out["Timestamp"]
        manifest_rows.append({
            "subject_id": subject_id,
            "status": "done",
            "rows_input": len(subject_df),
            "rows_output": len(out),
            "valid_cgm": int(out["CGM"].notna().sum()),
            "missing_cgm": int(out["CGM"].isna().sum()),
            "start_time": ts.min(),
            "end_time": ts.max(),
            "duration_days": (
                (ts.max() - ts.min()).total_seconds() / 86400.0
                if len(ts) and pd.notna(ts.min()) and pd.notna(ts.max())
                else np.nan
            ),
            "label": label,
            "error": "",
        })

    labels = pd.DataFrame(label_rows).sort_values("subject_id")
    labels.to_csv(labels_dir / "labels.csv", index=False)

    clinical.to_csv(metadata_dir / "clinical_metadata.csv", index=False)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(metadata_dir / "processing_manifest.csv", index=False)

    config = {
        "source_cgm_file": str(cgm_path.resolve()),
        "source_clinical_db": str(clinical_db_path.resolve()),
        "timestamp_source": "DisplayTime",
        "glucose_source": "GlucoseValue",
        "input_glucose_unit": "mg/dL",
        "output_glucose_unit": "mmol/L",
        "conversion": "CGM = GlucoseValue / 18.0156",
        "non_numeric_sensor_values": "converted to NaN and retained in sensor_code",
        "label_source": "clinical.diagnosis",
        "label_mapping": DIAGNOSIS_MAP,
        "smoothing": False,
        "resampling": False,
        "interpolation": False,
        "peak_detection": False,
        "feature_engineering": False,
    }
    with open(metadata_dir / "preprocessing_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("Finished.")
    print("Subjects exported:", len(labels))
    if len(labels):
        print("Class distribution:")
        print(labels["label"].value_counts(dropna=False).to_string())
    print("Output:", output_root.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Hall 2018 glucotypes CGM data for CGM Studio"
    )
    parser.add_argument("--cgm-file", required=True, type=Path)
    parser.add_argument("--clinical-db", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--keep-unlabeled",
        action="store_true",
        help="Export subjects even if diagnosis cannot be mapped",
    )
    args = parser.parse_args()

    prepare_dataset(
        cgm_path=args.cgm_file,
        clinical_db_path=args.clinical_db,
        output_root=args.output_root,
        drop_unlabeled=not args.keep_unlabeled,
    )


if __name__ == "__main__":
    main()
