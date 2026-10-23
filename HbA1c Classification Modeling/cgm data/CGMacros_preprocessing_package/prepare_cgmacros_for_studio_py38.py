#!/usr/bin/env python3
"""Prepare CGMacros for the CGM Studio interface.

Expected source layout (nested folders are supported):
    DATA_ROOT/
      CGMacros-001/CGMacros-001.csv
      CGMacros-002/CGMacros-002.csv
      ...

The script:
1. Recursively finds subject CSV files.
2. Converts Libre GL and Dexcom GL from mg/dL to mmol/L.
3. Creates a standardized CGM column for the interface.
4. Matches HbA1c from bio.csv.
5. Creates subject-level labels.csv and a processing manifest.

Example:
    python prepare_cgmacros_for_studio.py \
        --input-root /path/to/CGMacros \
        --bio-csv /path/to/bio.csv \
        --output-root ./CGMacros_studio_ready \
        --cgm-source dexcom
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd

MGDL_PER_MMOLL = 18.0156
SUBJECT_PATTERN = re.compile(r"CGMacros[-_ ]?(\d+)", re.IGNORECASE)


def normalize_subject_id(value: object, width: int = 3) -> str:
    """Normalize IDs such as 1, 001, CGMacros-001 to 001."""
    text = str(value).strip()
    match = re.search(r"(\d+)", text)
    if not match:
        raise ValueError(f"Cannot parse subject ID from: {value!r}")
    return str(int(match.group(1))).zfill(width)


def extract_subject_id(path: Path, width: int = 3) -> Optional[str]:
    """Extract subject ID from file name or parent folder names."""
    candidates = [path.stem, *[p.name for p in path.parents]]
    for candidate in candidates:
        match = SUBJECT_PATTERN.search(candidate)
        if match:
            return str(int(match.group(1))).zfill(width)
    return None


def classify_hba1c(value: float) -> str:
    """Classify HbA1c according to the project definition."""
    if pd.isna(value):
        return ""
    if value < 5.7:
        return "Healthy"
    if value <= 6.4:
        return "Prediabetes"
    return "T2D"


def load_bio_labels(bio_csv: Path, id_width: int = 3) -> pd.DataFrame:
    bio = pd.read_csv(bio_csv)
    required = {"subject", "A1c PDL (Lab)"}
    missing = required.difference(bio.columns)
    if missing:
        raise ValueError(f"bio.csv is missing required columns: {sorted(missing)}")

    labels = bio[["subject", "A1c PDL (Lab)"]].copy()
    labels["subject_id"] = labels["subject"].map(
        lambda x: normalize_subject_id(x, width=id_width)
    )
    labels["hba1c"] = pd.to_numeric(labels["A1c PDL (Lab)"], errors="coerce")
    labels["label"] = labels["hba1c"].map(classify_hba1c)
    labels = labels[["subject_id", "hba1c", "label"]]

    if labels["subject_id"].duplicated().any():
        dupes = labels.loc[labels["subject_id"].duplicated(), "subject_id"].tolist()
        raise ValueError(f"Duplicate subject IDs in bio.csv: {dupes}")

    return labels.sort_values("subject_id").reset_index(drop=True)


def choose_standard_cgm(
    df: pd.DataFrame,
    source: str,
) -> Tuple[pd.Series, pd.Series]:
    dexcom = df.get("Dexcom GL", pd.Series(np.nan, index=df.index, dtype=float))
    libre = df.get("Libre GL", pd.Series(np.nan, index=df.index, dtype=float))

    if source == "dexcom":
        cgm = dexcom.copy()
        src = pd.Series(np.where(cgm.notna(), "Dexcom", ""), index=df.index)
    elif source == "libre":
        cgm = libre.copy()
        src = pd.Series(np.where(cgm.notna(), "Libre", ""), index=df.index)
    elif source == "dexcom_then_libre":
        cgm = dexcom.combine_first(libre)
        src = pd.Series(
            np.where(dexcom.notna(), "Dexcom", np.where(libre.notna(), "Libre", "")),
            index=df.index,
        )
    else:
        raise ValueError(f"Unsupported CGM source: {source}")

    return cgm, src


def process_subject_csv(
    source_csv: Path,
    output_csv: Path,
    subject_id: str,
    hba1c: float,
    label: str,
    cgm_source: str,
) -> dict:
    df = pd.read_csv(source_csv)

    required = {"Timestamp"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}")
    if "Libre GL" not in df.columns and "Dexcom GL" not in df.columns:
        raise ValueError("Neither 'Libre GL' nor 'Dexcom GL' is present")

    # Remove common exported index columns.
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    invalid_time_rows = int(df["Timestamp"].isna().sum())
    df = df.dropna(subset=["Timestamp"]).copy()

    # Convert both sensor columns to mmol/L before any feature engineering.
    for col in ["Libre GL", "Dexcom GL"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / MGDL_PER_MMOLL

    df = df.sort_values("Timestamp", kind="stable")
    duplicated_timestamps = int(df["Timestamp"].duplicated().sum())
    df = df.drop_duplicates(subset=["Timestamp"], keep="first").reset_index(drop=True)

    cgm, cgm_source_series = choose_standard_cgm(df, cgm_source)
    df["CGM"] = cgm
    df["CGM source"] = cgm_source_series
    df["subject_id"] = subject_id
    df["hba1c"] = hba1c
    df["label"] = label
    df["Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Keep columns in a predictable order while preserving all original metadata.
    preferred = [
        "subject_id", "Timestamp", "CGM", "CGM source", "Libre GL", "Dexcom GL",
        "hba1c", "label", "Meal Type", "Calories", "Carbs", "Protein", "Fat",
        "Fiber", "Amount Consumed ", "HR", "Calories (Activity)", "METs", "Image path",
    ]
    ordered = [c for c in preferred if c in df.columns]
    ordered += [c for c in df.columns if c not in ordered]
    df = df[ordered]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    return {
        "subject_id": subject_id,
        "source_file": str(source_csv),
        "output_file": str(output_csv),
        "hba1c": None if pd.isna(hba1c) else float(hba1c),
        "label": label,
        "rows_output": int(len(df)),
        "invalid_timestamp_rows_removed": invalid_time_rows,
        "duplicate_timestamps_removed": duplicated_timestamps,
        "libre_valid": int(df["Libre GL"].notna().sum()) if "Libre GL" in df else 0,
        "dexcom_valid": int(df["Dexcom GL"].notna().sum()) if "Dexcom GL" in df else 0,
        "cgm_valid": int(df["CGM"].notna().sum()),
        "meal_rows": int(df["Meal Type"].notna().sum()) if "Meal Type" in df else 0,
        "status": "success",
        "error": "",
    }


def find_subject_csvs(input_root: Path) -> List[Path]:
    files = []
    for path in input_root.rglob("*.csv"):
        if path.name.lower() == "bio.csv":
            continue
        if extract_subject_id(path) is not None:
            files.append(path)
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--bio-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--cgm-source",
        choices=["dexcom", "libre", "dexcom_then_libre"],
        default="dexcom",
        help="Source used to create standardized CGM column (default: dexcom).",
    )
    parser.add_argument("--id-width", type=int, default=3)
    parser.add_argument(
        "--copy-bio",
        action="store_true",
        help="Copy bio.csv into output metadata directory.",
    )
    args = parser.parse_args()

    input_root = args.input_root.expanduser().resolve()
    bio_csv = args.bio_csv.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    raw_output = output_root / "raw" / "train"
    labels_output = output_root / "labels"
    metadata_output = output_root / "metadata"

    labels = load_bio_labels(bio_csv, id_width=args.id_width)
    label_map = labels.set_index("subject_id").to_dict("index")
    subject_csvs = find_subject_csvs(input_root)
    if not subject_csvs:
        raise FileNotFoundError(f"No CGMacros subject CSV files found under {input_root}")

    raw_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)
    metadata_output.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    seen_subjects: Set[str] = set()

    for source_csv in subject_csvs:
        subject_id = extract_subject_id(source_csv, width=args.id_width)
        if subject_id is None:
            continue

        # Ignore duplicate auxiliary CSVs for the same subject. Prefer exact subject filename.
        if subject_id in seen_subjects:
            manifest_rows.append({
                "subject_id": subject_id,
                "source_file": str(source_csv),
                "status": "skipped_duplicate_subject",
                "error": "Another CSV for this subject was already processed",
            })
            continue

        label_info = label_map.get(subject_id)
        if label_info is None:
            manifest_rows.append({
                "subject_id": subject_id,
                "source_file": str(source_csv),
                "status": "failed",
                "error": "Subject not found in bio.csv",
            })
            continue

        output_csv = raw_output / f"CGMacros-{subject_id}.csv"
        try:
            row = process_subject_csv(
                source_csv=source_csv,
                output_csv=output_csv,
                subject_id=subject_id,
                hba1c=label_info["hba1c"],
                label=label_info["label"],
                cgm_source=args.cgm_source,
            )
            manifest_rows.append(row)
            seen_subjects.add(subject_id)
        except Exception as exc:  # continue processing remaining subjects
            manifest_rows.append({
                "subject_id": subject_id,
                "source_file": str(source_csv),
                "status": "failed",
                "error": str(exc),
            })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(metadata_output / "processing_manifest.csv", index=False)

    successful_ids = sorted(
        manifest.loc[manifest.get("status", pd.Series(dtype=str)) == "success", "subject_id"]
        .astype(str)
        .unique()
        .tolist()
    )
    output_labels = labels[labels["subject_id"].isin(successful_ids)].copy()
    output_labels.to_csv(labels_output / "labels.csv", index=False)

    class_counts = output_labels["label"].value_counts(dropna=False).to_dict()
    config = {
        "input_root": str(input_root),
        "bio_csv": str(bio_csv),
        "output_root": str(output_root),
        "input_glucose_unit": "mg/dL",
        "output_glucose_unit": "mmol/L",
        "conversion_formula": "mmol/L = mg/dL / 18.0156",
        "standard_cgm_source": args.cgm_source,
        "hba1c_label_rule": {
            "Healthy": "HbA1c < 5.7",
            "Prediabetes": "5.7 <= HbA1c <= 6.4",
            "T2D": "HbA1c > 6.4",
        },
        "subjects_found": len(subject_csvs),
        "subjects_processed": len(successful_ids),
        "class_counts": class_counts,
    }
    with open(metadata_output / "preprocessing_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    if args.copy_bio:
        shutil.copy2(bio_csv, metadata_output / "bio.csv")

    print(f"Found CSV files: {len(subject_csvs)}")
    print(f"Successfully processed subjects: {len(successful_ids)}")
    print(f"Interface-ready training data: {raw_output}")
    print(f"Labels: {labels_output / 'labels.csv'}")
    print(f"Manifest: {metadata_output / 'processing_manifest.csv'}")
    print(f"Class counts: {class_counts}")


if __name__ == "__main__":
    main()
