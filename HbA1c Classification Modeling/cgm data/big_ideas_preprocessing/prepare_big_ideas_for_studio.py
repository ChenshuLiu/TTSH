"""Normalize the supplied BIG IDEAS Dexcom archive to Hall-style Studio CSVs.

Reads archive members in memory; never executes or extracts attachment code.
No interpolation, smoothing, resampling, peak detection, or model training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

DATASET = "BIG_IDEAS"
DIVISOR = 18.0156  # Same divisor as the supplied Hall2018 reference package.
TIME = "Timestamp (YYYY-MM-DDThh:mm:ss)"
GLUCOSE = "Glucose Value (mg/dL)"
LABEL_RULE = "Healthy:<5.7; Prediabetes:5.7<=HbA1c<6.5; T2D:>=6.5"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def label_from_hba1c(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if value <= 0:
        raise ValueError("HbA1c must be positive or missing")
    return "Healthy" if value < 5.7 else "Prediabetes" if value < 6.5 else "T2D"


def source_id(value: str) -> int:
    text = str(value).strip()
    if not re.fullmatch(r"\d+", text):
        raise ValueError(f"Invalid source patient number: {text!r}")
    return int(text)


def subject_id(number: int) -> str:
    return f"BIGIDEAS_{number:03d}"


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def device_value(frame: pd.DataFrame, kind: str) -> str:
    values = frame.loc[frame["Event Type"].str.strip().eq(kind), "Patient Info"]
    unique = sorted(set(values.astype(str).str.strip()) - {""})
    if len(unique) > 1:
        raise ValueError(f"Conflicting device metadata for {kind}")
    return unique[0] if unique else ""


def prepare(source_zip: Path, output_root: Path) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output is not empty: {output_root}. Choose a new output directory.")
    source_bytes = source_zip.read_bytes()
    with ZipFile(BytesIO(source_bytes)) as archive:
        names = [i.filename for i in archive.infolist() if not i.is_dir()
                 and "__MACOSX" not in Path(i.filename).parts and not Path(i.filename).name.startswith(".")]
        if len(names) != len(set(names)):
            raise ValueError("Archive has duplicate member names")
        demo_names = [n for n in names if Path(n).name.lower() == "demographics.csv"]
        if len(demo_names) != 1:
            raise ValueError("Expected exactly one Demographics.csv")
        demo_name = demo_names[0]
        demographics = pd.read_csv(BytesIO(archive.read(demo_name)), dtype=str, keep_default_na=False)
        if not {"ID", "Gender", "HbA1c"} <= set(demographics):
            raise ValueError("Demographics.csv requires ID, Gender, HbA1c")
        demographics["source_id"] = demographics.ID.map(source_id)
        if demographics.source_id.duplicated().any():
            raise ValueError("Duplicate Demographics patient IDs")
        hba1c = pd.to_numeric(demographics.HbA1c.replace("", np.nan), errors="coerce")
        if ((demographics.HbA1c.str.strip() != "") & hba1c.isna()).any():
            raise ValueError("Nonnumeric HbA1c; do not replace with a diagnosis or inferred value")
        demographics["hba1c_value"] = hba1c
        demographics = demographics.set_index("source_id")
        cgm_members = {}
        for name in names:
            match = re.fullmatch(r"Dexcom_(\d+)\.csv", Path(name).name, flags=re.I)
            if match:
                number = source_id(match.group(1))
                if number in cgm_members:
                    raise ValueError(f"Multiple CGM files for source ID {number}")
                cgm_members[number] = name
        if not cgm_members or set(cgm_members) != set(demographics.index):
            raise ValueError("CGM file numbers and Demographics IDs must match one-to-one")

        output_root.mkdir(parents=True, exist_ok=True)
        metadata_dir = output_root / "metadata"
        metadata_dir.mkdir()
        labels, clinical, manifests, gaps, excluded, hashes = [], [], [], [], [], []
        hashes.append({"source_file": demo_name, "bytes": len(archive.read(demo_name)),
                       "sha256": digest(archive.read(demo_name)), "role": "patient_labels_and_gender"})

        for number, member in sorted(cgm_members.items()):
            payload = archive.read(member)
            raw = pd.read_csv(BytesIO(payload), dtype=str, keep_default_na=False)
            required = {TIME, GLUCOSE, "Event Type", "Patient Info", "Source Device ID"}
            if not required <= set(raw):
                raise ValueError(f"Missing columns in {member}: {sorted(required - set(raw))}")
            sid = subject_id(number)
            person = demographics.loc[number]
            value = float(person.hba1c_value)
            label = label_from_hba1c(value)
            raw["source_row"] = np.arange(len(raw)) + 2  # Header is physical CSV line 1.
            is_egv = raw["Event Type"].str.strip().eq("EGV")
            for kind, count in raw.loc[~is_egv, "Event Type"].value_counts(dropna=False).items():
                excluded.append({"subject_id": sid, "source_file": member, "event_type": kind,
                                 "rows_excluded": int(count), "reason": "not_an_EGV_measurement"})
            egv = raw.loc[is_egv].copy()
            egv["parsed_time"] = pd.to_datetime(egv[TIME], errors="coerce", format="mixed")
            bad_time = egv.parsed_time.isna()
            invalid_rows = egv.loc[bad_time, ["source_row", TIME, GLUCOSE]]
            if len(invalid_rows):
                invalid_rows = invalid_rows.assign(subject_id=sid, source_file=member)
                save_csv(invalid_rows, metadata_dir / f"invalid_timestamp_{sid}.csv")
            egv = egv.loc[~bad_time].sort_values(["parsed_time", "source_row"], kind="stable")
            if egv.empty:
                raise ValueError(f"No valid EGV timestamps for {sid}")
            numeric = pd.to_numeric(egv[GLUCOSE], errors="coerce")
            numeric = numeric.where(np.isfinite(numeric))
            out = pd.DataFrame({
                "subject_id": sid, "Timestamp": egv.parsed_time,
                "CGM": numeric / DIVISOR, "CGM_unit": "mmol/L", "label": label, "hba1c": value,
                "dataset": DATASET, "glucose_original_mgdl": numeric,
                "sensor_code": egv[GLUCOSE].where(numeric.isna(), ""), "source_dataset": DATASET,
                "source_file": member, "source_row": egv.source_row,
                "event_type": egv["Event Type"], "source_device_id": egv["Source Device ID"],
                "source_timestamp": egv[TIME], "raw_glucose_value": egv[GLUCOSE],
            })
            duplicate_rows = int(out.Timestamp.duplicated().sum())
            if duplicate_rows:
                duplicate = out.loc[out.Timestamp.duplicated(keep=False)]
                save_csv(duplicate, metadata_dir / f"duplicate_timestamp_{sid}.csv")
                # Match Hall: same-patient exact timestamp -> numeric mean;
                # retain the provenance of every source record that contributed.
                aggregation = {c: "first" for c in out if c != "Timestamp"}
                aggregation.update({"CGM": "mean", "glucose_original_mgdl": "mean",
                                    "source_row": lambda s: ";".join(s.astype(str)),
                                    "raw_glucose_value": lambda s: ";".join(s.astype(str)),
                                    "sensor_code": lambda s: ";".join(sorted(set(s) - {""}))})
                out = out.groupby("Timestamp", as_index=False, sort=True).agg(aggregation)[out.columns]
            out = out.reset_index(drop=True)
            diff = out.Timestamp.diff().dt.total_seconds()
            for i in out.index[diff > 25 * 60]:
                gaps.append({"subject_id": sid, "start_time": out.Timestamp.iloc[i - 1].isoformat(sep=" "),
                             "end_time": out.Timestamp.iloc[i].isoformat(sep=" "),
                             "gap_minutes": float(diff.iloc[i] / 60), "action": "preserved_without_interpolation"})
            first, last = out.Timestamp.iloc[0], out.Timestamp.iloc[-1]
            manifests.append({"subject_id": sid, "status": "done", "dataset": DATASET,
                "source_file": member, "rows_input": len(raw), "egv_rows_input": int(is_egv.sum()),
                "non_egv_rows_excluded": int((~is_egv).sum()), "invalid_timestamp_rows": int(bad_time.sum()),
                "duplicate_timestamp_rows_merged": duplicate_rows, "rows_output": len(out),
                "valid_cgm": int(out.CGM.notna().sum()), "missing_cgm": int(out.CGM.isna().sum()),
                "start_time": first.isoformat(sep=" "), "end_time": last.isoformat(sep=" "),
                "duration_days": (last - first).total_seconds() / 86400,
                "median_interval_seconds": float(diff.median()), "max_gap_minutes": float(diff.max() / 60),
                "gaps_over_25_minutes": int((diff > 1500).sum()), "outcome_label_summary": label, "error": ""})
            labels.append({"subject_id": sid, "label": label, "hba1c": value,
                           "diagnosis_original": "", "dataset": DATASET, "label_source": "HbA1c",
                           "label_rule": LABEL_RULE, "hba1c_unit": "%", "source_subject_id": f"{number:03d}"})
            clinical.append({"subject_id": sid, "source_subject_id": f"{number:03d}", "dataset": DATASET,
                "gender": str(person.Gender).strip().upper(), "age": np.nan, "bmi": np.nan,
                "height_cm": np.nan, "weight_kg": np.nan, "hba1c": value, "hba1c_unit": "%",
                "hba1c_measurement_date": "", "gender_source": "Demographics.csv:Gender",
                "age_source": "not_provided", "bmi_source": "not_provided",
                "source_dob_text_unverified": device_value(raw, "DateOfBirth"),
                "dob_verification": "not_verified_not_used_to_derive_age",
                "source_file": member, "demographics_source_file": demo_name})
            out["Timestamp"] = out.Timestamp.dt.strftime("%Y-%m-%d %H:%M:%S")
            save_csv(out, output_root / "raw" / "train" / f"{sid}.csv")
            hashes.append({"source_file": member, "bytes": len(payload), "sha256": digest(payload), "role": "dexcom_export"})

    label_frame = pd.DataFrame(labels)
    manifest = pd.DataFrame(manifests)
    save_csv(label_frame, output_root / "labels" / "BIG_IDEAS_labels_hba1c.csv")
    save_csv(pd.DataFrame(clinical), metadata_dir / "clinical_metadata.csv")
    save_csv(manifest, metadata_dir / "processing_manifest.csv")
    save_csv(pd.DataFrame(hashes), metadata_dir / "source_file_manifest.csv")
    save_csv(pd.DataFrame(gaps, columns=["subject_id", "start_time", "end_time", "gap_minutes", "action"]),
             metadata_dir / "gap_audit.csv")
    save_csv(pd.DataFrame(excluded), metadata_dir / "excluded_event_counts.csv")
    mapping = [
        ("subject_id", "Dexcom_NNN.csv filename; Demographics.ID", "BIGIDEAS_ + 3-digit source ID"),
        ("Timestamp", TIME, "parse literal local wall-clock time; timezone unavailable"),
        ("CGM", GLUCOSE, f"EGV only; numeric mg/dL / {DIVISOR}"),
        ("CGM_unit", "Glucose Value (mg/dL) header", "mmol/L"),
        ("label", "Demographics.HbA1c", LABEL_RULE),
        ("hba1c", "Demographics.HbA1c", "unchanged; % convention, no source unit annotation"),
        ("gender", "Demographics.Gender", "retain FEMALE/MALE text; no numeric recoding"),
        ("age", "not provided", "leave missing; do not derive from unverified device DOB"),
        ("bmi", "not provided", "leave missing"),
    ]
    save_csv(pd.DataFrame(mapping, columns=["output_field", "source_field", "transformation"]), metadata_dir / "field_mapping.csv")
    counts = {name: int(label_frame.label.eq(name).sum()) for name in ["Healthy", "Prediabetes", "T2D"]}
    summary = {"patients": len(manifest), "labels": counts, "rows_input": int(manifest.rows_input.sum()),
        "egv_rows_input": int(manifest.egv_rows_input.sum()), "rows_output": int(manifest.rows_output.sum()),
        "non_egv_rows_excluded": int(manifest.non_egv_rows_excluded.sum()),
        "invalid_timestamp_rows": int(manifest.invalid_timestamp_rows.sum()),
        "duplicate_timestamp_rows_merged": int(manifest.duplicate_timestamp_rows_merged.sum()),
        "valid_cgm": int(manifest.valid_cgm.sum()), "missing_cgm": int(manifest.missing_cgm.sum()),
        "gaps_over_25_minutes": int(manifest.gaps_over_25_minutes.sum()),
        "max_gap_minutes": float(manifest.max_gap_minutes.max()),
        "clinical_available": ["gender"], "clinical_unavailable": ["age", "bmi", "height_cm", "weight_kg"]}
    assert summary["rows_input"] == summary["egv_rows_input"] + summary["non_egv_rows_excluded"]
    assert summary["egv_rows_input"] == summary["rows_output"] + summary["invalid_timestamp_rows"] + summary["duplicate_timestamp_rows_merged"]
    config = {"dataset": DATASET, "source_archive": source_zip.name, "source_archive_sha256": digest(source_bytes),
        "reference_format": "user-supplied hall2018_preprocessing.zip", "input_glucose_unit": "mg/dL",
        "output_glucose_unit": "mmol/L", "conversion": f"CGM = numeric glucose / {DIVISOR}",
        "event_filter": "Event Type == EGV; alert thresholds are not measurements", "timestamp_source": TIME,
        "timezone": "not_provided; preserve wall-clock time without timezone conversion",
        "invalid_timestamps": "exclude and audit", "nonnumeric_glucose": "missing CGM; retain sensor_code",
        "duplicate_timestamps": "same patient exact timestamps: mean glucose; source rows retained in audit",
        "gap_policy": "preserve all gaps; separately list gaps >25 minutes; no filling",
        "label_source": "Demographics.csv HbA1c", "hba1c_unit": "%",
        "hba1c_unit_note": "source header has no unit; use the same percent convention as the existing project; values 5.3-6.4",
        "hba1c_measurement_date": "not_provided", "label_rule": LABEL_RULE,
        "smoothing": False, "resampling": False, "interpolation": False, "peak_detection": False,
        "feature_engineering": False, "train_validation_split": False,
        "age_policy": "do not derive from unverified device DOB; 15/16 source DOB values are 1970-01-01",
        "summary": summary}
    write_json(metadata_dir / "preprocessing_config.json", config)
    dictionary = {"raw/train": {"subject_id": "Globally namespaced patient ID; repeated at every CGM timepoint",
        "Timestamp": "Original observed local wall-clock time, sorted, no resampling", "CGM": "CGM in mmol/L",
        "CGM_unit": "Always mmol/L", "label": "HbA1c-derived patient class, identical to labels CSV",
        "hba1c": "Patient HbA1c using project percent convention; an outcome, not a covariate",
        "dataset": DATASET, "glucose_original_mgdl": "Source numeric glucose before conversion",
        "sensor_code": "Source nonnumeric glucose token, blank when numeric", "source_dataset": DATASET,
        "source_file": "Member path inside original source archive", "source_row": "Physical CSV line including header line 1",
        "event_type": "EGV", "source_device_id": "Original source device label", "source_timestamp": "Original timestamp text",
        "raw_glucose_value": "Original glucose field text"},
        "clinical_metadata": {"gender": "Source FEMALE/MALE, not currently a v1.2.0 model feature",
            "age,bmi,height_cm,weight_kg": "Absent in source; blank, not zero or inferred",
            "source_dob_text_unverified": "Device birth-date text, retained for audit only, not verified clinical DOB"},
        "raw_train_folder": "Storage convention only; does not constitute an assigned training split"}
    write_json(metadata_dir / "data_dictionary.json", dictionary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-zip", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("BIG_IDEAS_studio_ready"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.input_zip, args.output_root), ensure_ascii=False, indent=2))
