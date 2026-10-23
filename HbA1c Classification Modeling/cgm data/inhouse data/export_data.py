"""Join saved subject-specific CGM predictions to their source events and labs."""

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parents[2] / "SEB_CGM_alignment"
PREDICTIONS = HERE.parent / "Subject-level SEG modeling/data/model_results.json"
sys.path.insert(0, str(ENGINE))
import concentration_mapping_dashboard as core


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(value, prefix=""):
    output = {}
    if isinstance(value, dict):
        for key, item in value.items():
            output.update(flatten(item, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        output[prefix] = json.dumps(value, allow_nan=False)
    else:
        output[prefix] = value
    return output


def write_csv(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (HERE/name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return fields


def main():
    saved = json.loads(PREDICTIONS.read_text())
    # Ordinal dashboard event IDs can change after annotation edits. Fail closed.
    for name, expected in saved["source_sha256"].items():
        if digest(ENGINE/"state"/name) != expected:
            raise ValueError(f"Source annotations changed: {name}; regenerate predictions before joining")
    event_lookup = {e["id"]:e for e in core.event_list()}
    pairs = core.alignment.load_peak_pair_state()["cases"]
    labs = core.alignment.load_lab_features_by_participant()
    if not labs:
        raise ValueError("Lab metadata could not be loaded")
    subjects = sorted({p["subject_id"] for p in saved["points"]})
    metadata, patients = {}, []
    for subject in subjects:
        lab = {k:v for k,v in labs.get(subject, {}).items() if k.lower() not in ("intials", "initials")}
        # Preserve categorical codes and original lab representations verbatim.
        item = dict(subject_id=subject, metadata_available=subject in labs,
                    hba1c_available=lab.get("hba1c") not in (None,"","NA","NAN"),
                    **{"patient."+k:v for k,v in lab.items()})
        patients.append(item)
        metadata[subject] = item
    metrics = {e["event_id"]:e for e in saved["event_metrics"]}
    event_rows, nested_events, by_event = [], {}, {}
    for eid in dict.fromkeys(p["event_id"] for p in saved["points"]):
        event = event_lookup[eid]
        pair = pairs[event["case_path"]]["paired_features"][event["pair_index"]]
        meta = pair["selected_peak_metadata"]
        identity = [event["subject_id"], event["case_path"], meta["seg"]["aggregate_id"], meta["cgm"]["aggregate_id"]]
        uid = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:20]
        points = [p for p in saved["points"] if p["event_id"] == eid]
        for point in points:
            assert point["subject_id"] == event["subject_id"]
            assert point["day"] == event["day"] and point["event_name"] == event["name"]
            assert core.parse_dt(point["event_start_timestamp"]) == core.parse_dt(meta["cgm"]["trough1_timestamp"])
            assert point["split"] == metrics[eid]["split"]
        # Keep every saved SEG/CGM metadata field, including AUC, values, and methods.
        details = {signal:meta[signal] for signal in ("seg", "cgm")}
        pair_details = {k:v for k,v in pair.items() if k != "selected_peak_metadata" and
                        not k.startswith("sel_") and "_sel" not in k and k != "triplet_complete"}
        base = dict(event_uid=uid, event_id=eid, subject_id=event["subject_id"],
                    day=event["day"], event_name=event["name"], split=metrics[eid]["split"],
                    case_path=event["case_path"], pair_index=event["pair_index"])
        flat = dict(base, **flatten(details), **flatten(pair_details,"pair"),
                    **flatten({k:v for k,v in metrics[eid].items() if k in
                               ("n","mard_percent","rmse","r_squared")},"prediction_metrics"))
        event_rows.append(flat)
        by_event[eid] = flat
        nested_events[eid] = dict(base, features=details, pair_metadata=pair_details,
                                  prediction_metrics=metrics[eid], timeseries=[])
    assert len({e["event_uid"] for e in event_rows}) == len(event_rows)
    timeseries, joined, seen = [], [], set()
    feature_names = ["intercept", "mean_seg_normalized", "mean_seg_normalized_squared",
                     "mean_seg_normalized_cubed", "mean_seg_rate_per_minute", "mean_signed_sigmoid_phase"]
    for point in sorted(saved["points"], key=lambda p:(p["subject_id"],p["timestamp"],p["event_id"])):
        eid, subject = point["event_id"], point["subject_id"]
        identity = (subject,point["timestamp"])
        assert identity not in seen, "Repeated CGM target timestamp within subject"
        seen.add(identity)
        row = dict(event_uid=by_event[eid]["event_uid"], subject_id=subject, event_id=eid,
                   event_name=point["event_name"], day=point["day"], split=point["split"],
                   cgm_timestamp=point["timestamp"], elapsed_seconds_from_cgm_event_start=point["elapsed_seconds"],
                   interval=point["interval"], actual_cgm_mmol_L=point["actual_cgm"],
                   predicted_cgm_mmol_L=point["predicted_cgm"],
                   **{"model_input."+name:point[f"feature_{i}"] for i,name in enumerate(feature_names)})
        timeseries.append(row)
        joined.append(dict(by_event[eid], **{k:v for k,v in metadata[subject].items() if k!="subject_id"},
                           **{k:v for k,v in row.items() if k not in by_event[eid]}))
        nested_events[eid]["timeseries"].append(row)
    counts = dict(subjects=len(subjects), events=len(event_rows), timeseries_points=len(timeseries),
                  train_points=sum(p["split"]=="train" for p in timeseries),
                  held_out_points=sum(p["split"]=="validation" for p in timeseries),
                  subjects_with_hba1c=sum(p["hba1c_available"] for p in patients))
    manifest = dict(created_at=datetime.now().isoformat(), counts=counts,
        prediction_source=str(PREDICTIONS.relative_to(HERE.parents[3])),
        source_sha256={"model_results.json":digest(PREDICTIONS), **saved["source_sha256"],
                       "Lab_data.xlsx":digest(Path(core.alignment.LAB_DATA_XLSX))},
        model=saved["model"], smoothing_ratio=saved["smoothing_ratio"], split=saved["split"],
        missing_metadata_subjects=[s for s in subjects if not metadata[s]["metadata_available"]],
        missing_hba1c_subjects=[s for s in subjects if not metadata[s]["hba1c_available"]],
        excluded_from_prediction_source=saved["exclusions"],
        source_model_coefficients=saved["coefficients"],
        omitted_lab_fields=["intials"],
        unit_notes={"CGM":"mmol/L", "durations":"seconds unless field explicitly states otherwise",
                    "saved_event_derivatives":"Derivative with respect to interval time normalized to 0..1; preserve source time_normalization and value_source fields",
                    "model_derivative":"Min-max normalized SEG change per native sensor minute, bin-averaged",
                    "timestamps":"Source local wall-clock timestamps (study timezone Asia/Singapore); not Unix milliseconds",
                    "patient_metadata":"Values/categorical codes preserved as recorded; no inferred category labels or unit conversions"})
    tables = {"predicted_cgm_timeseries.csv":timeseries, "event_features.csv":event_rows,
              "patient_metadata.csv":patients, "timeseries_with_features_and_metadata.csv":joined}
    dictionary = {}
    for filename, rows in tables.items():
        columns = write_csv(filename, rows)
        dictionary[filename] = {field:dict(nonmissing_count=sum(r.get(field) not in (None,"") for r in rows),
            source="Lab_data.xlsx" if field.startswith("patient.") else
                   "alignment_annotations.json: peak_pair_features" if field.startswith(("seg.","cgm.","pair.")) else
                   "Saved predictions or verified join identifiers") for field in columns}
    manifest["table_shapes"] = {name:dict(rows=len(rows),columns=len(dictionary[name])) for name,rows in tables.items()}
    (HERE/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False))
    (HERE/"data_dictionary.json").write_text(json.dumps(dictionary,indent=2,allow_nan=False))
    subject_dir = HERE/"by_subject"
    subject_dir.mkdir(exist_ok=True)
    for subject in subjects:
        payload = dict(subject_id=subject, patient_metadata=metadata[subject],
                       cgm_model_coefficients=saved["coefficients"][subject],
                       events=[e for e in nested_events.values() if e["subject_id"]==subject])
        (subject_dir/f"{subject}.json").write_text(json.dumps(payload,indent=2,allow_nan=False))
    # Confirm round-trip row counts and exact numerical predictions.
    with (HERE/"predicted_cgm_timeseries.csv").open() as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported)==len(saved["points"])
    original = {(p["subject_id"],p["timestamp"]):p for p in saved["points"]}
    for row in exported:
        p = original[row["subject_id"],row["cgm_timestamp"]]
        assert float(row["predicted_cgm_mmol_L"])==p["predicted_cgm"]
        assert float(row["actual_cgm_mmol_L"])==p["actual_cgm"]
    print(json.dumps(manifest["counts"],indent=2))
    print("Verified source hashes, event identities, unique timestamps, and exact prediction round-trip.")


if __name__ == "__main__":
    main()
