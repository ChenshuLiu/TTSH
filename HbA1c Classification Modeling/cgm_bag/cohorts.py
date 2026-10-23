"""Explicit public dataset cohorts for modality-specific training.

Dataset membership is declared by PublicData, never guessed from patient IDs.
Metadata eligibility is evaluated before training on labeled patients with
eligible events: every selected field must have at least one finite value in
each included dataset. Individual missing values still use train-only imputation.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from .metadata import CLINICAL_FEATURES


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _dataset_id(datasets):
    return "datasets_" + _digest(sorted(datasets))[:12]


def _source_membership(public, segments):
    patients = public.patients.copy()
    if "subject_id" not in patients or "label" not in patients:
        raise ValueError("Public patients need subject_id and label for dataset cohort selection.")
    patients["subject_id"] = patients.subject_id.astype(str)
    patients["dataset"] = patients.get("dataset", pd.Series("public", index=patients.index)).fillna("public").astype(str).str.strip()
    patients.loc[patients.dataset.eq(""), "dataset"] = "public"
    for sid, frame in patients.groupby("subject_id"):
        if not sid.strip() or frame.dataset.nunique() != 1 or frame.label.nunique(dropna=False) != 1:
            raise ValueError(f"Conflicting public patient/dataset identity for {sid!r}; use globally unique patient IDs.")
    patients = patients.drop_duplicates("subject_id").set_index("subject_id")
    if {"subject_id", "dataset"} <= set(public.timeseries):
        declarations = public.timeseries[["subject_id", "dataset"]].drop_duplicates()
        for row in declarations.itertuples(index=False):
            sid = str(row.subject_id)
            if sid not in patients.index or str(row.dataset) != patients.loc[sid, "dataset"]:
                raise ValueError(f"Conflicting public waveform/patient dataset for {sid!r}.")
    seen = set()
    for segment in segments:
        sid, uid = str(segment.subject_id), str(segment.segment_id)
        if sid not in patients.index:
            raise ValueError(f"Extracted patient {sid!r} has no declared public dataset.")
        if str(segment.dataset) != patients.loc[sid, "dataset"] or int(segment.label) != int(patients.loc[sid, "label"]):
            raise ValueError(f"Conflicting extracted patient/dataset/label identity for {sid!r}.")
        if str(segment.dataset).lower() == "private":
            raise ValueError("Private events cannot be selected as a public training cohort.")
        if not uid.strip() or (sid, uid) in seen:
            raise ValueError("Public cohort events require unique nonempty patient/event keys.")
        seen.add((sid, uid))
    return patients


def _clinical_frame(public):
    clinical = public.clinical.copy()
    if clinical.empty and "subject_id" not in clinical:
        return pd.DataFrame(columns=["subject_id"]).set_index("subject_id")
    if "subject_id" not in clinical:
        raise ValueError("Public clinical metadata needs subject_id.")
    clinical["subject_id"] = clinical.subject_id.astype(str)
    if clinical.subject_id.duplicated().any():
        raise ValueError("Public clinical metadata must contain one canonical row per patient.")
    return clinical.set_index("subject_id")


def dataset_coverage(public, segments, metadata_features=()):
    """One row per declared dataset; count people once regardless of event count."""
    segments = list(segments)
    features = list(metadata_features)
    if len(set(features)) != len(features) or not set(features) <= set(CLINICAL_FEATURES):
        raise ValueError("Select unique supported clinical metadata fields.")
    patients, clinical = _source_membership(public, segments), _clinical_frame(public)
    rows = []
    for dataset in sorted(patients.dataset.unique()):
        source = patients.loc[patients.dataset.eq(dataset)]
        items = [item for item in segments if str(item.subject_id) in source.index]
        eligible_ids = sorted({str(item.subject_id) for item in items})
        labeled_ids = sorted({str(item.subject_id) for item in items if int(item.label) >= 0})
        row = {"dataset": dataset, "imported_patient_count": len(source), "patient_count": len(eligible_ids),
               "labeled_patient_count": len(labeled_ids), "event_count": len(items),
               "labeled_event_count": sum(int(item.label) >= 0 for item in items)}
        missing = []
        for feature in features:
            values = (pd.to_numeric(clinical[feature].reindex(labeled_ids), errors="coerce").to_numpy(float)
                      if feature in clinical else np.full(len(labeled_ids), np.nan))
            available = int(np.isfinite(values).sum())
            row[f"{feature}_available_patients"] = available
            row[f"{feature}_coverage"] = available / len(labeled_ids) if labeled_ids else 0.
            if not available:
                missing.append(feature)
        row.update(metadata_eligible=bool(features) and bool(labeled_ids) and not missing,
                   metadata_missing_fields=",".join(missing), metadata_features=",".join(features))
        rows.append(row)
    return pd.DataFrame(rows)


def _segment_fingerprint(segments):
    assignment = [(str(item.subject_id), str(item.segment_id), int(item.label),
                   hashlib.sha256(np.asarray(item.values, dtype=np.float32).tobytes()).hexdigest())
                  for item in sorted(segments, key=lambda item: (str(item.subject_id), str(item.segment_id)))]
    # Match training.patient_stratified_split's existing cohort fingerprint.
    return hashlib.sha256(json.dumps(assignment, separators=(",", ":")).encode()).hexdigest()


def _cohort(public, segments, datasets, patients, metadata_features, role):
    items = [item for item in segments if patients.loc[str(item.subject_id), "dataset"] in datasets]
    if not items:
        raise ValueError("Selected public datasets have no eligible extracted events.")
    ids = sorted({str(item.subject_id) for item in items})
    actual_datasets = sorted(set(patients.loc[ids, "dataset"]))
    labeled_ids = sorted({str(item.subject_id) for item in items if int(item.label) >= 0})
    subject_datasets = {sid: str(patients.loc[sid, "dataset"]) for sid in ids}
    identity = {"datasets": actual_datasets, "subject_datasets": subject_datasets,
                "labeled_patient_ids": labeled_ids, "data_fingerprint_sha256": _segment_fingerprint(items)}
    audit = {"version": 1, "cohort_id": _dataset_id(actual_datasets), "cohort_label": " + ".join(actual_datasets),
             **identity, "cohort_membership_sha256": _digest(identity), "requested_datasets": sorted(datasets),
             "patient_count": len(ids), "labeled_patient_count": len(labeled_ids), "event_count": len(items),
             "labeled_event_count": sum(int(item.label) >= 0 for item in items),
             "dataset_patient_counts": {name: sum(value == name for value in subject_datasets.values()) for name in actual_datasets},
             "metadata_eligibility_fields": list(metadata_features), "roles": [role],
             "comparison_note": "Only models with this same public cohort and patient split form a controlled modality comparison."}
    clinical = public.clinical.copy()
    if "subject_id" in clinical:
        clinical = clinical.loc[clinical.subject_id.astype(str).isin(ids)].copy()
    return {"segments": items, "clinical": clinical, "audit": audit}


def build_cohort_jobs(public, segments, architectures, modes, selected_event_features,
                      feature_variants=("all",), *, metadata_features=(), nonmetadata_datasets=None,
                      metadata_datasets=None, matched_baselines=False):
    """Return jobs and keyed cohorts; metadata modes use only eligible datasets.

    Matched baselines repeat the selected nonmetadata modes on the metadata
    cohort. Actual identical cohorts/jobs are deduplicated. When all models use
    one cohort, existing model keys remain unchanged for compatibility.
    """
    from .ablation import build_jobs
    segments = list(segments)
    base_jobs = build_jobs(architectures, modes, selected_event_features, feature_variants)
    coverage = dataset_coverage(public, segments, metadata_features)
    patients = _source_membership(public, segments)
    available = set(coverage.dataset)
    use_metadata = any(job["use_metadata"] for job in base_jobs)
    nonmeta = sorted(available if nonmetadata_datasets is None else set(nonmetadata_datasets))
    meta = sorted(set(coverage.loc[coverage.metadata_eligible, "dataset"]) if metadata_datasets is None else set(metadata_datasets))
    for names in (nonmeta, meta):
        if set(names) - available:
            raise ValueError("Unknown public datasets: " + ", ".join(sorted(set(names) - available)))
    if use_metadata:
        if not metadata_features or not meta:
            raise ValueError("Metadata modes require selected clinical fields and at least one metadata-capable dataset.")
        bad = coverage.loc[coverage.dataset.isin(meta) & ~coverage.metadata_eligible]
        if len(bad):
            detail = "; ".join(f"{row.dataset}: {row.metadata_missing_fields or 'no labeled eligible patients'}" for row in bad.itertuples())
            raise ValueError("Metadata datasets have no observed values for selected fields: " + detail)
    if any(not job["use_metadata"] for job in base_jobs) and not nonmeta:
        raise ValueError("Select at least one dataset for nonmetadata models.")
    cohorts, jobs, seen = {}, [], set()
    for job in base_jobs:
        choices = [(meta, "metadata") if job["use_metadata"] else (nonmeta, "all_public")]
        if not job["use_metadata"] and matched_baselines and use_metadata:
            choices.append((meta, "matched_nonmetadata_baseline"))
        for names, role in choices:
            cohort = _cohort(public, segments, names, patients, metadata_features if role != "all_public" else (), role)
            audit = cohort["audit"]
            cohort_id = audit["cohort_id"]
            if cohort_id in cohorts:
                existing = cohorts[cohort_id]["audit"]
                existing["roles"] = sorted(set(existing["roles"] + [role]))
                existing["metadata_eligibility_fields"] = sorted(set(existing["metadata_eligibility_fields"] + audit["metadata_eligibility_fields"]))
            else:
                cohorts[cohort_id] = cohort
            pair = (cohort_id, job["model_key"])
            if pair in seen:
                continue
            seen.add(pair)
            jobs.append({**job, "cohort_id": cohort_id, "cohort_label": audit["cohort_label"],
                         "cohort_datasets": audit["datasets"], "cohort_role": role})
    if len(cohorts) > 1:
        for job in jobs:
            job["model_key"] += "__" + job["cohort_id"]
    return jobs, cohorts


def annotate_result(result, cohort):
    """Attach verifiable public membership after training and before saving."""
    audit = copy.deepcopy(cohort["audit"])
    previous = result.split.get("public_cohort")
    result.split["public_cohort"] = audit
    try:
        cohort_audit(result)
    except Exception:
        if previous is None:
            result.split.pop("public_cohort", None)
        else:
            result.split["public_cohort"] = previous
        raise
    return result


def cohort_audit(result):
    """Validate explicit cohorts; old unannotated models stay in one strict group."""
    audit = result.split.get("public_cohort")
    if audit is None:
        return None
    if not isinstance(audit, dict) or audit.get("version") != 1:
        raise ValueError("Invalid explicit public cohort audit.")
    try:
        datasets = audit["datasets"]
        subject_datasets = audit["subject_datasets"]
        labeled_ids = audit["labeled_patient_ids"]
        identity = {"datasets": datasets, "subject_datasets": subject_datasets,
                    "labeled_patient_ids": labeled_ids, "data_fingerprint_sha256": audit["data_fingerprint_sha256"]}
        valid = (bool(datasets) and datasets == sorted(set(datasets))
                 and set(subject_datasets.values()) == set(datasets)
                 and audit["cohort_id"] == _dataset_id(datasets)
                 and audit["cohort_membership_sha256"] == _digest(identity)
                 and sorted(set(result.split["train_ids"]) | set(result.split["val_ids"])) == labeled_ids
                 and set(labeled_ids) <= set(subject_datasets)
                 and set(labeled_ids) | set(result.split.get("excluded_unlabeled_ids", [])) == set(subject_datasets)
                 and audit["patient_count"] == len(subject_datasets)
                 and audit["labeled_patient_count"] == len(labeled_ids)
                 and set(result.config.metadata_features) <= set(audit["metadata_eligibility_fields"])
                 and audit["event_count"] >= audit["labeled_event_count"]
                 and audit["labeled_event_count"] == result.split["train_segment_count"] + result.split["val_segment_count"]
                 and audit["data_fingerprint_sha256"] == result.split["data_fingerprint_sha256"])
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Explicit public cohort audit disagrees with trained patient/event membership.")
    return audit


def cohort_columns(result):
    audit = cohort_audit(result)
    return {"public_cohort_id": audit["cohort_id"] if audit else "legacy_public",
            "public_cohort_label": audit["cohort_label"] if audit else "Unspecified legacy public cohort",
            "public_datasets": ",".join(audit["datasets"]) if audit else "",
            "public_patient_count": audit["labeled_patient_count"] if audit else len(result.split["train_ids"]) + len(result.split["val_ids"]),
            "public_event_count": audit["labeled_event_count"] if audit else (result.split.get("train_segment_count") or 0) + (result.split.get("val_segment_count") or 0),
            "public_cohort_explicit": audit is not None}
