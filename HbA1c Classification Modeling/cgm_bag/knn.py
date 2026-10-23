"""CGM-only event KNN with patient-disjoint fitting and a pickle-free checkpoint."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from .training import CLASS_NAMES, PROBABILITY_COLUMNS, CGMScaler, patient_stratified_split


@dataclass
class KNNConfig:
    n_neighbors: int = 5
    weights: str = "distance"
    metric: str = "euclidean"
    val_fraction: float = .25
    seed: int = 42
    length: int = 64

    def validate(self):
        for name in ("n_neighbors", "length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.length < 4:
            raise ValueError("length must be at least 4.")
        if self.weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'.")
        if self.metric not in {"euclidean", "manhattan", "chebyshev", "minkowski", "cosine"}:
            raise ValueError("Choose euclidean, manhattan, chebyshev, minkowski, or cosine metric.")
        if isinstance(self.val_fraction, bool) or not np.isfinite(self.val_fraction) or not 0 < self.val_fraction < 1:
            raise ValueError("val_fraction must lie strictly between 0 and 1.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2 ** 32:
            raise ValueError("seed must be an integer in [0, 2**32).")

    @property
    def architecture(self):
        return "knn"

    @property
    def sample_unit(self):
        return "event"

    @property
    def use_cgm(self):
        return True


@dataclass
class KNNResult:
    model: KNeighborsClassifier
    scaler: CGMScaler
    config: KNNConfig
    split: dict
    train_values: np.ndarray
    train_labels: np.ndarray
    train_subject_ids: np.ndarray
    train_segment_ids: np.ndarray
    cohort_manifest: list
    checkpoint_path: str | None = None

    @property
    def train_ids(self):
        return self.split["train_ids"]

    @property
    def val_ids(self):
        return self.split["val_ids"]


def _records(segments, length):
    """Copy only IDs, label and CGM; metadata/info/features are never accessed."""
    records, seen, labels = [], set(), {}
    for segment in segments:
        raw_sid, raw_eid = segment.subject_id, segment.segment_id
        if raw_sid is None or raw_eid is None:
            raise ValueError("Patient and segment IDs must be nonempty.")
        sid, eid = str(raw_sid), str(raw_eid)
        if not sid.strip() or not eid.strip() or sid != sid.strip() or eid != eid.strip():
            raise ValueError("Patient and segment IDs must be nonempty without surrounding whitespace.")
        if (sid, eid) in seen:
            raise ValueError(f"Duplicate event: {(sid, eid)}")
        seen.add((sid, eid))
        label = segment.label
        if isinstance(label, (bool, np.bool_)) or label not in (-1, 0, 1, 2):
            raise ValueError("Labels must be -1, 0, 1 or 2.")
        label = int(label)
        if sid in labels and labels[sid] != label:
            raise ValueError(f"Patient {sid} has inconsistent event labels.")
        labels[sid] = label
        values = np.array(segment.values, dtype=np.float32, copy=True)
        if values.shape != (length,) or not np.isfinite(values).all():
            raise ValueError(f"Event {(sid, eid)} requires {length} finite CGM values.")
        records.append(SimpleNamespace(subject_id=sid, segment_id=eid, label=label, values=values))
    return records


def _manifest(records):
    return [[s.subject_id, s.segment_id, s.label, hashlib.sha256(s.values.tobytes()).hexdigest()]
            for s in sorted(records, key=lambda s: (s.subject_id, s.segment_id))]


def _fingerprint(manifest):
    return hashlib.sha256(json.dumps(manifest, separators=(",", ":")).encode()).hexdigest()


def _ids(value, name):
    if (not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() or x != x.strip() for x in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{name} must contain unique nonempty patient IDs.")
    return set(value)


def _audit_split(split, manifest):
    """Check the entire cohort, including waveform hashes, before reusing a split."""
    if not isinstance(split, dict) or not isinstance(manifest, list) or not manifest:
        raise ValueError("A complete patient split and cohort manifest are required.")
    subjects, seen = {}, set()
    for row in manifest:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError("Invalid cohort segment manifest.")
        sid, eid, label, digest = row
        if (not isinstance(sid, str) or not sid.strip() or sid != sid.strip()
                or not isinstance(eid, str) or not eid.strip() or eid != eid.strip()
                or isinstance(label, bool) or not isinstance(label, int) or label not in (-1, 0, 1, 2)
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest) or (sid, eid) in seen):
            raise ValueError("Invalid or duplicate cohort segment manifest entry.")
        if sid in subjects and subjects[sid] != label:
            raise ValueError("Inconsistent patient labels in cohort manifest.")
        subjects[sid] = label
        seen.add((sid, eid))
    if manifest != sorted(manifest, key=lambda row: (row[0], row[1])):
        raise ValueError("Cohort segment manifest must be in canonical order.")
    if split.get("data_fingerprint_sha256") != _fingerprint(manifest):
        raise ValueError("Split data fingerprint does not match these CGM segments.")
    train = _ids(split.get("train_ids"), "train_ids")
    val = _ids(split.get("val_ids"), "val_ids")
    excluded = _ids(split.get("excluded_unlabeled_ids", []), "excluded_unlabeled_ids")
    labeled = {sid for sid, label in subjects.items() if label >= 0}
    if train & val or train | val != labeled or excluded != set(subjects) - labeled:
        raise ValueError("Train/validation must be a complete patient-disjoint partition; exclude only unlabeled patients.")
    if split.get("patient_overlap", []):
        raise ValueError("The saved split records overlapping patients.")
    for name, ids in (("train", train), ("val", val)):
        counts = {class_name: sum(subjects[sid] == label for sid in ids)
                  for label, class_name in enumerate(CLASS_NAMES)}
        if not all(counts.values()):
            raise ValueError("All three classes must be present in BOTH training and validation patients.")
        expected_count = sum(row[0] in ids for row in manifest)
        for field, expected in ((f"{name}_class_counts", counts), (f"{name}_segment_count", expected_count)):
            if field in split and split[field] != expected:
                raise ValueError(f"Saved {field} disagrees with the supplied CGM segments.")
    return train, val


def _fit_model(config, scaler, values, labels):
    if config.n_neighbors > len(values):
        raise ValueError(f"n_neighbors={config.n_neighbors} exceeds {len(values)} training events; choose k using public validation only.")
    if set(labels.tolist()) != {0, 1, 2}:
        raise ValueError("KNN training events must contain all three classes.")
    model = KNeighborsClassifier(n_neighbors=config.n_neighbors, weights=config.weights,
                                 metric=config.metric, algorithm="brute", n_jobs=1)
    model.fit(scaler.transform_values(values), labels)
    return model


def train_knn(segments, config: KNNConfig, segment_config=None, split=None) -> KNNResult:
    """Fit one CGM sequence per event; all splitting and scaling is patient-disjoint."""
    config = copy.deepcopy(config)
    config.validate()
    segments = list(segments)
    if any(str(getattr(s, "dataset", "")).lower() == "private" for s in segments):
        raise ValueError("Private test events cannot be used for KNN training or validation.")
    records = _records(segments, config.length)
    manifest = _manifest(records)
    audited = (patient_stratified_split(records, config.val_fraction, config.seed)
               if split is None else copy.deepcopy(split))
    train_ids, val_ids = _audit_split(audited, manifest)
    if segment_config is not None:
        segment_config.validate()
        if segment_config.length != config.length:
            raise ValueError("Segment configuration and KNN lengths disagree.")
        if "segment_config" in audited and audited["segment_config"] != asdict(segment_config):
            raise ValueError("Reused split has a different segment configuration.")
        audited["segment_config"] = asdict(segment_config)
    elif "segment_config" in audited and audited["segment_config"].get("length") != config.length:
        raise ValueError("Saved segment configuration and KNN lengths disagree.")
    train = sorted((s for s in records if s.subject_id in train_ids), key=lambda s: (s.subject_id, s.segment_id))
    scaler = CGMScaler.fit(train, config.length)
    values = np.stack([s.values for s in train])
    labels = np.asarray([s.label for s in train], dtype=np.int64)
    model = _fit_model(config, scaler, values, labels)
    audited.update({"sample_unit": "event", "validation_unit": "event", "use_cgm": True,
        "metadata_features": [], "event_features": [], "metadata_preprocessing_fit": "not used",
        "event_feature_preprocessing_fit": "not used", "normalization_fit": "training patients only",
        "train_sample_count": len(train), "val_sample_count": sum(s.subject_id in val_ids for s in records),
        "train_patient_count": len(train_ids), "val_patient_count": len(val_ids),
        "training_bags": "KNN stores every public training event as one CGM vector; no bags",
        "inference_bags": "Each event is predicted separately; patient probabilities average event probabilities",
        "estimator": "KNeighborsClassifier", "n_neighbors": config.n_neighbors,
        "neighbor_selection": "Preset or public-validation choice only; never selected using private outcomes"})
    return KNNResult(model, scaler, config, audited, values, labels,
                     np.asarray([s.subject_id for s in train]), np.asarray([s.segment_id for s in train]), manifest)


def predict_knn(result: KNNResult, segments) -> dict:
    """Return event probabilities and unweighted within-patient mean probabilities."""
    records = _records(segments, result.config.length)
    event_columns = ["subject_id", "segment_id", "label", *PROBABILITY_COLUMNS,
                     "predicted_class", "predicted_label", "confidence", "prediction_type"]
    patient_columns = ["subject_id", "label", *PROBABILITY_COLUMNS, "predicted_class",
                       "predicted_label", "confidence", "n_segments"]
    if not np.array_equal(result.model.classes_, [0, 1, 2]):
        raise ValueError("KNN class order must be Healthy, Prediabetes, T2D.")
    probabilities = (result.model.predict_proba(result.scaler.transform_values(np.stack([s.values for s in records])))
                     if records else np.empty((0, 3)))
    rows = []
    for record, probability in zip(records, probabilities):
        predicted = int(probability.argmax())
        rows.append({"subject_id": record.subject_id, "segment_id": record.segment_id, "label": record.label,
            **dict(zip(PROBABILITY_COLUMNS, map(float, probability))), "predicted_class": predicted,
            "predicted_label": CLASS_NAMES[predicted], "confidence": float(probability.max()), "prediction_type": "event"})
    events = pd.DataFrame(rows, columns=event_columns)
    patients = []
    for sid, group in events.groupby("subject_id", sort=True):
        probability = group[list(PROBABILITY_COLUMNS)].mean().to_numpy(float)
        predicted = int(probability.argmax())
        patients.append({"subject_id": sid, "label": int(group.label.iloc[0]),
            **dict(zip(PROBABILITY_COLUMNS, map(float, probability))), "predicted_class": predicted,
            "predicted_label": CLASS_NAMES[predicted], "confidence": float(probability.max()), "n_segments": len(group)})
    return {"events": events, "segments": events.copy(), "patients": pd.DataFrame(patients, columns=patient_columns),
            "evaluation_unit": "event", "assessment": events.copy()}


def _checkpoint_records(result):
    values, labels, subjects, segments = result.train_values, result.train_labels, result.train_subject_ids, result.train_segment_ids
    if (values.ndim != 2 or labels.ndim != 1 or subjects.ndim != 1 or segments.ndim != 1
            or len({len(values), len(labels), len(subjects), len(segments)}) != 1
            or values.dtype.kind != "f" or labels.dtype.kind not in "iu"
            or subjects.dtype.kind != "U" or segments.dtype.kind != "U"):
        raise ValueError("Invalid KNN checkpoint array shapes or dtypes.")
    records = _records([SimpleNamespace(subject_id=sid, segment_id=eid, label=label, values=curve)
                        for sid, eid, label, curve in zip(subjects, segments, labels, values)], result.config.length)
    train_ids, _ = _audit_split(result.split, result.cohort_manifest)
    expected = [row for row in result.cohort_manifest if row[0] in train_ids]
    if _manifest(records) != expected or list(subjects) != [row[0] for row in expected] or list(segments) != [row[1] for row in expected]:
        raise ValueError("Saved training arrays do not match the patient split and CGM hashes.")
    fitted = CGMScaler.fit(records, result.config.length)
    if (result.scaler.length != result.config.length
            or set(result.scaler.fitted_subject_ids) != train_ids
            or not np.isfinite(result.scaler.baseline_curve).all()
            or not np.isclose(result.scaler.mean, fitted.mean, rtol=1e-12, atol=1e-12)
            or not np.isclose(result.scaler.std, fitted.std, rtol=1e-12, atol=1e-12)
            or not np.array_equal(result.scaler.baseline_curve, fitted.baseline_curve)):
        raise ValueError("KNN normalization does not match the saved public training events.")
    segment_config = result.split.get("segment_config")
    if segment_config is not None and segment_config.get("length") != result.config.length:
        raise ValueError("Saved segment configuration and KNN lengths disagree.")
    return records


def save_knn(result: KNNResult, path) -> str:
    """Write a single NPZ containing numeric/string arrays and an embedded JSON manifest."""
    result.config.validate()
    _checkpoint_records(result)
    manifest = {"schema": "cgm-only-event-knn-v1", "class_names": list(CLASS_NAMES),
                "config": asdict(result.config), "scaler": result.scaler.to_dict(),
                "split": result.split, "cohort_manifest": result.cohort_manifest}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, manifest=np.asarray(json.dumps(manifest, ensure_ascii=False, allow_nan=False)),
            train_values=result.train_values, train_labels=result.train_labels,
            train_subject_ids=result.train_subject_ids, train_segment_ids=result.train_segment_ids)
    result.checkpoint_path = str(path)
    return str(path)


def load_knn(path) -> KNNResult:
    """Read only the specified NPZ; rebuild the estimator from saved training vectors."""
    fields = {"manifest", "train_values", "train_labels", "train_subject_ids", "train_segment_ids"}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != fields or len(archive.files) != len(fields):
            raise ValueError("Unsupported KNN checkpoint arrays.")
        text = archive["manifest"]
        if text.shape != () or text.dtype.kind != "U":
            raise ValueError("KNN manifest must be a scalar JSON string.")
        def invalid_constant(value):
            raise ValueError(f"Non-finite JSON value: {value}")
        payload = json.loads(text.item(), parse_constant=invalid_constant)
        if (not isinstance(payload, dict) or payload.get("schema") != "cgm-only-event-knn-v1"
                or payload.get("class_names") != list(CLASS_NAMES)):
            raise ValueError("Unsupported KNN checkpoint schema or class order.")
        config = KNNConfig(**payload["config"])
        config.validate()
        scaler = CGMScaler.from_dict(payload["scaler"])
        result = KNNResult(None, scaler, config, payload["split"], archive["train_values"].copy(),
            archive["train_labels"].copy(), archive["train_subject_ids"].copy(), archive["train_segment_ids"].copy(),
            payload["cohort_manifest"], str(path))
    _checkpoint_records(result)
    result.model = _fit_model(config, scaler, result.train_values, result.train_labels)
    return result
