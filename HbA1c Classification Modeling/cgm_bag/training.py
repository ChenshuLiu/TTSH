"""Patient-disjoint training, training-only normalization, and audited inference."""
from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn

from .models import PatientBagModel, build_model
from .metadata import MetadataPreprocessor
from .event_features import FEATURE_NAMES, EventFeaturePreprocessor, event_features_frame

CLASS_NAMES = ("Healthy", "Prediabetes", "T2D")
PROBABILITY_COLUMNS = tuple(f"p_{name}" for name in CLASS_NAMES)


@dataclass
class TrainConfig:
    architecture: str = "1d_cnn"
    k: int = 5
    length: int = 64
    pooling: str = "attention"
    val_fraction: float = 0.25
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 7
    hidden_dim: int = 32
    dropout: float = 0.1
    seed: int = 42
    class_weights: bool = True
    device: str = "cpu"
    sample_unit: str = "patient_bag"
    metadata_features: list[str] = field(default_factory=list)
    metadata_dim: int = 0
    event_features: list[str] = field(default_factory=list)
    event_feature_dim: int = 0
    use_cgm: bool = True

    def validate(self):
        for name in ("k", "length", "epochs", "batch_size", "patience", "hidden_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not 0 < self.val_fraction < 1:
            raise ValueError("val_fraction must lie strictly between 0 and 1.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive; weight_decay must be non-negative.")
        if self.length < 4 or self.hidden_dim < 4 or self.hidden_dim % 2:
            raise ValueError("length must be >= 4; hidden_dim must be even and >= 4.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0,1).")
        if not isinstance(self.use_cgm, bool):
            raise ValueError("use_cgm must be a boolean.")
        if self.use_cgm and self.architecture not in {"1d_cnn", "tcn", "bigru", "transformer"}:
            raise ValueError("CGM models require 1d_cnn, tcn, bigru or transformer.")
        if not self.use_cgm and self.architecture != "mlp":
            raise ValueError("Models without CGM require architecture='mlp'.")
        if self.pooling not in {"mean", "attention"}:
            raise ValueError("Choose mean or attention pooling.")
        if self.sample_unit not in {"patient_bag", "event"}:
            raise ValueError("sample_unit must be 'patient_bag' or 'event'.")
        if self.sample_unit == "event" and self.k != 1:
            raise ValueError("Event training requires k=1: each input is one event/segment.")
        if not isinstance(self.metadata_features, list) or len(set(self.metadata_features)) != len(self.metadata_features):
            raise ValueError("metadata_features must be a list of unique feature names.")
        if not set(self.metadata_features) <= {"age", "bmi", "height_cm", "weight_kg", "sbp", "dbp"}:
            raise ValueError("Only baseline age, BMI, height, weight and blood pressure can be used as metadata features.")
        if not isinstance(self.metadata_dim, int) or self.metadata_dim < 0:
            raise ValueError("metadata_dim must be a non-negative integer.")
        if self.metadata_dim not in (0, 2 * len(self.metadata_features)):
            raise ValueError("metadata_dim must match the selected features and their missing-value indicators.")
        if (not isinstance(self.event_features, list) or any(not isinstance(name, str) for name in self.event_features)
                or len(set(self.event_features)) != len(self.event_features)
                or not set(self.event_features) <= set(FEATURE_NAMES)):
            raise ValueError("event_features must be a list of unique unified CGM feature names.")
        if (isinstance(self.event_feature_dim, bool) or not isinstance(self.event_feature_dim, int)
                or self.event_feature_dim not in (0, 2 * len(self.event_features))):
            raise ValueError("event_feature_dim must match selected event features and missing-value indicators.")
        if self.event_features and (self.sample_unit != "event" or self.k != 1):
            raise ValueError("Unified CGM event features currently require event mode with K=1.")
        if not self.use_cgm:
            if not (self.metadata_features or self.event_features):
                raise ValueError("Models without CGM require at least one metadata or event feature.")
            if self.sample_unit != "event" or self.k != 1:
                raise ValueError("Tabular ablations without CGM require event mode with K=1.")


def _group_segments(segments: Iterable[Any], length: int | None = None) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    seen: set[tuple[str, str]] = set()
    for segment in segments:
        subject_id = str(segment.subject_id)
        segment_id = str(segment.segment_id)
        if not subject_id.strip() or not segment_id.strip():
            raise ValueError("Patient and segment IDs must be non-empty.")
        key = (subject_id, segment_id)
        if key in seen:
            raise ValueError(f"Duplicate segment ID for patient {subject_id}: {segment_id}")
        seen.add(key)
        values = np.asarray(segment.values, dtype=np.float32)
        if values.ndim != 1 or values.size < 4 or not np.isfinite(values).all():
            raise ValueError(f"Segment {key} must contain >=4 finite CGM values.")
        if length is not None and values.size != length:
            raise ValueError(f"Segment {key} has length {values.size}; expected {length}. Re-extract using the training length.")
        if int(segment.label) not in (-1, 0, 1, 2):
            raise ValueError(f"Invalid label for {subject_id}; use -1, 0, 1 or 2.")
        grouped.setdefault(subject_id, []).append(segment)
    for subject_id, items in grouped.items():
        labels = {int(s.label) for s in items}
        if len(labels) != 1:
            raise ValueError(f"Patient {subject_id} has inconsistent labels across segments: {sorted(labels)}")
        items.sort(key=lambda s: str(s.segment_id))
    return dict(sorted(grouped.items()))


def patient_stratified_split(segments: Iterable[Any], val_fraction=0.25, seed=42) -> dict:
    """Keep every patient's segments together and require every class in both sets.

    At least two labeled patients per class are needed. With very small cohorts,
    the actual fraction may differ from the requested fraction to retain classes.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0,1).")
    grouped = _group_segments(segments)
    by_class = {label: [sid for sid, items in grouped.items() if int(items[0].label) == label] for label in range(3)}
    counts = {CLASS_NAMES[label]: len(ids) for label, ids in by_class.items()}
    if any(value < 2 for value in counts.values()):
        raise ValueError(
            "Patient-stratified three-class training needs at least 2 labeled patients in EACH class "
            f"(one train and one validation). Available patient counts: {counts}. "
            "Add patients; do not split a patient's segments across sets."
        )
    rng = np.random.default_rng(seed)
    train_ids, val_ids = [], []
    for label, ids in by_class.items():
        shuffled = rng.permutation(ids).tolist()
        n_val = min(len(ids) - 1, max(1, int(round(len(ids) * val_fraction))))
        val_ids.extend(shuffled[:n_val])
        train_ids.extend(shuffled[n_val:])
    train_ids, val_ids = sorted(train_ids), sorted(val_ids)
    assert not set(train_ids).intersection(val_ids)
    assignment = [(sid, str(s.segment_id), int(s.label), hashlib.sha256(np.asarray(s.values, dtype=np.float32).tobytes()).hexdigest())
                  for sid, items in grouped.items() for s in items]
    return {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "excluded_unlabeled_ids": [sid for sid, items in grouped.items() if int(items[0].label) == -1],
        "class_counts": counts,
        "train_class_counts": {CLASS_NAMES[i]: sum(int(grouped[sid][0].label) == i for sid in train_ids) for i in range(3)},
        "val_class_counts": {CLASS_NAMES[i]: sum(int(grouped[sid][0].label) == i for sid in val_ids) for i in range(3)},
        "train_segment_count": sum(len(grouped[sid]) for sid in train_ids),
        "val_segment_count": sum(len(grouped[sid]) for sid in val_ids),
        "patient_overlap": [],
        "seed": int(seed), "requested_val_fraction": float(val_fraction),
        "effective_val_fraction": len(val_ids) / (len(train_ids) + len(val_ids)),
        "data_fingerprint_sha256": hashlib.sha256(json.dumps(assignment, separators=(",", ":")).encode()).hexdigest(),
        "split_unit": "subject_id", "normalization_fit": "training patients only",
        "training_bags": "One uniformly sampled bag per patient per epoch; no replacement within bag; masked padding",
        "inference_bags": "All segments sorted by segment_id; disjoint K chunks; probabilities weighted by valid segment count",
    }


@dataclass
class CGMScaler:
    """Patient-balanced global z-score, never fitted on validation/private values."""
    mean: float
    std: float
    length: int
    fitted_subject_ids: list[str]
    baseline_curve: np.ndarray

    @classmethod
    def fit(cls, segments: Iterable[Any], length: int):
        grouped = _group_segments(segments, length)
        if not grouped:
            raise ValueError("Cannot fit normalization without training patients.")
        curves = [np.stack([s.values for s in items]).astype(np.float64) for items in grouped.values()]
        mean = float(np.mean([x.mean() for x in curves]))
        variance = float(np.mean([np.mean((x - mean) ** 2) for x in curves]))
        baseline = np.mean([x.mean(axis=0) for x in curves], axis=0).astype(np.float32)
        return cls(mean, max(float(np.sqrt(variance)), 1e-6), length, list(grouped), baseline)

    def transform_values(self, values) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim < 1 or array.shape[-1] != self.length:
            raise ValueError(f"Expected raw CGM sequence length {self.length}; re-extract segments using the saved configuration.")
        if not np.isfinite(array).all():
            raise ValueError("CGM sequences must be finite before normalization.")
        return ((array - self.mean) / self.std).astype(np.float32)

    transform = transform_values

    def to_dict(self):
        return {"mean": self.mean, "std": self.std, "length": self.length,
                "fitted_subject_ids": self.fitted_subject_ids, "baseline_curve": self.baseline_curve.tolist(),
                "method": "patient-balanced global training-only z-score"}

    @classmethod
    def from_dict(cls, data):
        scaler = cls(float(data["mean"]), float(data["std"]), int(data["length"]),
                     list(data["fitted_subject_ids"]), np.asarray(data["baseline_curve"], dtype=np.float32))
        if scaler.std <= 0 or not np.isfinite([scaler.mean, scaler.std]).all() or scaler.baseline_curve.shape != (scaler.length,):
            raise ValueError("Invalid checkpoint normalization statistics.")
        return scaler


def _pack_arrays(arrays: list[np.ndarray], k: int, length: int):
    if k < 1 or not arrays or any(len(a) < 1 or len(a) > k for a in arrays):
        raise ValueError("Each bag must contain between 1 and K segments.")
    values = np.zeros((len(arrays), k, length), dtype=np.float32)
    mask = np.zeros((len(arrays), k), dtype=bool)
    for index, array in enumerate(arrays):
        values[index, :len(array)] = array
        mask[index, :len(array)] = True
    return values, mask


def sample_patient_bag(values: np.ndarray, k: int, rng: np.random.Generator):
    """One patient's uniform sample without replacement, with explicit padding."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not len(values) or k < 1:
        raise ValueError("Need a non-empty [segments,length] array and positive K.")
    chosen = rng.choice(len(values), size=min(k, len(values)), replace=False)
    packed, mask = _pack_arrays([values[chosen]], k, values.shape[1])
    return packed[0], mask[0], chosen


def _forward_model(model: PatientBagModel, values, mask, metadata=None, event_features=None):
    """Keep established CGM/clinical call shapes for model hooks and old callers."""
    if model.event_feature_dim:
        return model(values, mask, metadata, event_features)
    return model(values, mask, metadata) if model.metadata_dim else model(values, mask)


def predict_bag_probabilities(model: PatientBagModel, normalized_bags, mask, metadata=None, event_features=None) -> np.ndarray:
    """Shared deterministic probability path for pre-normalized padded bags."""
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        values = torch.as_tensor(normalized_bags, dtype=torch.float32, device=device)
        valid = torch.as_tensor(mask, dtype=torch.bool, device=device)
        clinical = None if metadata is None else torch.as_tensor(metadata, dtype=torch.float32, device=device)
        features = None if event_features is None else torch.as_tensor(event_features, dtype=torch.float32, device=device)
        output = _forward_model(model, values, valid, clinical, features)
        return torch.softmax(output["patient_logits"], dim=-1).cpu().numpy()


def _predict_raw_details(model: PatientBagModel, scaler: CGMScaler, raw_segments, k: int, metadata=None, event_features=None):
    values = np.asarray(raw_segments, dtype=np.float32)
    if values.ndim != 2 or not len(values) or k < 1:
        raise ValueError("Need a non-empty [segments,length] array and positive K.")
    normalized = scaler.transform_values(values) if model.use_cgm else np.zeros_like(values, dtype=np.float32)
    if model.event_feature_dim:
        if k != 1:
            raise ValueError("Unified CGM event features currently require event mode with K=1.")
        event_features = np.asarray(event_features, dtype=np.float32) if event_features is not None else None
        if (event_features is None or event_features.shape != (len(values), model.event_feature_dim)
                or not np.isfinite(event_features).all()):
            raise ValueError(f"This model requires finite encoded event features [segments, {model.event_feature_dim}].")
    if model.metadata_dim:
        metadata = np.asarray(metadata, dtype=np.float32) if metadata is not None else None
        if metadata is None or metadata.shape != (model.metadata_dim,) or not np.isfinite(metadata).all():
            raise ValueError(f"This model requires a finite encoded metadata vector of length {model.metadata_dim}.")
    chunks = [normalized[start:start + k] for start in range(0, len(values), k)]
    counts = np.asarray([len(chunk) for chunk in chunks])
    model.eval()
    device = next(model.parameters()).device
    bag_probabilities, segment_probabilities, attentions = [], [], []
    # Bounded batches also keep large public validation cohorts inexpensive.
    with torch.inference_mode():
        for start in range(0, len(chunks), 64):
            batch_chunks = chunks[start:start + 64]
            packed, mask = _pack_arrays(batch_chunks, k, scaler.length)
            clinical = (torch.as_tensor(np.repeat(metadata[None, :], len(batch_chunks), axis=0), device=device)
                        if model.metadata_dim else None)
            tensor_values, tensor_mask = torch.as_tensor(packed, device=device), torch.as_tensor(mask, device=device)
            features = (torch.as_tensor(event_features[start:start + len(batch_chunks), None, :], device=device)
                        if model.event_feature_dim else None)
            output = _forward_model(model, tensor_values, tensor_mask, clinical, features)
            bag_probabilities.append(torch.softmax(output["patient_logits"], dim=-1).cpu().numpy())
            segment_probs = torch.softmax(output["segment_logits"], dim=-1).cpu().numpy()
            segment_probabilities.extend(segment_probs[i, :len(chunk)] for i, chunk in enumerate(batch_chunks))
            attentions.extend(output["attention"].cpu().numpy()[i, :len(chunk)] for i, chunk in enumerate(batch_chunks))
    bag_probabilities = np.concatenate(bag_probabilities)
    patient_probability = np.average(bag_probabilities, axis=0, weights=counts).astype(np.float32)
    return patient_probability, np.concatenate(segment_probabilities), bag_probabilities, counts, attentions


def predict_raw_patient(model: PatientBagModel, scaler: CGMScaler, raw_segments, k: int, metadata=None,
                        event_features=None) -> np.ndarray:
    """Same patient estimator as external testing, for SHAP perturbation callbacks.

    Preserve segment order and membership: predict_segments sorts segment_id;
    supply SHAP's raw input rows in that same order. It partitions all rows into
    disjoint K bags and averages bag probabilities using valid counts as weights.
    """
    return _predict_raw_details(model, scaler, raw_segments, k, metadata=metadata, event_features=event_features)[0]


predict_patient_values = predict_raw_patient


def predict_segments(model: PatientBagModel, scaler: CGMScaler, segments: Iterable[Any], k: int,
                     sample_unit: str = "patient_bag", metadata_preprocessor: MetadataPreprocessor | None = None,
                     clinical_metadata: pd.DataFrame | None = None,
                     event_feature_preprocessor: EventFeaturePreprocessor | None = None):
    """Return patient, event/weak-instance, and bag-provenance probability tables.

    Event models use K=1, so the event probability is exactly the same estimator
    as a one-row predict_raw_patient call. Patient probabilities remain the mean
    of that patient's event probabilities for an additional aggregate report.
    """
    if sample_unit not in {"patient_bag", "event"}:
        raise ValueError("sample_unit must be 'patient_bag' or 'event'.")
    if sample_unit == "event" and k != 1:
        raise ValueError("Event prediction requires k=1.")
    if not model.use_cgm and (sample_unit != "event" or k != 1):
        raise ValueError("Tabular ablations without CGM require event mode with K=1.")
    if model.event_feature_dim:
        if sample_unit != "event" or k != 1:
            raise ValueError("Unified CGM event features currently require event mode with K=1.")
        if event_feature_preprocessor is None or event_feature_preprocessor.dim != model.event_feature_dim:
            raise ValueError("Event-feature prediction requires the model's fitted public event-feature preprocessor.")
    grouped = _group_segments(segments, scaler.length)
    encoded_metadata = None
    if model.metadata_dim:
        if metadata_preprocessor is None or clinical_metadata is None:
            raise ValueError("CGM + metadata prediction requires the fitted preprocessor and a clinical metadata table.")
        if metadata_preprocessor.dim != model.metadata_dim:
            raise ValueError("Metadata preprocessing dimensions disagree with the model.")
        encoded_metadata = dict(zip(grouped, metadata_preprocessor.transform(clinical_metadata, list(grouped))))
    patient_rows, segment_rows, bag_rows = [], [], []
    for subject_id, items in grouped.items():
        raw = np.stack([s.values for s in items])
        patient_prob, segment_prob, bag_prob, counts, attentions = _predict_raw_details(
            model, scaler, raw, k, metadata=encoded_metadata[subject_id] if encoded_metadata is not None else None,
            event_features=event_feature_preprocessor.transform(items) if model.event_feature_dim else None)
        label = int(items[0].label)
        patient_rows.append({
            "subject_id": subject_id, "label": label,
            **dict(zip(PROBABILITY_COLUMNS, map(float, patient_prob))),
            "predicted_class": int(patient_prob.argmax()), "predicted_label": CLASS_NAMES[int(patient_prob.argmax())],
            "confidence": float(patient_prob.max()), "n_segments": len(items), "n_bags": len(counts),
        })
        for bag_index, (bag_p, count) in enumerate(zip(bag_prob, counts)):
            start = bag_index * k
            bag_id = f"{subject_id}::bag_{bag_index:04d}"
            bag_rows.append({
                "subject_id": subject_id, "bag_id": bag_id, "label": label,
                "segment_ids": json.dumps([str(s.segment_id) for s in items[start:start + count]], ensure_ascii=False),
                "valid_count": int(count), "padded_count": int(k - count), "aggregation_weight": float(count / len(items)),
                **dict(zip(PROBABILITY_COLUMNS, map(float, bag_p))),
            })
            for within_bag, item in enumerate(items[start:start + count]):
                p = segment_prob[start + within_bag]
                info = getattr(item, "info", {}) or {}
                segment_rows.append({
                    "subject_id": subject_id, "segment_id": str(item.segment_id), "bag_id": bag_id,
                    "event_uid": str(info.get("event_uid") or item.segment_id),
                    "event_name": info.get("event_name", ""),
                    "label": label, **dict(zip(PROBABILITY_COLUMNS, map(float, p))),
                    "predicted_class": int(p.argmax()), "predicted_label": CLASS_NAMES[int(p.argmax())],
                    "attention_weight_within_bag": float(attentions[bag_index][within_bag]),
                    "quality": float(getattr(item, "quality", float("nan"))),
                    "prediction_type": "event" if sample_unit == "event" else "weak_single_instance",
                })
    patient_columns = ["subject_id", "label", *PROBABILITY_COLUMNS, "predicted_class", "predicted_label", "confidence", "n_segments", "n_bags"]
    segment_columns = ["subject_id", "segment_id", "event_uid", "event_name", "bag_id", "label", *PROBABILITY_COLUMNS, "predicted_class", "predicted_label", "attention_weight_within_bag", "quality", "prediction_type"]
    bag_columns = ["subject_id", "bag_id", "label", "segment_ids", "valid_count", "padded_count", "aggregation_weight", *PROBABILITY_COLUMNS]
    return pd.DataFrame(patient_rows, columns=patient_columns), pd.DataFrame(segment_rows, columns=segment_columns), pd.DataFrame(bag_rows, columns=bag_columns)


@dataclass
class TrainingResult:
    model: PatientBagModel
    scaler: CGMScaler
    config: TrainConfig
    split: dict
    history: pd.DataFrame
    best_epoch: int
    checkpoint_path: str | None = None
    metadata_preprocessor: MetadataPreprocessor | None = None
    event_feature_preprocessor: EventFeaturePreprocessor | None = None

    @property
    def train_ids(self):
        return self.split["train_ids"]

    @property
    def val_ids(self):
        return self.split["val_ids"]


def event_feature_preprocessor_sha256(processor: EventFeaturePreprocessor) -> str:
    """Bind the saved feature recipe, fit membership, and statistics together."""
    return hashlib.sha256(json.dumps(processor.to_dict(), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def audit_event_feature_preprocessor(result: TrainingResult) -> None:
    processor, config, split = result.event_feature_preprocessor, result.config, result.split
    config.validate()
    if (result.model.use_cgm != config.use_cgm or result.model.architecture != config.architecture
            or result.model.metadata_dim != config.metadata_dim):
        raise ValueError("Checkpoint model input branches and configuration disagree.")
    if result.model.event_feature_dim != config.event_feature_dim:
        raise ValueError("Checkpoint event-feature model dimensions and configuration disagree.")
    if not config.event_feature_dim:
        if processor is not None or config.event_features:
            raise ValueError("Checkpoint event-feature configuration and preprocessing disagree.")
        return
    if (processor is None or processor.dim != config.event_feature_dim
            or processor.features != config.event_features
            or set(processor.fitted_subject_ids) != set(split["train_ids"])
            or processor.fitted_segment_ids != split.get("event_feature_fitted_segment_ids")
            or event_feature_preprocessor_sha256(processor) != split.get("event_feature_preprocessing_sha256")):
        raise ValueError("Checkpoint failed public training-only event-feature preprocessing audit.")
    try:
        keys = [json.loads(key) for key in processor.fitted_segment_ids]
        valid = (len(keys) == len({tuple(key) for key in keys})
                 and all(isinstance(key, list) and len(key) == 2 and all(isinstance(value, str) and value for value in key)
                         for key in keys)
                 and {key[0] for key in keys} == set(split["train_ids"]))
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Checkpoint failed event-feature training-segment membership audit.")


def validate_segment_config(result: TrainingResult, config) -> None:
    """Enforce the public extraction recipe when it is present in a checkpoint.

    Legacy checkpoints did not save this binding, so they continue to accept a
    supplied recipe of matching length. Newly bound checkpoints require every
    extraction and quality-control setting to match exactly.
    """
    config.validate()
    if config.length != result.config.length:
        raise ValueError("Segment configuration length does not match the saved model.")
    saved = result.split.get("segment_config")
    if saved is not None and saved != asdict(config):
        raise ValueError("Segment configuration does not match the public checkpoint's saved extraction/QC recipe. "
                         "Use the preprocessing.json from the same training run; interval kind and gap limits must remain unchanged.")


def save_checkpoint(result: TrainingResult, path: str | Path) -> str:
    audit_event_feature_preprocessor(result)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "state_dict": {name: tensor.detach().cpu().clone() for name, tensor in result.model.state_dict().items()},
        "config": asdict(result.config), "scaler": result.scaler.to_dict(),
        "split": result.split, "best_epoch": result.best_epoch,
        "history": result.history.to_dict(orient="records"), "class_names": list(CLASS_NAMES),
        "metadata_preprocessor": result.metadata_preprocessor.to_dict() if result.metadata_preprocessor is not None else None,
        "event_feature_preprocessor": (result.event_feature_preprocessor.to_dict()
                                       if result.event_feature_preprocessor is not None else None),
    }
    torch.save(payload, path)
    return str(path)


def load_checkpoint(path: str | Path | Any, device="cpu") -> TrainingResult:
    """Load only a tensor/primitive dictionary; never arbitrary Python objects."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Unsupported checkpoint schema or class order.")
    config = TrainConfig(**{**payload["config"], "device": device})
    config.validate()
    model = build_model(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    scaler = CGMScaler.from_dict(payload["scaler"])
    if scaler.length != config.length:
        raise ValueError("Checkpoint model and normalization lengths disagree.")
    split = payload["split"]
    if set(split["train_ids"]) & set(split["val_ids"]) or set(scaler.fitted_subject_ids) != set(split["train_ids"]):
        raise ValueError("Checkpoint failed patient-level split/normalization audit.")
    metadata_preprocessor = (MetadataPreprocessor.from_dict(payload["metadata_preprocessor"])
                             if payload.get("metadata_preprocessor") is not None else None)
    if config.metadata_dim:
        if (metadata_preprocessor is None or metadata_preprocessor.dim != config.metadata_dim
                or metadata_preprocessor.features != config.metadata_features
                or set(metadata_preprocessor.fitted_subject_ids) != set(split["train_ids"])):
            raise ValueError("Checkpoint failed metadata training-patient preprocessing audit.")
    elif metadata_preprocessor is not None or config.metadata_features:
        raise ValueError("Checkpoint metadata configuration and preprocessing disagree.")
    event_feature_preprocessor = (EventFeaturePreprocessor.from_dict(payload["event_feature_preprocessor"])
                                  if payload.get("event_feature_preprocessor") is not None else None)
    result = TrainingResult(model, scaler, config, split, pd.DataFrame(payload["history"]), int(payload["best_epoch"]),
                            str(path), metadata_preprocessor, event_feature_preprocessor)
    audit_event_feature_preprocessor(result)
    # Audit optional dataset provenance at the entry point, including standalone
    # reload/external-test paths that never build a multi-model comparison.
    from .cohorts import cohort_audit
    cohort_audit(result)
    return result


def train_model(segments: Iterable[Any], config: TrainConfig, output_dir: str | Path | None = None,
                progress_callback: Callable[[dict], None] | None = None,
                clinical_metadata: pd.DataFrame | None = None, *, segment_config=None) -> TrainingResult:
    config = copy.deepcopy(config)
    config.validate()
    segments = list(segments)
    if any(str(getattr(segment, "dataset", "")).lower() == "private" for segment in segments):
        raise ValueError("Private test segments cannot be used for training or validation.")
    grouped = _group_segments(segments, config.length)
    split = patient_stratified_split(segments, config.val_fraction, config.seed)
    if segment_config is not None:
        segment_config.validate()
        if segment_config.length != config.length:
            raise ValueError("Segment configuration length does not match the training model.")
        split["segment_config"] = asdict(segment_config)
    metadata_preprocessor = None
    train_metadata = None
    if config.metadata_features:
        if clinical_metadata is None:
            raise ValueError("CGM + metadata training requires a clinical metadata table.")
        metadata_preprocessor = MetadataPreprocessor.fit(clinical_metadata, config.metadata_features, split["train_ids"])
        config.metadata_dim = metadata_preprocessor.dim
        train_metadata = dict(zip(split["train_ids"], metadata_preprocessor.transform(clinical_metadata, split["train_ids"])))
    else:
        config.metadata_dim = 0
    split["metadata_features"] = config.metadata_features
    split["metadata_preprocessing_fit"] = "training patients only" if metadata_preprocessor is not None else "not used"
    if train_metadata is not None:
        encoded = np.stack(list(train_metadata.values()))
        feature_count = len(config.metadata_features)
        split["metadata_training_coverage"] = {
            feature: {"available_patients": int((encoded[:, feature_count + index] == 0).sum()),
                      "training_patients": len(split["train_ids"])}
            for index, feature in enumerate(config.metadata_features)
        }
        split["metadata_encoding"] = "Training medians, training z-scores, and one missing-value indicator per clinical feature"
    train_segments = [s for sid in split["train_ids"] for s in grouped[sid]]
    val_segments = [s for sid in split["val_ids"] for s in grouped[sid]]
    event_feature_preprocessor, train_features = None, None
    if config.event_features:
        event_feature_preprocessor = EventFeaturePreprocessor.fit(train_segments, config.event_features)
        config.event_feature_dim = event_feature_preprocessor.dim
        train_features = event_feature_preprocessor.transform(train_segments)
        split["event_feature_fitted_segment_ids"] = event_feature_preprocessor.fitted_segment_ids.copy()
        split["event_feature_preprocessing_sha256"] = event_feature_preprocessor_sha256(event_feature_preprocessor)
        split["event_feature_encoding"] = "Public training-event medians, z-scores, and one missing-value indicator per feature"
    else:
        config.event_feature_dim = 0
    split["event_features"] = config.event_features.copy()
    split["event_feature_preprocessing_fit"] = "public training events only" if train_features is not None else "not used"
    event_mode = config.sample_unit == "event"
    split.update({
        "sample_unit": config.sample_unit,
        "use_cgm": config.use_cgm,
        "cgm_waveform_used_as_model_input": config.use_cgm,
        "cgm_scaler_role": ("model input normalization" if config.use_cgm else
                            "public training-only compatibility artifact; not used by the tabular model"),
        "train_sample_count": len(train_segments) if event_mode else len(split["train_ids"]),
        "val_sample_count": len(val_segments) if event_mode else len(split["val_ids"]),
        "train_patient_count": len(split["train_ids"]),
        "val_patient_count": len(split["val_ids"]),
        "epoch_sample_counts": [],
        "validation_unit": "event" if event_mode else "patient",
        "class_weight_unit": "training events" if event_mode else "training patients",
    })
    if event_mode:
        split["training_bags"] = "Every training event once per epoch in shuffled order; K=1; subject-disjoint validation"
        split["inference_bags"] = "Each event separately with K=1; optional patient report averages all event probabilities"
    scaler = CGMScaler.fit(train_segments, config.length)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = build_model(config).to(config.device)
    train_values = {sid: (scaler.transform_values(np.stack([s.values for s in grouped[sid]])) if config.use_cgm
                         else np.zeros((len(grouped[sid]), config.length), dtype=np.float32)) for sid in split["train_ids"]}
    labels = {sid: int(grouped[sid][0].label) for sid in split["train_ids"]}
    class_targets = [int(segment.label) for segment in train_segments] if event_mode else list(labels.values())
    counts = np.bincount(class_targets, minlength=3)
    weights = len(class_targets) / (3.0 * counts) if config.class_weights else np.ones(3)
    split["class_weight_counts"] = dict(zip(CLASS_NAMES, map(int, counts)))
    split["class_weights"] = dict(zip(CLASS_NAMES, map(float, weights)))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=config.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state, history = None, []
    event_values = ((scaler.transform_values(np.stack([segment.values for segment in train_segments])) if config.use_cgm
                     else np.zeros((len(train_segments), config.length), dtype=np.float32)) if event_mode else None)
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(len(train_segments) if event_mode else split["train_ids"]).tolist()
        loss_sum, weight_sum = 0.0, 0.0
        samples_seen = 0
        for start in range(0, len(order), config.batch_size):
            batch = order[start:start + config.batch_size]
            if event_mode:
                x = torch.tensor(event_values[batch, None, :], device=config.device)
                mask = torch.ones((len(batch), 1), dtype=torch.bool, device=config.device)
                batch_labels = [int(train_segments[index].label) for index in batch]
                batch_subjects = [str(train_segments[index].subject_id) for index in batch]
            else:
                bags = [sample_patient_bag(train_values[sid], config.k, rng) for sid in batch]
                x = torch.tensor(np.stack([bag[0] for bag in bags]), device=config.device)
                mask = torch.tensor(np.stack([bag[1] for bag in bags]), device=config.device)
                batch_labels = [labels[sid] for sid in batch]
                batch_subjects = batch
            targets = torch.tensor(batch_labels, dtype=torch.long, device=config.device)
            optimizer.zero_grad(set_to_none=True)
            clinical = (torch.tensor(np.stack([train_metadata[sid] for sid in batch_subjects]), device=config.device)
                        if train_metadata is not None else None)
            features = (torch.tensor(train_features[batch, None, :], device=config.device)
                        if train_features is not None else None)
            output = _forward_model(model, x, mask, clinical, features)
            loss = criterion(output["patient_logits"], targets)
            if not torch.isfinite(loss):
                raise ValueError("Training loss became non-finite. Check data scale and learning rate.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_weight = float(weights[batch_labels].sum())
            loss_sum += loss.item() * batch_weight
            weight_sum += batch_weight
            samples_seen += len(batch)
        split["epoch_sample_counts"].append(samples_seen)
        val_patients, val_events, _ = predict_segments(model, scaler, val_segments, config.k, sample_unit=config.sample_unit,
                                                     metadata_preprocessor=metadata_preprocessor, clinical_metadata=clinical_metadata,
                                                     event_feature_preprocessor=event_feature_preprocessor)
        val_frame = val_events if event_mode else val_patients
        y = val_frame["label"].to_numpy(dtype=int)
        probabilities = val_frame[list(PROBABILITY_COLUMNS)].to_numpy()
        val_losses = -np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-8, 1.0))
        val_loss = float(np.average(val_losses, weights=weights[y]))
        row = {"epoch": epoch, "train_loss": loss_sum / weight_sum,
               "val_loss": val_loss, "val_accuracy": float(np.mean(probabilities.argmax(axis=1) == y)),
               "val_event_accuracy": float(np.mean(val_events.predicted_class == val_events.label)),
               "val_patient_accuracy": float(np.mean(val_patients.predicted_class == val_patients.label)),
               "train_samples_seen": samples_seen}
        history.append(row)
        if val_loss < best_loss - 1e-7:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if progress_callback:
            progress_callback({**row, "best_epoch": best_epoch, "max_epochs": config.epochs})
        if stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    result = TrainingResult(model, scaler, copy.deepcopy(config), split, pd.DataFrame(history), best_epoch,
                            metadata_preprocessor=metadata_preprocessor, event_feature_preprocessor=event_feature_preprocessor)
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        result.checkpoint_path = save_checkpoint(result, output / "model.pt")
        result.history.to_csv(output / "training_history.csv", index=False)
        (output / "split_audit.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "train_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
        (output / "normalizer.json").write_text(json.dumps(scaler.to_dict(), indent=2), encoding="utf-8")
        if metadata_preprocessor is not None:
            (output / "metadata_preprocessing.json").write_text(
                json.dumps(metadata_preprocessor.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        if event_feature_preprocessor is not None:
            (output / "event_feature_preprocessing.json").write_text(
                json.dumps(event_feature_preprocessor.to_dict(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            ordered = train_segments + val_segments
            raw_features = event_features_frame(ordered, config.event_features)
            raw_features["public_split"] = ["train"] * len(train_segments) + ["validation"] * len(val_segments)
            raw_features.to_csv(output / "event_feature_inputs.csv", index=False)
            encoded_features = pd.DataFrame(event_feature_preprocessor.transform(ordered),
                                            columns=event_feature_preprocessor.feature_names)
            encoded_features.insert(0, "segment_id", [str(item.segment_id) for item in ordered])
            encoded_features.insert(0, "subject_id", [str(item.subject_id) for item in ordered])
            encoded_features["public_split"] = raw_features["public_split"].to_numpy()
            encoded_features.to_csv(output / "event_feature_encoded_inputs.csv", index=False)
    return result
