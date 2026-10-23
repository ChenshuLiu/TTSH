"""Portable inference for the five saved CGM-only HbA1c classifiers.

Input events contain ``subject_id``, ``event_id`` and exactly 64 chronological
CGM values in mmol/L. Segment selection and time interpolation must already be
complete. The checkpoint's public-training normalization is applied internally.

Python API::

    loaded = load_model("transformer", seed=42)
    tables = predict(loaded, events)
    tables["events"].to_csv("event_predictions.csv", index=False)

The synthetic example is for checking the interface, not evaluating accuracy.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from numbers import Real
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from cgm_bag.knn import load_knn, predict_knn
from cgm_bag.training import CLASS_NAMES, PROBABILITY_COLUMNS, load_checkpoint, predict_segments


ARCHITECTURES = ("knn", "1d_cnn", "tcn", "bigru", "transformer")
SEEDS = (42, 43, 44, 45, 46)
SEQUENCE_LENGTH = 64
DEFAULT_BUNDLE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LoadedModel:
    """Original loaded checkpoint plus its model/seed identity."""

    architecture: str
    seed: int
    checkpoint_path: Path
    result: Any


def load_model(
    architecture: str,
    seed: int = 42,
    bundle_dir: str | Path | None = None,
) -> LoadedModel:
    """Load one saved classifier on CPU, retaining the original checkpoint audit.

    ``bundle_dir`` is the folder containing ``checkpoints/``. Its default is
    relative to this file, so callers can run from any working directory.
    KNN reconstructs its search index from saved public training vectors; this
    does not tune a model, recompute its normalization, or use the input events.
    """
    if not isinstance(architecture, str) or architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}.")
    bundle = DEFAULT_BUNDLE_DIR if bundle_dir is None else Path(bundle_dir).expanduser().resolve()
    filename = "model.npz" if architecture == "knn" else "model.pt"
    checkpoint = bundle / "checkpoints" / architecture / f"seed_{seed}" / filename
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    result = load_knn(checkpoint) if architecture == "knn" else load_checkpoint(checkpoint, device="cpu")
    config = result.config
    if config.architecture != architecture or config.seed != seed:
        raise ValueError("Checkpoint architecture/seed does not match the selected model.")
    if (
        config.length != SEQUENCE_LENGTH
        or config.sample_unit != "event"
        or not config.use_cgm
        or getattr(config, "k", 1) != 1
        or getattr(config, "metadata_dim", 0) != 0
        or getattr(config, "event_feature_dim", 0) != 0
        or getattr(config, "metadata_features", [])
        or getattr(config, "event_features", [])
    ):
        raise ValueError("This inference entry point requires a CGM-only, 64-point event checkpoint (K=1).")
    return LoadedModel(architecture, seed, checkpoint, result)


def _identifier(value: Any, field: str, row: int) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"Event {row}: {field} must be a nonempty string without surrounding whitespace.")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"Event {row}: {field} cannot contain control characters.")
    return value


def validate_events(events: Iterable[Mapping[str, Any]]) -> list[SimpleNamespace]:
    """Validate the public API input without modifying/normalizing its values.

    IDs must be strings. Event IDs are unique within each patient; the same
    event ID may be used by different patients. Only the three documented
    fields are accepted, preventing accidental use of metadata or target labels.
    """
    if isinstance(events, (str, bytes, Mapping)) or not isinstance(events, Iterable):
        raise ValueError("events must be a nonempty iterable of event objects.")
    records: list[SimpleNamespace] = []
    seen: set[tuple[str, str]] = set()
    expected_fields = {"subject_id", "event_id", "values"}
    for row, event in enumerate(events, start=1):
        if not isinstance(event, Mapping) or set(event) != expected_fields:
            raise ValueError(f"Event {row}: provide exactly subject_id, event_id and values.")
        subject_id = _identifier(event["subject_id"], "subject_id", row)
        event_id = _identifier(event["event_id"], "event_id", row)
        key = (subject_id, event_id)
        if key in seen:
            raise ValueError(f"Event {row}: duplicate (subject_id, event_id).")
        seen.add(key)
        raw_values = event["values"]
        if not isinstance(raw_values, (list, tuple, np.ndarray)):
            raise ValueError(f"Event {row}: values must be a sequence of {SEQUENCE_LENGTH} numbers.")
        values = np.asarray(raw_values, dtype=object)
        if values.shape != (SEQUENCE_LENGTH,):
            raise ValueError(f"Event {row}: values must have shape ({SEQUENCE_LENGTH},), received {values.shape}.")
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in values):
            raise ValueError(f"Event {row}: values must contain numbers; nulls, booleans and numeric strings are not accepted.")
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                numeric = np.asarray(values, dtype=np.float32)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(f"Event {row}: CGM values must be finite float32 numbers.") from error
        if not np.isfinite(numeric).all():
            raise ValueError(f"Event {row}: NaN, infinity and float32 overflow are not accepted.")
        records.append(SimpleNamespace(subject_id=subject_id, segment_id=event_id,
                                       values=numeric.copy(), label=-1))
    if not records:
        raise ValueError("Provide at least one event.")
    return records


def predict(loaded: LoadedModel, events: Iterable[Mapping[str, Any]]) -> dict[str, pd.DataFrame]:
    """Return event and patient probabilities for all three original classes.

    Each event is predicted independently. Each patient's probability vector is
    the arithmetic mean of all their event probability vectors. No clinical
    features, hand-crafted features, labels, or input-dependent fitting are used.
    """
    if not isinstance(loaded, LoadedModel):
        raise TypeError("loaded must be the object returned by load_model().")
    records = validate_events(events)
    if loaded.architecture == "knn":
        raw = predict_knn(loaded.result, records)
        patients, event_table = raw["patients"], raw["events"]
    else:
        patients, event_table, _ = predict_segments(
            loaded.result.model, loaded.result.scaler, records, k=1, sample_unit="event"
        )
    event_table = event_table.rename(columns={"segment_id": "event_id"}).copy()
    event_table["confidence"] = event_table[list(PROBABILITY_COLUMNS)].max(axis=1)
    event_columns = ["subject_id", "event_id", *PROBABILITY_COLUMNS,
                     "predicted_class", "predicted_label", "confidence"]
    patient_columns = ["subject_id", *PROBABILITY_COLUMNS, "predicted_class",
                       "predicted_label", "confidence", "n_events"]
    event_table = event_table[event_columns].sort_values(["subject_id", "event_id"]).reset_index(drop=True)
    patients = patients.rename(columns={"n_segments": "n_events"})[patient_columns]
    patients = patients.sort_values("subject_id").reset_index(drop=True)
    return {"events": event_table, "patients": patients}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON value is not allowed: {value}")


def read_events(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON list of presegmented, resampled events."""
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        events = json.load(stream, parse_constant=_reject_json_constant)
    if not isinstance(events, list):
        raise ValueError("The input JSON must be a list of event objects.")
    validate_events(events)
    return events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=ARCHITECTURES, default="transformer")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--input", type=Path, required=True,
                        help="JSON list with subject_id, event_id, values (64 chronological mmol/L points).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR,
                        help="Bundle folder containing checkpoints/; defaults to this script's folder.")
    args = parser.parse_args(argv)
    try:
        events = read_events(args.input)
        loaded = load_model(args.model, args.seed, args.bundle_dir)
        tables = predict(loaded, events)
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(output_dir / f"{name}_probabilities.csv", index=False)
        summary = {
            "architecture": loaded.architecture,
            "seed": loaded.seed,
            "class_order": list(CLASS_NAMES),
            "input_unit": "mmol/L",
            "points_per_event": SEQUENCE_LENGTH,
            "n_events": len(tables["events"]),
            "n_patients": len(tables["patients"]),
            "patient_aggregation": "arithmetic mean of the patient's event probabilities",
            "normalization": "saved public-training normalization; no fit on inference events",
            "class_filter": "none: Healthy, Prediabetes and T2D are all retained",
        }
        (output_dir / "prediction_run.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        parser.exit(2, f"Prediction failed: {error}\n")
    print(f"Saved {len(tables['events'])} event and {len(tables['patients'])} patient predictions to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
