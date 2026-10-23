"""Verify all shipped checkpoints using synthetic CGM, without training.

Run ``python verify_bundle.py --report verification_report.json`` from any
working directory. The report contains counts and validation status only;
no patient identifiers, CGM waveforms, or predictions are written to it.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from types import SimpleNamespace

import numpy as np
import torch

from cgm_bag.knn import load_knn, predict_knn
from cgm_bag.training import CLASS_NAMES, PROBABILITY_COLUMNS, load_checkpoint, predict_segments


ARCHITECTURES = ("knn", "1d_cnn", "tcn", "bigru", "transformer")
SEEDS = (42, 43, 44, 45, 46)
BUNDLE_ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bundle_path(relative):
    require(isinstance(relative, str) and bool(relative), "Missing bundle-relative path.")
    path = PurePosixPath(relative)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in relative,
            "Unsafe bundle-relative path.")
    resolved = (BUNDLE_ROOT / path).resolve()
    require(resolved.is_relative_to(BUNDLE_ROOT), "Path escapes the bundle directory.")
    require(resolved.is_file(), "Required bundle file is missing.")
    return resolved


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synthetic_events():
    """Deterministic artificial events, never drawn from a patient dataset."""
    time = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    curves = (
        5.0 + 2.0 * np.sin(np.pi * time) ** 2,
        6.2 + 3.5 * np.sin(np.pi * time) ** 2,
        8.0 + 5.0 * np.sin(np.pi * time) ** 2,
    )
    return [SimpleNamespace(subject_id=f"synthetic_{1 if index < 2 else 2}",
                            segment_id=f"synthetic_event_{index + 1}", label=-1,
                            values=curve.astype(np.float32))
            for index, curve in enumerate(curves)]


def verify_probability_table(frame, expected_rows):
    probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    require(probabilities.shape == (expected_rows, 3), "Unexpected probability table shape.")
    require(np.isfinite(probabilities).all(), "Non-finite prediction probability.")
    require(((probabilities >= 0) & (probabilities <= 1)).all(), "Probability outside [0, 1].")
    require(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0),
            "Class probabilities do not sum to one.")
    predicted = probabilities.argmax(axis=1)
    require(np.array_equal(frame["predicted_class"].to_numpy(), predicted),
            "Predicted class does not match probability argmax.")
    require(frame["predicted_label"].tolist() == [CLASS_NAMES[index] for index in predicted],
            "Predicted label does not match class order.")
    if "confidence" in frame:
        require(np.allclose(frame["confidence"].to_numpy(), probabilities.max(axis=1),
                            atol=1e-6, rtol=0), "Confidence does not match maximum probability.")


def verify_predictions(events, patients, records):
    verify_probability_table(events, 3)
    verify_probability_table(patients, 2)
    expected_keys = {(record.subject_id, record.segment_id) for record in records}
    actual_keys = list(zip(events["subject_id"], events["segment_id"]))
    require(len(actual_keys) == len(set(actual_keys)) and set(actual_keys) == expected_keys,
            "Events were dropped or duplicated.")
    require(patients["subject_id"].is_unique, "Duplicate patient output.")
    require(set(patients["subject_id"]) == {record.subject_id for record in records},
            "Patient output does not cover all input patients.")
    require((events["label"] == -1).all() and (patients["label"] == -1).all(),
            "Synthetic unknown labels changed during inference.")
    for _, patient in patients.iterrows():
        group = events.loc[events["subject_id"] == patient["subject_id"]]
        require(np.allclose(patient[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float),
                            group[list(PROBABILITY_COLUMNS)].mean().to_numpy(),
                            atol=1e-6, rtol=0),
                "Patient probabilities differ from mean event probabilities.")
        require(int(patient["n_segments"]) == len(group), "Patient event count is incorrect.")


def verify_one(entry):
    checkpoint = bundle_path(entry["checkpoint"])
    require(sha256(checkpoint) == entry["sha256"], "Checkpoint SHA256 does not match manifest.")
    require(checkpoint.stat().st_size == entry["bytes"], "Checkpoint size does not match manifest.")
    saved_config = json.loads(bundle_path(entry["config"]).read_text(encoding="utf-8"))
    architecture, seed = entry["architecture"], entry["seed"]
    records = synthetic_events()
    if architecture == "knn":
        result = load_knn(checkpoint)
        require(result.train_values.ndim == 2 and result.train_values.shape[1] == 64,
                "KNN reference vectors are not 64-point CGM inputs.")
        require(result.config.metric == "euclidean" and result.config.weights == "distance"
                and result.config.n_neighbors == 5, "Unexpected KNN configuration.")
    else:
        result = load_checkpoint(checkpoint, device="cpu")
        config = result.config
        require(config.architecture == architecture, "Architecture does not match manifest.")
        require(config.use_cgm and config.k == 1 and config.sample_unit == "event"
                and config.pooling == "mean", "Model is not the CGM-only event configuration.")
        require(config.metadata_dim == 0 and config.metadata_features == []
                and config.event_feature_dim == 0 and config.event_features == []
                and result.metadata_preprocessor is None and result.event_feature_preprocessor is None,
                "Checkpoint unexpectedly uses metadata or engineered features.")
        require(result.model.use_cgm and result.model.metadata_dim == 0
                and result.model.event_feature_dim == 0, "Model inputs disagree with CGM-only configuration.")
    require(asdict(result.config) == saved_config, "Saved configuration differs from checkpoint.")
    require(result.config.seed == seed and result.config.length == 64 and entry["length"] == 64,
            "Seed or sequence length does not match manifest.")
    with torch.inference_mode():
        if architecture == "knn":
            predictions = predict_knn(result, records)
            events, patients = predictions["events"], predictions["patients"]
        else:
            patients, events, _ = predict_segments(result.model, result.scaler, records,
                                                   k=1, sample_unit="event")
    verify_predictions(events, patients, records)


def verify_bundle():
    torch.set_num_threads(2)
    manifest = json.loads((BUNDLE_ROOT / "model_manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("schema") == "hba1c-model-handoff-v1", "Unsupported manifest schema.")
    require(manifest.get("class_names") == list(CLASS_NAMES), "Manifest class order is incorrect.")
    require(manifest.get("architectures") == list(ARCHITECTURES)
            and manifest.get("seeds") == list(SEEDS), "Manifest model/seed roster is incorrect.")
    entries = manifest["checkpoints"]
    pairs = [(entry["architecture"], entry["seed"]) for entry in entries]
    expected = {(architecture, seed) for architecture in ARCHITECTURES for seed in SEEDS}
    require(len(entries) == manifest.get("checkpoint_count") == 25
            and len(set(pairs)) == 25 and set(pairs) == expected,
            "Manifest must include each of the 25 model/seed combinations exactly once.")
    require(len({entry["checkpoint"] for entry in entries}) == 25,
            "Manifest reuses a checkpoint path.")
    source_checks = []
    for source in manifest["source_files"]:
        require(sha256(bundle_path(source["path"])) == source["sha256"],
                "Original model source SHA256 does not match manifest.")
        source_checks.append({"path": source["path"], "status": "passed"})
    results = []
    for entry in entries:
        row = {"architecture": entry["architecture"], "seed": entry["seed"]}
        try:
            verify_one(entry)
            row.update(status="passed", checkpoint_sha256="matched", cgm_only=True,
                       synthetic_events=3, synthetic_patients=2, probability_checks="passed",
                       patient_aggregation="passed")
        except Exception as error:
            # Loader errors can contain public dataset identifiers. Keep these out
            # of the shareable summary while returning a nonzero process status.
            row.update(status="failed", error_type=type(error).__name__)
            print(f"Verification failed: {entry['architecture']} seed {entry['seed']} "
                  f"({type(error).__name__}).", file=sys.stderr)
        results.append(row)
    passed = sum(row["status"] == "passed" for row in results)
    return {"schema": "hba1c-bundle-verification-v1", "status": "passed" if passed == 25 else "failed",
            "checkpoint_count": len(results), "passed": passed, "failed": len(results) - passed,
            "source_files_verified": len(source_checks), "input_type": "fixed_synthetic_CGM_only",
            "training_performed": False, "private_data_read": False,
            "checks": ["complete_unique_model_seed_roster", "checkpoint_sha256_and_size",
                       "original_source_sha256", "strict_original_checkpoint_loader_audits",
                       "CGM_only_configuration", "three_class_probabilities", "argmax_class_labels",
                       "patient_mean_event_probabilities"],
            "source_files": source_checks, "models": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Optional JSON report output path.")
    args = parser.parse_args()
    try:
        report = verify_bundle()
    except Exception as error:
        report = {"schema": "hba1c-bundle-verification-v1", "status": "failed",
                  "stage": "manifest_or_source_integrity", "error_type": type(error).__name__}
        print(f"Bundle integrity verification failed ({type(error).__name__}).", file=sys.stderr)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if report["status"] == "passed":
        print(f"PASS: {report['passed']}/25 checkpoints; 5 models x 5 seeds. "
              "SHA256, CGM-only configuration, three-class inference and patient aggregation verified. "
              "No training or private test data used.")
        return 0
    print(f"FAIL: {report.get('passed', 0)}/25 checkpoints verified.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
