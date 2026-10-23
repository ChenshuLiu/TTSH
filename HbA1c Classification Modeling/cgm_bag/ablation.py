"""Explicit modality experiment plans and public-cohort comparison audits.

Waveform modes run once per requested CGM encoder. Tabular modes use a single
MLP each, without duplicating them under four unused encoder names. These
helpers never choose a model using private test performance.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

import pandas as pd

from .event_features import FEATURE_NAMES


MODE_LABELS = {
    "cgm": "仅 CGM 波形",
    "features": "仅 CGM 事件特征",
    "cgm_features": "CGM 波形 + 事件特征",
    "metadata": "仅临床 metadata",
    "cgm_metadata": "CGM 波形 + metadata",
    "metadata_features": "metadata + CGM 事件特征",
    "cgm_metadata_features": "CGM 波形 + metadata + 事件特征",
}
# Tuple order is (use_cgm, use_event_features, use_metadata).
MODES = {
    "cgm": (True, False, False),
    "features": (False, True, False),
    "cgm_features": (True, True, False),
    "metadata": (False, False, True),
    "cgm_metadata": (True, False, True),
    "metadata_features": (False, True, True),
    "cgm_metadata_features": (True, True, True),
}
FEATURE_VARIANT_LABELS = {
    "all": "全部所选事件特征",
    "durations": "仅时长特征",
    "derivatives": "仅平均导数特征",
}
CGM_ARCHITECTURES = ("1d_cnn", "tcn", "bigru", "transformer")


def mode_parts(mode: str) -> tuple[bool, bool, bool]:
    """Return (waveform enabled, event features enabled, metadata enabled)."""
    if mode not in MODES:
        raise ValueError(f"Unsupported input mode {mode!r}; choose from {list(MODE_LABELS)}")
    return MODES[mode]


def _get(config: Any, name: str, default: Any = None) -> Any:
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def _feature_names(features: Sequence[str]) -> list[str]:
    if isinstance(features, (str, bytes)):
        raise ValueError("Event features must be a sequence of feature names")
    values = list(features)
    if any(not isinstance(name, str) or name not in FEATURE_NAMES for name in values):
        raise ValueError("Unsupported unified CGM event feature name")
    return list(dict.fromkeys(values))


def feature_variant_from_names(features: Sequence[str]) -> str:
    """Infer a reload-stable variant from the actual selected feature types.

    Empty means no feature branch. A duration-only custom selection is still
    'durations', including when it was requested through the 'all' option.
    """
    names = _feature_names(features)
    if not names:
        return "none"
    if all("_duration_" in name for name in names):
        return "durations"
    if all("_mean_derivative_" in name for name in names):
        return "derivatives"
    return "all"


def mode_from_config(config: Any) -> str:
    use_cgm = _get(config, "use_cgm", True)
    if not isinstance(use_cgm, bool):
        raise ValueError("use_cgm must be a boolean")
    event_features = _feature_names(_get(config, "event_features", []))
    parts = (use_cgm, bool(event_features), bool(_get(config, "metadata_features", [])))
    for mode, candidate in MODES.items():
        if parts == candidate:
            return mode
    raise ValueError("A model must enable at least one input modality")


def _model_key(architecture: str, mode: str, features: Sequence[str]) -> str:
    use_cgm, use_features, use_metadata = mode_parts(mode)
    if use_cgm:
        if architecture not in CGM_ARCHITECTURES:
            raise ValueError("Waveform modes require a supported CGM architecture")
    elif architecture != "mlp":
        raise ValueError("Tabular-only modes require architecture='mlp'")
    suffix = ("__metadata" if use_metadata else "") + ("__features" if use_features else "")
    # Keep the existing checkpoint folder convention for both optional branches:
    # '__metadata__features', including the separator before each branch.
    variant = feature_variant_from_names(features)
    if use_features and variant in {"durations", "derivatives"}:
        suffix += f"__{variant}"
    return architecture + suffix


def model_key_from_config(config: Any) -> str:
    """Canonical model name using saved modalities and actual feature types."""
    mode = mode_from_config(config)
    return _model_key(_get(config, "architecture", "1d_cnn"), mode,
                      _get(config, "event_features", []))


def build_jobs(architectures: Sequence[str], modes: Sequence[str],
               selected_event_features: Sequence[str],
               feature_variants: Sequence[str] = ("all",)) -> list[dict[str, Any]]:
    """Build deterministic, deduplicated jobs for the selected modality study.

    Feature variants filter the user's selected names, preserving their order.
    Modes without features run once per applicable architecture and ignore the
    variant count. Tabular-only modes each use one MLP. If 'all' and a partial
    variant yield the same actual features, only one canonical job is emitted.
    """
    if any(isinstance(value, (str, bytes)) for value in (architectures, modes, feature_variants)):
        raise ValueError("Architectures, modes and feature variants must be sequences")
    architectures = list(dict.fromkeys(architectures))
    modes = list(dict.fromkeys(modes))
    variants = list(dict.fromkeys(feature_variants))
    features = _feature_names(selected_event_features)
    if not modes:
        raise ValueError("Select at least one input mode")
    if not variants or any(variant not in FEATURE_VARIANT_LABELS for variant in variants):
        raise ValueError("Select at least one supported event feature variant")
    if any(name not in CGM_ARCHITECTURES for name in architectures):
        raise ValueError("Unsupported CGM architecture in experiment plan")
    parts = {mode: mode_parts(mode) for mode in modes}
    if any(flags[0] for flags in parts.values()) and not architectures:
        raise ValueError("Select a CGM architecture for waveform input modes")
    if any(flags[1] for flags in parts.values()) and not features:
        raise ValueError("Feature input modes require selected CGM event features")
    subsets = {}
    if any(flags[1] for flags in parts.values()):
        for variant in variants:
            names = [name for name in features if variant == "all" or
                     (variant == "durations" and "_duration_" in name) or
                     (variant == "derivatives" and "_mean_derivative_" in name)]
            if not names:
                raise ValueError(f"Requested event feature variant {variant!r} has no selected features")
            subsets[variant] = names
    jobs = []
    seen: set[str] = set()
    for mode in modes:
        use_cgm, use_features, use_metadata = parts[mode]
        for architecture in architectures if use_cgm else ("mlp",):
            for names in subsets.values() if use_features else ([],):
                key = _model_key(architecture, mode, names)
                if key in seen:
                    continue
                seen.add(key)
                jobs.append({"model_key": key, "architecture": architecture, "mode": mode,
                             "use_cgm": use_cgm, "event_features": list(names),
                             "use_metadata": use_metadata,
                             "feature_variant": feature_variant_from_names(names)})
    return jobs


def _patient_ids(split: Mapping[str, Any], field: str) -> list[str]:
    raw = split.get(field)
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"Public comparison requires nonempty {field}")
    if any(not isinstance(value, str) or not value.strip() for value in raw) or len(set(raw)) != len(raw):
        raise ValueError(f"Public comparison requires unique nonempty patient IDs in {field}")
    return sorted(raw)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                                     sort_keys=True).encode("utf-8")).hexdigest()


def compare_public_models(results: Mapping[str, Any]) -> pd.DataFrame:
    """Assert matched patient splits within each declared public dataset cohort.

    The table records hyperparameters instead of assuming they are equal.
    Available source-data fingerprints must agree within a cohort, so equal
    patient IDs cannot hide different segment cohorts. Explicit audited dataset
    cohorts may differ, and their comparisons are labeled as confounded by
    training cohort. Legacy models remain one strict group. Adapted models cannot be
    included in this public comparison. Validation metrics are checkpoint-era
    training-history values; they are not independent test-set performance.
    """
    from .cohorts import cohort_columns
    rows = []
    for result in results.values():
        if not isinstance(result.split, Mapping) or result.split.get("adaptation") is not None:
            raise ValueError("Public ablation comparison requires public-trained models")
    cohorts = {key: cohort_columns(result) for key, result in results.items()}
    explicit = bool(cohorts) and all(row["public_cohort_explicit"] for row in cohorts.values())
    reference_splits, fingerprints = {}, {}
    for key, result in results.items():
        config, split = result.config, result.split
        if not isinstance(split, Mapping) or split.get("adaptation") is not None:
            raise ValueError("Public ablation comparison requires public-trained models")
        train_ids, val_ids = _patient_ids(split, "train_ids"), _patient_ids(split, "val_ids")
        if set(train_ids) & set(val_ids):
            raise ValueError("A model's public training and validation patients overlap")
        current_split = (train_ids, val_ids)
        group = cohorts[key]["public_cohort_id"] if explicit else "legacy_public"
        if group not in reference_splits:
            reference_splits[group] = current_split
        elif reference_splits[group] != current_split:
            raise ValueError("Ablation models must use identical public training and validation patient membership")
        data_fingerprint = split.get("data_fingerprint_sha256")
        if data_fingerprint:
            fingerprints.setdefault(group, set()).add(str(data_fingerprint))
            if len(fingerprints[group]) > 1:
                raise ValueError("Ablation models have different public data fingerprints/segment cohorts")
        mode = mode_from_config(config)
        use_cgm, use_features, use_metadata = mode_parts(mode)
        features = _feature_names(_get(config, "event_features", []))
        metadata = list(_get(config, "metadata_features", []))
        variant = feature_variant_from_names(features)
        row = {"model_key": str(key), "canonical_model_key": model_key_from_config(config),
               **cohorts[key],
               "mode": mode, "input_mode": mode, "input_label": MODE_LABELS[mode],
               "architecture": _get(config, "architecture"), "use_cgm": use_cgm,
               "use_event_features": use_features, "use_metadata": use_metadata,
               "event_feature_count": len(features), "event_features": ",".join(features),
               "metadata_feature_count": len(metadata), "metadata_features": ",".join(metadata),
               "feature_variant": variant, "feature_variant_label": FEATURE_VARIANT_LABELS.get(variant, "无事件特征"),
               "sample_unit": _get(config, "sample_unit", "patient_bag"),
               "train_patient_count": len(train_ids), "val_patient_count": len(val_ids),
               "train_membership_sha256": _fingerprint(train_ids),
               "val_membership_sha256": _fingerprint(val_ids),
               "split_membership_sha256": _fingerprint({"train": train_ids, "validation": val_ids}),
               "same_patient_membership": True,
               "data_fingerprint_sha256": data_fingerprint or "",
               "train_segment_count": split.get("train_segment_count"),
               "val_segment_count": split.get("val_segment_count"),
               "best_epoch": getattr(result, "best_epoch", None)}
        for name in ("seed", "epochs", "batch_size", "learning_rate", "weight_decay", "patience",
                     "hidden_dim", "dropout", "class_weights", "val_fraction"):
            row[name] = _get(config, name)
        parameters = getattr(getattr(result, "model", None), "parameters", None)
        if callable(parameters):
            tensors = list(parameters())
            row["total_parameter_count"] = sum(int(tensor.numel()) for tensor in tensors)
            row["trainable_parameter_count"] = sum(int(tensor.numel()) for tensor in tensors if tensor.requires_grad)
        else:
            row["total_parameter_count"] = row["trainable_parameter_count"] = None
        history = getattr(result, "history", None)
        selected = (history.loc[history.epoch == row["best_epoch"]]
                    if isinstance(history, pd.DataFrame) and "epoch" in history else pd.DataFrame())
        for metric in ("val_loss", "val_accuracy", "val_event_accuracy", "val_patient_accuracy"):
            row[metric] = float(selected.iloc[-1][metric]) if len(selected) and metric in selected else None
        rows.append(row)
    if rows:
        same_split = len({row["split_membership_sha256"] for row in rows}) == 1
        multiple_cohorts = len({row["public_cohort_id"] for row in rows}) > 1
        for row in rows:
            row["same_patient_membership"] = same_split
            row["same_patient_membership_within_cohort"] = True
            row["modality_comparison_group"] = row["public_cohort_id"]
            row["cross_cohort_comparison"] = multiple_cohorts
            row["comparability"] = ("Compare modalities within each public cohort only; cross-cohort differences also reflect training datasets and patients."
                                    if multiple_cohorts else "Same public cohort and patient split; controlled modality comparison.")
    return pd.DataFrame(rows)
