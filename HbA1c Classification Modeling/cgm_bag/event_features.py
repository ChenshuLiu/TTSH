"""Portable event features measured on native CGM observations.

SEG current, saved event statistics, HbA1c and clinical metadata are never read.
Only the selected signal, its timestamps, and the already selected landmarks
and private phase assignments enter these calculations.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd


FEATURE_SCHEMA = "unified_cgm_event_features_v1"
FEATURE_NAMES = (
    "trough1_to_trough2_duration_min",
    "trough1_to_peak_duration_min",
    "peak_to_trough2_duration_min",
    "trough1_to_peak_mean_derivative_per_min",
    "peak_to_trough2_mean_derivative_per_min",
)
RISING_FEATURES = (FEATURE_NAMES[1], FEATURE_NAMES[3])
FEATURE_LABELS = {
    FEATURE_NAMES[0]: "总时长（分钟）",
    FEATURE_NAMES[1]: "上升时长（分钟）",
    FEATURE_NAMES[2]: "下降时长（分钟）",
    FEATURE_NAMES[3]: "上升平均导数（mmol/L/分钟）",
    FEATURE_NAMES[4]: "下降平均导数（mmol/L/分钟）",
}
FEATURE_UNITS = {name: "min" if "duration" in name else "mmol/L/min" for name in FEATURE_NAMES}
_SIGNALS = {"glucose_mmol_L", "predicted_cgm_mmol_L", "actual_cgm_mmol_L"}


def _phase_statistics(frame: pd.DataFrame, time_col: str, value_col: str,
                      start: pd.Timestamp, end: pd.Timestamp,
                      max_gap_minutes: float) -> tuple[float | None, dict[str, Any]]:
    """Mean of adjacent native slopes, with no interpolation or missing bridge.

    Missing rows invalidate their adjacent pairs. Duplicate timestamps and long
    gaps make the entire phase derivative unavailable rather than choose an
    arbitrary duplicate or summarize only one side of a discontinuous phase.
    """
    phase = frame.sort_values(time_col, kind="stable")
    times = pd.to_datetime(phase[time_col], errors="coerce")
    values = pd.to_numeric(phase[value_col], errors="coerce").to_numpy(float)
    finite = np.isfinite(values) & times.notna().to_numpy()
    observed = times.loc[finite]
    stats: dict[str, Any] = {
        "requested_start": str(start), "requested_end": str(end),
        "rows": int(len(phase)), "finite_points": int(finite.sum()),
        "missing_points": int((~finite).sum()),
        "first_finite_timestamp": str(observed.iloc[0]) if len(observed) else None,
        "last_finite_timestamp": str(observed.iloc[-1]) if len(observed) else None,
        "start_coverage_offset_min": float((observed.iloc[0] - start).total_seconds() / 60) if len(observed) else None,
        "end_coverage_offset_min": float((end - observed.iloc[-1]).total_seconds() / 60) if len(observed) else None,
        "adjacent_pairs": max(0, len(phase) - 1),
        "valid_adjacent_pairs": 0, "missing_adjacent_pairs": 0,
        "duplicate_timestamp_pairs": 0, "long_gap_pairs": 0,
        "positive_slope_pairs": 0, "negative_slope_pairs": 0, "zero_slope_pairs": 0,
    }
    if len(phase) < 2:
        return None, {**stats, "reason": "fewer_than_two_native_rows"}
    delta = times.diff().dt.total_seconds().to_numpy(float)[1:] / 60.0
    adjacent_finite = finite[1:] & finite[:-1]
    duplicates = np.isfinite(delta) & (delta == 0)
    long_gap = np.isfinite(delta) & (delta > max_gap_minutes)
    valid = adjacent_finite & np.isfinite(delta) & (delta > 0) & ~long_gap
    stats.update(
        valid_adjacent_pairs=int(valid.sum()),
        missing_adjacent_pairs=int((~adjacent_finite).sum()),
        duplicate_timestamp_pairs=int(duplicates.sum()),
        long_gap_pairs=int(long_gap.sum()),
        max_native_gap_min=float(np.max(delta[np.isfinite(delta)])) if np.isfinite(delta).any() else None,
    )
    if duplicates.any():
        return None, {**stats, "reason": "duplicate_timestamps"}
    if long_gap.any():
        return None, {**stats, "reason": "long_native_gap"}
    if times.isna().any():
        return None, {**stats, "reason": "invalid_timestamp"}
    if not valid.any():
        return None, {**stats, "reason": "no_valid_adjacent_slopes"}
    slopes = np.diff(values)[valid] / delta[valid]
    stats.update(positive_slope_pairs=int((slopes > 0).sum()),
                 negative_slope_pairs=int((slopes < 0).sum()),
                 zero_slope_pairs=int((slopes == 0).sum()))
    result = float(slopes.mean())
    if not np.isfinite(result):
        return None, {**stats, "reason": "nonfinite_derivative"}
    return result, {**stats, "reason": "available", "mean_sign": int(np.sign(result))}


def compute_event_features(frame: pd.DataFrame, time_col: str, value_col: str,
                           start: pd.Timestamp, peak: pd.Timestamp, end: pd.Timestamp,
                           *, kind: str, max_gap_minutes: float,
                           phase_col: str | None = None) -> dict[str, Any]:
    """Return feature values and reproducible provenance for one selected event.

    Durations use the selected landmark timestamps. Derivatives use only native
    rows inside that selected window. A rising-only selection never reads the
    falling limb or invents trough2. For private data, source phase labels own
    their rows even when a timestamp equals the peak.
    """
    if value_col not in _SIGNALS:
        raise ValueError("Event features require a canonical CGM signal")
    if kind not in {"trough_to_peak", "trough_peak_trough"}:
        raise ValueError("Unsupported event feature interval")
    if not np.isfinite(max_gap_minutes) or max_gap_minutes <= 0:
        raise ValueError("Event feature max_gap_minutes must be positive and finite")
    start, peak, end = map(pd.Timestamp, (start, peak, end))
    if any(pd.isna(t) for t in (start, peak, end)) or not start < peak <= end:
        raise ValueError("Event features require valid, ordered selected landmarks")
    if kind == "trough_to_peak" and peak != end:
        raise ValueError("Rising-only event must end at its selected peak")
    if kind == "trough_peak_trough" and not peak < end:
        raise ValueError("Full event requires a falling limb")
    selected = frame.loc[frame[time_col].between(start, end)].copy()
    phase_authoritative = phase_col is not None and phase_col in selected
    if phase_authoritative:
        phases = selected[phase_col].fillna("").astype(str).str.strip().str.lower()
        if not phases.isin({"trough1_to_peak", "peak_to_trough2"}).all():
            raise ValueError("Invalid source phase assignment for CGM event features")
        rise = selected.loc[phases.eq("trough1_to_peak")]
        fall = selected.loc[phases.eq("peak_to_trough2")]
    else:
        rise = selected.loc[selected[time_col].between(start, peak)]
        fall = selected.loc[selected[time_col].between(peak, end)]
    features: dict[str, float | None] = dict.fromkeys(FEATURE_NAMES)
    reasons = {name: "not_requested_interval" for name in FEATURE_NAMES}
    features[FEATURE_NAMES[1]] = float((peak - start).total_seconds() / 60)
    reasons[FEATURE_NAMES[1]] = "available"
    rising_derivative, rising_qc = _phase_statistics(rise, time_col, value_col, start, peak, max_gap_minutes)
    features[FEATURE_NAMES[3]] = rising_derivative
    reasons[FEATURE_NAMES[3]] = rising_qc["reason"]
    falling_qc: dict[str, Any] = {"reason": "not_requested_interval"}
    if kind == "trough_peak_trough":
        features[FEATURE_NAMES[0]] = float((end - start).total_seconds() / 60)
        features[FEATURE_NAMES[2]] = float((end - peak).total_seconds() / 60)
        reasons[FEATURE_NAMES[0]] = reasons[FEATURE_NAMES[2]] = "available"
        falling_derivative, falling_qc = _phase_statistics(fall, time_col, value_col, peak, end, max_gap_minutes)
        features[FEATURE_NAMES[4]] = falling_derivative
        reasons[FEATURE_NAMES[4]] = falling_qc["reason"]
    qc = {
        "kind": kind, "signal": value_col,
        "duration_policy": "selected_landmark_timestamp_difference_minutes",
        "derivative_policy": "signed_unweighted_mean_of_adjacent_native_slopes",
        "resampling_used": False, "endpoint_interpolation_used": False,
        "missing_policy": "exclude_pairs_touching_nonfinite_rows_never_bridge",
        "duplicate_policy": "entire_phase_derivative_unavailable",
        "gap_policy": "entire_phase_derivative_unavailable_if_native_gap_exceeds_limit",
        "max_gap_minutes": float(max_gap_minutes),
        "boundary_convention": "source_phase_assignment_authoritative" if phase_authoritative else "both_limbs_include_peak_timestamp",
        "feature_reasons": reasons, "rising": rising_qc, "falling": falling_qc,
    }
    return {"cgm_features": features, "cgm_feature_qc": qc,
            "cgm_feature_schema": FEATURE_SCHEMA, "cgm_feature_signal": value_col}


def _validate_features(features: Sequence[str]) -> list[str]:
    result = list(features)
    if not result or len(set(result)) != len(result) or set(result) - set(FEATURE_NAMES):
        raise ValueError("Select unique supported CGM event features")
    return result


def _segment_key(segment: Any) -> str:
    return json.dumps([str(segment.subject_id), str(segment.segment_id)], ensure_ascii=False, separators=(",", ":"))


def _feature_values(segments: Sequence[Any], features: list[str], *, fitting: bool = False) -> np.ndarray:
    values = np.empty((len(segments), len(features)), dtype=float)
    for row, segment in enumerate(segments):
        info = segment.info
        if info.get("cgm_feature_schema") != FEATURE_SCHEMA:
            raise ValueError(f"Segment {segment.segment_id} lacks the current CGM event feature schema; extract segments again")
        signal = info.get("cgm_feature_signal")
        if signal not in _SIGNALS or (segment.dataset == "private") != (signal != "glucose_mmol_L"):
            raise ValueError("CGM event feature source does not match the segment dataset")
        if fitting and signal != "glucose_mmol_L":
            raise ValueError("CGM event preprocessing must be fitted on public training segments only")
        raw = info.get("cgm_features")
        if not isinstance(raw, dict) or set(raw) != set(FEATURE_NAMES):
            raise ValueError("Incomplete CGM event feature schema")
        reasons = info.get("cgm_feature_qc", {}).get("feature_reasons", {})
        for col, feature in enumerate(features):
            if reasons.get(feature) == "not_requested_interval" or (info.get("kind") == "trough_to_peak" and feature not in RISING_FEATURES):
                raise ValueError(f"CGM event feature {feature} is unavailable for this interval; use rising features or a full trough-peak-trough model")
            value = raw[feature]
            if value is None:
                values[row, col] = np.nan
            elif isinstance(value, (bool, str)) or not np.isscalar(value) or not np.isfinite(float(value)):
                raise ValueError(f"Invalid raw CGM event feature {feature}; use finite numeric values or None")
            else:
                values[row, col] = float(value)
    return values


@dataclass
class EventFeaturePreprocessor:
    features: list[str]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    fitted_subject_ids: list[str]
    fitted_segment_ids: list[str]

    @property
    def dim(self) -> int:
        return 2 * len(self.features)

    @property
    def feature_names(self) -> list[str]:
        return [f"{f}_z" for f in self.features] + [f"{f}_missing" for f in self.features]

    @property
    def feature_groups(self) -> dict[str, list[int]]:
        return {f: [i, i + len(self.features)] for i, f in enumerate(self.features)}

    @property
    def baseline_vector(self) -> np.ndarray:
        return np.concatenate([(self.medians - self.means) / self.scales,
                               np.zeros(len(self.features))]).astype(np.float32)

    @classmethod
    def fit(cls, segments: Sequence[Any], features: Sequence[str]) -> "EventFeaturePreprocessor":
        features = _validate_features(features)
        segments = list(segments)
        if not segments:
            raise ValueError("CGM event preprocessing requires public training segments")
        keys = [_segment_key(segment) for segment in segments]
        ids = list(dict.fromkeys(str(segment.subject_id) for segment in segments))
        if len(keys) != len(set(keys)) or any(not str(s.subject_id).strip() or not str(s.segment_id).strip() for s in segments):
            raise ValueError("CGM event preprocessing requires unique nonempty patient/event keys")
        values = _feature_values(segments, features, fitting=True)
        missing = np.isnan(values)
        absent = [f for i, f in enumerate(features) if missing[:, i].all()]
        if absent:
            raise ValueError(f"CGM event features have no finite values in public training segments: {absent}")
        medians = np.nanmedian(values, axis=0)
        imputed = np.where(missing, medians, values)
        means, scales = imputed.mean(axis=0), imputed.std(axis=0)
        if not all(np.isfinite(array).all() for array in (medians, means, scales)):
            raise ValueError("CGM event training feature statistics are nonfinite")
        scales = np.where(scales > 1e-8, scales, 1.0)
        return cls(features, medians, means, scales, ids, keys)

    def transform(self, segments: Sequence[Any]) -> np.ndarray:
        values = _feature_values(list(segments), self.features)
        missing = np.isnan(values)
        imputed = np.where(missing, self.medians, values)
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.concatenate([(imputed - self.means) / self.scales,
                                     missing.astype(float)], axis=1).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("CGM event features exceed the finite model input range")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, "schema": FEATURE_SCHEMA, "features": list(self.features),
                "medians": self.medians.tolist(), "means": self.means.tolist(), "scales": self.scales.tolist(),
                "fitted_subject_ids": list(self.fitted_subject_ids), "fitted_segment_ids": list(self.fitted_segment_ids),
                "feature_names": self.feature_names,
                "units": {f: FEATURE_UNITS[f] for f in self.features},
                "fit_signal": "glucose_mmol_L"}

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "EventFeaturePreprocessor":
        if state.get("version") != 1 or state.get("schema") != FEATURE_SCHEMA or state.get("fit_signal") != "glucose_mmol_L":
            raise ValueError("Unsupported CGM event preprocessor schema or training signal")
        features = _validate_features(state.get("features", []))
        try:
            arrays = [np.asarray(state[k], dtype=float) for k in ("medians", "means", "scales")]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid CGM event preprocessor statistics") from exc
        if any(a.shape != (len(features),) or not np.isfinite(a).all() for a in arrays) or (arrays[2] <= 0).any():
            raise ValueError("Invalid CGM event preprocessor statistics")
        ids, keys = state.get("fitted_subject_ids", []), state.get("fitted_segment_ids", [])
        if not isinstance(ids, list) or not ids or any(not isinstance(s, str) or not s.strip() for s in ids) or len(set(ids)) != len(ids):
            raise ValueError("Invalid fitted CGM event training subject IDs")
        try:
            decoded = [json.loads(key) for key in keys]
            valid = (isinstance(keys, list) and bool(keys) and len(set(keys)) == len(keys) and
                     all(isinstance(pair, list) and len(pair) == 2 and
                         all(isinstance(s, str) and s.strip() for s in pair) for pair in decoded) and
                     {pair[0] for pair in decoded} == set(ids) and
                     all(json.dumps(pair, ensure_ascii=False, separators=(",", ":")) == key for pair, key in zip(decoded, keys)))
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("Invalid fitted CGM event training segment keys")
        result = cls(features, *arrays, list(ids), list(keys))
        if state.get("feature_names") != result.feature_names or state.get("units") != {f: FEATURE_UNITS[f] for f in features}:
            raise ValueError("Invalid CGM event preprocessor feature names or units")
        return result


def event_features_frame(segments: Sequence[Any], features: Sequence[str] | None = None) -> pd.DataFrame:
    """Export one row per event, including raw values and full feature audit."""
    selected = list(FEATURE_NAMES) if features is None else _validate_features(features)
    rows = []
    for segment in segments:
        info = segment.info
        raw = info.get("cgm_features", {})
        qc = info.get("cgm_feature_qc", {})
        rows.append({"subject_id": segment.subject_id, "segment_id": segment.segment_id,
                     "event_uid": info.get("event_uid", ""), "dataset": segment.dataset,
                     "label": segment.label, "kind": info.get("kind", ""),
                     "cgm_feature_schema": info.get("cgm_feature_schema", ""),
                     "cgm_feature_signal": info.get("cgm_feature_signal", ""),
                     **{feature: raw.get(feature) for feature in selected},
                     **{f"{feature}_reason": qc.get("feature_reasons", {}).get(feature, "missing_schema") for feature in selected},
                     "cgm_feature_qc_json": json.dumps(qc, ensure_ascii=False, sort_keys=True, allow_nan=False)})
    return pd.DataFrame(rows)
