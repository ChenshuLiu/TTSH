"""Feature engineering for labeled CGM/SEG/SEL temporal peak annotations.

Features calculated by this module
----------------------------------
For each labeled peak aggregate in ``selected_peak_metadata``:

1. ``duration_trough1_to_peak_seconds``
   Timestamp duration from the left trough label to the peak label.
2. ``duration_peak_to_trough2_seconds``
   Timestamp duration from the peak label to the right trough label.
3. ``duration_trough1_to_trough2_seconds``
   Timestamp duration across the full peak window.
4. ``point_values``
   Matched signal values at trough1, peak, and trough2.  CGM stores
   ``cgm_value``; SEG/SEL store ``raw_delta_value``.  SEG also stores
   ``converted_concentration`` using the supplied conversion function.  SEG
   and SEL additionally store explicit uM converted concentrations using
   signal-specific raw-delta equations.
5. ``trough1_to_trough2_auc``
   Trapezoidal area under the curve over real timestamp seconds.  CGM uses
   ``cgm_auc_seconds``; SEG/SEL use ``raw_delta_auc_seconds``.  SEG also stores
   ``converted_concentration_auc_seconds``.  SEG/SEL uM conversions store
   signal-specific converted concentration AUC values.
6. ``interval_derivative_features``
   First-order derivative summaries after normalizing each interval duration
   to a 0-1 time axis.  Intervals are ``trough1_to_peak``, ``meal_to_peak``,
   and ``peak_to_trough2``.  Each interval stores max, min, and average
   derivative plus point/derivative counts.
7. ``num_preceding_peaks_since_instance`` / ``num_presceding_peak``
   Optional count of same-signal peaks between the preceding diet instance and
   the current peak when aggregate context is supplied.

Pipeline usage
--------------
Use ``process_annotation_payload(...)`` to update an in-memory annotation JSON
payload while preserving its existing structure.  Use
``export_feature_engineered_annotations(...)`` to read an annotation JSON file,
backfill all supported uni-modal temporal features, and write a JSON file with
the same top-level structure used by the alignment dashboard.
"""

import copy
import json
import math
import os
import re
import sys
import argparse
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Resolve loaders lazily to avoid circular imports with the dashboard server.

def load_cgm_points_in_range(*args, **kwargs):
    # Reuse the running server when launched as a script.
    main_module = sys.modules.get("__main__")
    if getattr(main_module, "__file__", None) and os.path.realpath(main_module.__file__) == os.path.realpath(os.path.join(os.path.dirname(__file__), "server.py")):
        alignment = main_module
    else:
        import server as alignment
    return alignment.load_cgm_points_in_range(*args, **kwargs)


def load_processed_series(*args, **kwargs):
    # Reuse the running server when launched as a script.
    main_module = sys.modules.get("__main__")
    if getattr(main_module, "__file__", None) and os.path.realpath(main_module.__file__) == os.path.realpath(os.path.join(os.path.dirname(__file__), "server.py")):
        alignment = main_module
    else:
        import server as alignment
    return alignment.load_processed_series(*args, **kwargs)


def load_tool1_series(*args, **kwargs):
    # Reuse the running server when launched as a script.
    main_module = sys.modules.get("__main__")
    if getattr(main_module, "__file__", None) and os.path.realpath(main_module.__file__) == os.path.realpath(os.path.join(os.path.dirname(__file__), "server.py")):
        alignment = main_module
    else:
        import server as alignment
    return alignment.load_tool1_series(*args, **kwargs)


def safe_rel_path(*args, **kwargs):
    # Reuse the running server when launched as a script.
    main_module = sys.modules.get("__main__")
    if getattr(main_module, "__file__", None) and os.path.realpath(main_module.__file__) == os.path.realpath(os.path.join(os.path.dirname(__file__), "server.py")):
        alignment = main_module
    else:
        import server as alignment
    return alignment.safe_rel_path(*args, **kwargs)



FEATURE_ENGINEERING_VERSION = "2026-07-04-seb-um-concentration-v2"


def default_parse_timestamp(text):
    s = str(text or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def default_concentration_from_delta(delta_value):
    if delta_value is None:
        return None
    try:
        return (float(delta_value) - 0.0188) / 0.00185
    except Exception:
        return None


def converted_concentration_um(signal, raw_delta_value):
    value = to_finite_float(raw_delta_value)
    if value is None:
        return None
    sig = str(signal or "").strip().lower()
    if sig == "sel":
        return (value - 0.00638) / 2.9145e-4
    if sig == "seg":
        return (value - 0.00453) / 0.00128
    return None


def calculated_feature_manifest():
    """Return a compact manifest of feature groups produced by this module."""
    return {
        "version": FEATURE_ENGINEERING_VERSION,
        "uni_modal_peak_metadata_features": [
            "duration_trough1_to_peak_seconds",
            "duration_peak_to_trough2_seconds",
            "duration_trough1_to_trough2_seconds",
            "point_values.{trough1,peak,trough2}.{matched_timestamp,matched_index,time_offset_seconds,cgm_value,raw_delta_value,converted_concentration,SEG_converted_concentration_uM,SEL_converted_concentration_uM}",
            "trough1_to_trough2_auc.{duration_seconds,point_count,cgm_auc_seconds,raw_delta_auc_seconds,converted_concentration_auc_seconds,SEG_converted_concentration_uM_auc_seconds,SEL_converted_concentration_uM_auc_seconds}",
            "interval_derivative_features.{trough1_to_peak,meal_to_peak,peak_to_trough2}.{max_derivative,min_derivative,average_derivative,point_count,derivative_count}",
            "num_preceding_peaks_since_instance",
            "num_presceding_peak",
        ],
    }


def source_rel_from_processed_path(rel_path):
    rel = safe_rel_path(rel_path)
    marker = "annotated_data/series/"
    if marker in rel:
        return "raw_data/sensor/" + rel.split(marker, 1)[1]
    return rel


def independent_cgm_series_for_case(rel_path, processed_rows=None, parse_timestamp=None):
    if parse_timestamp is None:
        raise ValueError("parse_timestamp is required")
    if processed_rows is None:
        processed_rows = load_processed_series(rel_path)
    if not processed_rows:
        return []
    start_dt = parse_timestamp(processed_rows[0].get("Timestamp", ""))
    end_dt = parse_timestamp(processed_rows[-1].get("Timestamp", ""))
    if start_dt is None or end_dt is None:
        return []
    try:
        cleaned_points = load_cgm_points_in_range(rel_path, start_dt, end_dt)
    except Exception:
        cleaned_points = []
    if cleaned_points:
        return [
            {"Timestamp": point.get("timestamp", ""), "CGM": point.get("value")}
            for point in cleaned_points
            if point.get("timestamp") and point.get("value") is not None
        ]
    source_rel = source_rel_from_processed_path(rel_path)
    try:
        source_rows = load_tool1_series(source_rel)
    except Exception:
        source_rows = []
    out = []
    for row in source_rows:
        cgm = row.get("CGM")
        if cgm is None:
            continue
        dt = parse_timestamp(row.get("Timestamp", ""))
        if dt is None or dt < start_dt or dt > end_dt:
            continue
        try:
            value = float(cgm)
        except Exception:
            continue
        out.append({"Timestamp": row.get("Timestamp", ""), "CGM": value})
    if out:
        return out
    # Fallback for cases where only the processed file is available.
    for row in processed_rows:
        cgm = row.get("CGM")
        if cgm is None:
            continue
        try:
            value = float(cgm)
        except Exception:
            continue
        out.append({"Timestamp": row.get("Timestamp", ""), "CGM": value})
    return out


def to_finite_float(value):
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def signal_value_column(signal):
    sig = str(signal or "").strip().lower()
    if sig == "sel":
        return "Delta_Channel2_minus_Channel3"
    if sig == "cgm":
        return "CGM"
    return "Delta_Channel2_minus_Channel1"


def signal_points_for_interval(rows, signal, start_ts, end_ts, parse_timestamp, cgm_points=None):
    start_dt = parse_timestamp(start_ts)
    end_dt = parse_timestamp(end_ts)
    if start_dt is None or end_dt is None or end_dt <= start_dt:
        return []
    sig = str(signal or "").strip().lower()
    source_rows = cgm_points if sig == "cgm" and cgm_points is not None else rows
    column = signal_value_column(sig)
    points = []
    for row in source_rows or []:
        dt = parse_timestamp(row.get("Timestamp", ""))
        if dt is None or dt < start_dt or dt > end_dt:
            continue
        value = to_finite_float(row.get(column))
        if value is None:
            continue
        points.append((dt, value))
    points.sort(key=lambda item: item[0])
    return points


def normalized_interval_derivative_summary(points, start_ts, end_ts, parse_timestamp):
    start_dt = parse_timestamp(start_ts)
    end_dt = parse_timestamp(end_ts)
    if start_dt is None or end_dt is None or end_dt <= start_dt:
        return {
            "max_derivative": None,
            "min_derivative": None,
            "average_derivative": None,
            "point_count": len(points or []),
            "derivative_count": 0,
            "time_normalized_start": 0,
            "time_normalized_end": 1,
        }
    duration = (end_dt - start_dt).total_seconds()
    normalized = []
    for dt, value in points or []:
        x = (dt - start_dt).total_seconds() / duration
        if x < 0 or x > 1:
            continue
        normalized.append((x, value))
    slopes = []
    for idx in range(1, len(normalized)):
        x0, y0 = normalized[idx - 1]
        x1, y1 = normalized[idx]
        dx = x1 - x0
        if dx <= 0:
            continue
        slopes.append((y1 - y0) / dx)
    if not slopes:
        return {
            "max_derivative": None,
            "min_derivative": None,
            "average_derivative": None,
            "point_count": len(normalized),
            "derivative_count": 0,
            "time_normalized_start": 0,
            "time_normalized_end": 1,
        }
    return {
        "max_derivative": max(slopes),
        "min_derivative": min(slopes),
        "average_derivative": sum(slopes) / len(slopes),
        "point_count": len(normalized),
        "derivative_count": len(slopes),
        "time_normalized_start": 0,
        "time_normalized_end": 1,
    }


def nearest_signal_value_at_timestamp(rows, signal, timestamp, parse_timestamp):
    target_dt = parse_timestamp(timestamp)
    if target_dt is None:
        return None
    column = signal_value_column(signal)
    best = None
    best_dist = None
    for idx, row in enumerate(rows or []):
        value = to_finite_float(row.get(column))
        if value is None:
            continue
        dt = parse_timestamp(row.get("Timestamp", ""))
        if dt is None:
            continue
        dist = abs((dt - target_dt).total_seconds())
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = {
                "index": idx,
                "timestamp": row.get("Timestamp", ""),
                "time_offset_seconds": dist,
                "value": value,
            }
    return best


def aggregate_point_indices(aggregate_id):
    match = re.search(r"_p(\d+)_l(\d+)_r(\d+)$", str(aggregate_id or ""))
    if not match:
        return {}
    return {
        "peak": int(match.group(1)),
        "trough1": int(match.group(2)),
        "trough2": int(match.group(3)),
    }


def signal_value_at_index(rows, signal, index):
    if index is None:
        return None
    try:
        idx = int(index)
    except Exception:
        return None
    if idx < 0 or idx >= len(rows or []):
        return None
    row = rows[idx]
    value = to_finite_float(row.get(signal_value_column(signal)))
    if value is None:
        return None
    return {
        "index": idx,
        "timestamp": row.get("Timestamp", ""),
        "time_offset_seconds": 0,
        "value": value,
    }


def point_value_features_for_peak(rows, peak_meta, parse_timestamp, concentration_from_delta, cgm_points=None):
    if not isinstance(peak_meta, dict):
        return {}
    signal = str(peak_meta.get("signal") or "").strip().lower()
    point_specs = {
        "trough1": peak_meta.get("trough1_timestamp", ""),
        "peak": peak_meta.get("peak_timestamp", ""),
        "trough2": peak_meta.get("trough2_timestamp", ""),
    }
    index_lookup = aggregate_point_indices(peak_meta.get("aggregate_id"))
    source_rows = cgm_points if signal == "cgm" and cgm_points is not None else rows
    out = {}
    for point_name, timestamp in point_specs.items():
        # CGM aggregate indices can be based on the display timeline; when an
        # independent CGM series is available, timestamp matching is safer.
        matched = None
        if not (signal == "cgm" and cgm_points is not None):
            matched = signal_value_at_index(rows, signal, index_lookup.get(point_name))
        if not matched:
            matched = nearest_signal_value_at_timestamp(source_rows, signal, timestamp, parse_timestamp)
        raw_value = matched.get("value") if isinstance(matched, dict) else None
        out[point_name] = {
            "label_timestamp": timestamp,
            "matched_timestamp": matched.get("timestamp") if isinstance(matched, dict) else "",
            "matched_index": matched.get("index") if isinstance(matched, dict) else None,
            "time_offset_seconds": matched.get("time_offset_seconds") if isinstance(matched, dict) else None,
            "signal": signal,
            "value_source": signal_value_column(signal),
            "raw_delta_value": raw_value if signal in ("seg", "sel") else None,
            "cgm_value": raw_value if signal == "cgm" else None,
        }
        if signal == "seg":
            out[point_name]["converted_concentration"] = concentration_from_delta(raw_value)
            out[point_name]["conversion_formula"] = "conc = (delta - 0.0188) / 0.00185"
            out[point_name]["SEG_converted_concentration_uM"] = converted_concentration_um(signal, raw_value)
            out[point_name]["SEG_uM_conversion_formula"] = "SEG_converted_concentration_uM = (raw_delta_value - 0.00453) / 0.00128"
        elif signal == "sel":
            out[point_name]["SEL_converted_concentration_uM"] = converted_concentration_um(signal, raw_value)
            out[point_name]["SEL_uM_conversion_formula"] = "SEL_converted_concentration_uM = (raw_delta_value - 0.00638) / 2.9145e-4"
    return out


def trough_to_trough_auc_features_for_peak(rows, peak_meta, parse_timestamp, concentration_from_delta, cgm_points=None):
    if not isinstance(peak_meta, dict):
        return {}
    signal = str(peak_meta.get("signal") or "").strip().lower()
    index_lookup = aggregate_point_indices(peak_meta.get("aggregate_id"))
    start_idx = index_lookup.get("trough1")
    end_idx = index_lookup.get("trough2")
    points = []
    if signal == "cgm" and cgm_points is not None:
        points = signal_points_for_interval(
            rows,
            signal,
            peak_meta.get("trough1_timestamp", ""),
            peak_meta.get("trough2_timestamp", ""),
            parse_timestamp,
            cgm_points=cgm_points,
        )
    elif start_idx is not None and end_idx is not None:
        lo = max(0, min(int(start_idx), int(end_idx)))
        hi = min(len(rows or []) - 1, max(int(start_idx), int(end_idx)))
        for idx in range(lo, hi + 1):
            row = rows[idx]
            dt = parse_timestamp(row.get("Timestamp", ""))
            value = to_finite_float(row.get(signal_value_column(signal)))
            if dt is None or value is None:
                continue
            points.append((dt, value))
    else:
        points = signal_points_for_interval(
            rows,
            signal,
            peak_meta.get("trough1_timestamp", ""),
            peak_meta.get("trough2_timestamp", ""),
            parse_timestamp,
        )
    points.sort(key=lambda item: item[0])
    auc_seconds = None
    duration_seconds = None
    if len(points) >= 2:
        duration_seconds = max(0.0, (points[-1][0] - points[0][0]).total_seconds())
        auc_seconds = 0.0
        for idx in range(1, len(points)):
            dt_seconds = (points[idx][0] - points[idx - 1][0]).total_seconds()
            if dt_seconds <= 0:
                continue
            auc_seconds += ((points[idx - 1][1] + points[idx][1]) / 2.0) * dt_seconds
    out = {
        "start_timestamp": points[0][0].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if points else peak_meta.get("trough1_timestamp", ""),
        "end_timestamp": points[-1][0].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if points else peak_meta.get("trough2_timestamp", ""),
        "signal": signal,
        "value_source": signal_value_column(signal),
        "method": "trapezoidal_auc_over_timestamp_seconds",
        "point_count": len(points),
        "duration_seconds": duration_seconds,
        "raw_delta_auc_seconds": auc_seconds if signal in ("seg", "sel") else None,
        "cgm_auc_seconds": auc_seconds if signal == "cgm" else None,
    }
    if signal == "seg":
        converted_auc = None
        if len(points) >= 2:
            converted_points = [(dt, concentration_from_delta(value)) for dt, value in points]
            converted_points = [(dt, value) for dt, value in converted_points if value is not None]
            if len(converted_points) >= 2:
                converted_auc = 0.0
                for idx in range(1, len(converted_points)):
                    dt_seconds = (converted_points[idx][0] - converted_points[idx - 1][0]).total_seconds()
                    if dt_seconds <= 0:
                        continue
                    converted_auc += ((converted_points[idx - 1][1] + converted_points[idx][1]) / 2.0) * dt_seconds
        out["converted_concentration_auc_seconds"] = converted_auc
        out["conversion_formula"] = "conc = (delta - 0.0188) / 0.00185"
    if signal in ("seg", "sel"):
        converted_um_auc = None
        if len(points) >= 2:
            converted_um_points = [(dt, converted_concentration_um(signal, value)) for dt, value in points]
            converted_um_points = [(dt, value) for dt, value in converted_um_points if value is not None]
            if len(converted_um_points) >= 2:
                converted_um_auc = 0.0
                for idx in range(1, len(converted_um_points)):
                    dt_seconds = (converted_um_points[idx][0] - converted_um_points[idx - 1][0]).total_seconds()
                    if dt_seconds <= 0:
                        continue
                    converted_um_auc += ((converted_um_points[idx - 1][1] + converted_um_points[idx][1]) / 2.0) * dt_seconds
        if signal == "seg":
            out["SEG_converted_concentration_uM_auc_seconds"] = converted_um_auc
            out["SEG_uM_conversion_formula"] = "SEG_converted_concentration_uM = (raw_delta_value - 0.00453) / 0.00128"
        else:
            out["SEL_converted_concentration_uM_auc_seconds"] = converted_um_auc
            out["SEL_uM_conversion_formula"] = "SEL_converted_concentration_uM = (raw_delta_value - 0.00638) / 2.9145e-4"
    return out


def interval_derivative_features_for_peak(rows, peak_meta, parse_timestamp, cgm_points=None):
    if not isinstance(peak_meta, dict):
        return {}
    signal = peak_meta.get("signal")
    trough1_ts = peak_meta.get("trough1_timestamp", "")
    peak_ts = peak_meta.get("peak_timestamp", "")
    trough2_ts = peak_meta.get("trough2_timestamp", "")
    preceding = peak_meta.get("preceding_instance")
    meal_ts = preceding.get("instance_timestamp", "") if isinstance(preceding, dict) else ""
    intervals = {
        "trough1_to_peak": (trough1_ts, peak_ts),
        "meal_to_peak": (meal_ts, peak_ts),
        "peak_to_trough2": (peak_ts, trough2_ts),
    }
    out = {}
    for name, (start_ts, end_ts) in intervals.items():
        points = signal_points_for_interval(rows, signal, start_ts, end_ts, parse_timestamp, cgm_points=cgm_points)
        summary = normalized_interval_derivative_summary(points, start_ts, end_ts, parse_timestamp)
        summary["start_timestamp"] = start_ts
        summary["end_timestamp"] = end_ts
        summary["signal"] = signal
        summary["value_source"] = signal_value_column(signal)
        summary["time_normalization"] = "interval_duration_scaled_to_0_1"
        out[name] = summary
    return out


def timestamp_duration_seconds(start_ts, end_ts, parse_timestamp=default_parse_timestamp):
    start_dt = parse_timestamp(start_ts)
    end_dt = parse_timestamp(end_ts)
    if start_dt is None or end_dt is None:
        return None
    return (end_dt - start_dt).total_seconds()


def augment_peak_metadata_with_durations(peak_meta, parse_timestamp=default_parse_timestamp):
    if not isinstance(peak_meta, dict):
        return peak_meta
    trough1_ts = peak_meta.get("trough1_timestamp", "")
    peak_ts = peak_meta.get("peak_timestamp", "")
    trough2_ts = peak_meta.get("trough2_timestamp", "")
    intervals = {
        "duration_trough1_to_peak": (trough1_ts, peak_ts),
        "duration_peak_to_trough2": (peak_ts, trough2_ts),
        "duration_trough1_to_trough2": (trough1_ts, trough2_ts),
    }
    for prefix, (start_ts, end_ts) in intervals.items():
        seconds = timestamp_duration_seconds(start_ts, end_ts, parse_timestamp=parse_timestamp)
        peak_meta[f"{prefix}_seconds"] = None if seconds is None else max(0.0, seconds)
    return peak_meta


def augment_peak_metadata_with_preceding_peak_count(peak_meta, aggregates, parse_timestamp=default_parse_timestamp):
    if not isinstance(peak_meta, dict):
        return peak_meta
    preceding = peak_meta.get("preceding_instance")
    instance_ts = preceding.get("instance_timestamp", "") if isinstance(preceding, dict) else ""
    instance_dt = parse_timestamp(instance_ts)
    peak_dt = parse_timestamp(peak_meta.get("peak_timestamp", ""))
    signal = str(peak_meta.get("signal") or "").strip().lower()
    count = None
    if instance_dt is not None and peak_dt is not None and signal:
        count = 0
        for agg in aggregates or []:
            if not isinstance(agg, dict):
                continue
            if str(agg.get("signal", "")).strip().lower() != signal:
                continue
            other_peak_dt = parse_timestamp(agg.get("peak", {}).get("timestamp", ""))
            if other_peak_dt is None:
                continue
            if instance_dt < other_peak_dt < peak_dt:
                count += 1
    peak_meta["num_preceding_peaks_since_instance"] = count
    peak_meta["num_presceding_peak"] = count
    return peak_meta


def augment_peak_metadata_with_temporal_features(
    rows,
    peak_meta,
    parse_timestamp=default_parse_timestamp,
    concentration_from_delta=default_concentration_from_delta,
    cgm_points=None,
    aggregates=None,
):
    """Backfill all supported temporal features for one peak metadata object."""
    if not isinstance(peak_meta, dict):
        return peak_meta
    augment_peak_metadata_with_durations(peak_meta, parse_timestamp=parse_timestamp)
    augment_peak_metadata_with_preceding_peak_count(
        peak_meta,
        aggregates,
        parse_timestamp=parse_timestamp,
    )
    peak_meta["point_values"] = point_value_features_for_peak(
        rows,
        peak_meta,
        parse_timestamp,
        concentration_from_delta,
        cgm_points=cgm_points,
    )
    peak_meta["trough1_to_trough2_auc"] = trough_to_trough_auc_features_for_peak(
        rows,
        peak_meta,
        parse_timestamp,
        concentration_from_delta,
        cgm_points=cgm_points,
    )
    peak_meta["interval_derivative_features"] = interval_derivative_features_for_peak(
        rows,
        peak_meta,
        parse_timestamp,
        cgm_points=cgm_points,
    )
    peak_meta["feature_engineering_version"] = FEATURE_ENGINEERING_VERSION
    return peak_meta


def _case_feature_lists(payload):
    """Yield ``(case_path, feature_list)`` pairs from supported annotation shapes."""
    if not isinstance(payload, dict):
        return
    cases = payload.get("peak_pair_features", {}).get("cases", {})
    if isinstance(cases, dict):
        for rel_path, case_state in cases.items():
            if isinstance(case_state, dict):
                features = case_state.get("paired_features")
                if isinstance(features, list):
                    yield rel_path, features
    cases = payload.get("paired_features", {}).get("cases", {})
    if isinstance(cases, dict):
        for rel_path, case_state in cases.items():
            if isinstance(case_state, dict):
                features = case_state.get("paired_features")
                if isinstance(features, list):
                    yield rel_path, features
    saved_segments = payload.get("saved_segments", {})
    if isinstance(saved_segments, dict):
        for rel_path, case_state in saved_segments.items():
            if isinstance(case_state, dict):
                features = case_state.get("peak_pairs")
                if isinstance(features, list):
                    yield rel_path, features


def _case_aggregates(payload, rel_path):
    saved_segments = payload.get("saved_segments", {}) if isinstance(payload, dict) else {}
    case_state = saved_segments.get(rel_path, {}) if isinstance(saved_segments, dict) else {}
    aggregates = []
    if isinstance(case_state, dict):
        for key in ("peak_aggregates", "aggregates", "label_aggregates"):
            value = case_state.get(key)
            if isinstance(value, list):
                aggregates.extend(value)
    return aggregates


def remove_pair_engineered_fields(feature_record):
    """Remove obsolete pair-level engineered timing fields from one record."""
    if not isinstance(feature_record, dict):
        return 0
    removed = 0
    exact_keys = {
        "peak_time_difference_seconds",
        "abs_peak_time_difference_seconds",
        "feature_engineering_version",
    }
    for key in list(feature_record.keys()):
        key_l = str(key).lower()
        is_pair_timing = (
            key in exact_keys
            or key_l.startswith("cgm_seg_peak_time_difference")
            or key_l.startswith("abs_cgm_seg_peak_time_difference")
            or key_l.startswith("cgm_sel_peak_time_difference")
            or key_l.startswith("abs_cgm_sel_peak_time_difference")
            or key_l.startswith("seg_sel_peak_time_difference")
            or key_l.startswith("abs_seg_sel_peak_time_difference")
            or key_l.startswith("delay_")
        )
        if is_pair_timing:
            feature_record.pop(key, None)
            removed += 1
    return removed


def process_annotation_payload(
    payload,
    parse_timestamp=default_parse_timestamp,
    concentration_from_delta=default_concentration_from_delta,
    row_loader=load_processed_series,
    cgm_series_loader=independent_cgm_series_for_case,
    load_independent_cgm=True,
    case_paths=None,
    in_place=False,
):
    """Backfill temporal features into an alignment annotation payload.

    The returned payload keeps the same data structure as the input.  No missing
    labels are imputed; only existing uni-modal peak metadata entries are updated.
    Pair-level timing/delay features are intentionally not calculated here, and
    obsolete pair-level engineered fields are removed from processed records.
    """
    out = payload if in_place else copy.deepcopy(payload)
    summary = {
        "cases_seen": 0,
        "feature_records_seen": 0,
        "peak_metadata_updated": 0,
        "pair_engineered_fields_removed": 0,
        "case_errors": [],
        "feature_engineering_version": FEATURE_ENGINEERING_VERSION,
    }
    if not isinstance(out, dict):
        return out, {**summary, "error": "payload is not a JSON object"}

    rows_cache = {}
    cgm_cache = {}
    seen_cases = set()
    case_path_filter = set(case_paths or []) if case_paths else None
    for rel_path, feature_list in _case_feature_lists(out):
        if case_path_filter is not None and rel_path not in case_path_filter:
            continue
        summary["feature_records_seen"] += len(feature_list)
        if rel_path not in seen_cases:
            seen_cases.add(rel_path)
            summary["cases_seen"] += 1
        try:
            if rel_path not in rows_cache:
                rows_cache[rel_path] = row_loader(rel_path)
            rows = rows_cache[rel_path]
            if rel_path not in cgm_cache:
                if load_independent_cgm:
                    try:
                        cgm_cache[rel_path] = cgm_series_loader(
                            rel_path,
                            processed_rows=rows,
                            parse_timestamp=parse_timestamp,
                        )
                    except TypeError:
                        cgm_cache[rel_path] = cgm_series_loader(rel_path, processed_rows=rows)
                else:
                    cgm_cache[rel_path] = None
            cgm_points = cgm_cache[rel_path]
            aggregates = _case_aggregates(out, rel_path)
            for pair in feature_list:
                if not isinstance(pair, dict):
                    continue
                summary["pair_engineered_fields_removed"] += remove_pair_engineered_fields(pair)
                selected = pair.get("selected_peak_metadata")
                if not isinstance(selected, dict):
                    continue
                for signal, peak_meta in selected.items():
                    if not isinstance(peak_meta, dict):
                        continue
                    signal_cgm_points = cgm_points if str(signal).lower() == "cgm" else None
                    augment_peak_metadata_with_temporal_features(
                        rows,
                        peak_meta,
                        parse_timestamp=parse_timestamp,
                        concentration_from_delta=concentration_from_delta,
                        cgm_points=signal_cgm_points,
                        aggregates=aggregates,
                    )
                    summary["peak_metadata_updated"] += 1
        except Exception as exc:
            summary["case_errors"].append({"path": rel_path, "error": str(exc)})

    meta = out.setdefault("meta", {})
    if isinstance(meta, dict):
        meta["feature_engineering"] = {
            "version": FEATURE_ENGINEERING_VERSION,
            "processed_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "summary": summary,
            "features": calculated_feature_manifest(),
        }
    return out, summary


def export_feature_engineered_annotations(
    input_json_path,
    output_json_path,
    parse_timestamp=default_parse_timestamp,
    concentration_from_delta=default_concentration_from_delta,
    row_loader=load_processed_series,
    cgm_series_loader=independent_cgm_series_for_case,
    load_independent_cgm=True,
    case_paths=None,
    indent=2,
):
    """Read annotations, backfill features, and export dashboard-compatible JSON."""
    with open(input_json_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    processed, summary = process_annotation_payload(
        payload,
        parse_timestamp=parse_timestamp,
        concentration_from_delta=concentration_from_delta,
        row_loader=row_loader,
        cgm_series_loader=cgm_series_loader,
        load_independent_cgm=load_independent_cgm,
        case_paths=case_paths,
        in_place=False,
    )
    output_dir = os.path.dirname(os.path.abspath(output_json_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as handle:
        json.dump(processed, handle, indent=indent)
        handle.write("\n")
    return summary


def _main():
    parser = argparse.ArgumentParser(
        description="Backfill CGM/SEG/SEL temporal features into an alignment annotation JSON file."
    )
    parser.add_argument("input_json", help="Input alignment annotation JSON.")
    parser.add_argument("output_json", help="Output JSON path preserving the dashboard annotation structure.")
    parser.add_argument(
        "--case-path",
        action="append",
        default=None,
        help="Optional case path to process. Can be repeated. Defaults to all cases.",
    )
    parser.add_argument(
        "--fast-cgm",
        action="store_true",
        help="Use CGM values already present in processed rows instead of loading independent CGM series.",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation level.")
    args = parser.parse_args()
    summary = export_feature_engineered_annotations(
        args.input_json,
        args.output_json,
        load_independent_cgm=not args.fast_cgm,
        case_paths=args.case_path,
        indent=args.indent,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
