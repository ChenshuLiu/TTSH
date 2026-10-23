#!/usr/bin/env python3

import csv
import json
import math
import os
import posixpath
import re
import signal
import sys
import threading
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from statistics import median
from urllib.parse import parse_qs, urlparse

from data_paths import ROOT_DIR as DATA_ROOT, ALIGNMENT_DIR as DATA_ALIGNMENT_DIR, STATE_DIR as DATA_STATE_DIR

ROOT_DIR = str(DATA_ROOT)

# Shared data loaders and processing helpers merged from toolkit_server.py.
DATA_DIR = os.path.join(ROOT_DIR, 'raw_data', 'sensor')

CGM_DIR = os.path.join(ROOT_DIR, 'raw_data', 'cgm')

TOOL2_OUTPUT_DIR = os.path.join(ROOT_DIR, 'annotated_data', 'series')

def normalize_header(name):
    return ''.join((name or '').strip().lower().split())

def safe_rel_path(rel_path):
    normalized = posixpath.normpath((rel_path or '').replace('\\', '/'))
    if normalized in ('', '.'):
        raise ValueError('Missing relative path.')
    if normalized.startswith('../') or normalized == '..':
        raise ValueError('Invalid relative path.')
    return normalized

def abs_from_rel(rel_path):
    rel_path = safe_rel_path(rel_path)
    abs_path = os.path.abspath(os.path.join(ROOT_DIR, rel_path))
    if abs_path != ROOT_DIR and (not abs_path.startswith(ROOT_DIR + os.sep)):
        raise ValueError('Path must stay inside the project root.')
    return abs_path

def assert_under(abs_path, root_path, message):
    root_path = os.path.abspath(root_path)
    abs_path = os.path.abspath(abs_path)
    if abs_path != root_path and (not abs_path.startswith(root_path + os.sep)):
        raise ValueError(message)
    return abs_path

def parse_timestamp_info(value):
    if isinstance(value, (int, float)):
        return (excel_serial_to_datetime(value), 'subsecond')
    value = str(value).strip()
    if value:
        try:
            numeric_value = float(value)
            if 20000 < numeric_value < 80000:
                return (excel_serial_to_datetime(numeric_value), 'subsecond')
        except ValueError:
            pass
    for fmt, precision in (('%Y-%m-%d %H:%M:%S.%f', 'subsecond'), ('%Y-%m-%d %H:%M:%S', 'second'), ('%m/%d/%y %H:%M:%S', 'second'), ('%m/%d/%y %H:%M', 'minute'), ('%m/%d/%y %I:%M:%S', 'second'), ('%m/%d/%y %I:%M:%S %p', 'second'), ('%m/%d/%y %I:%M %p', 'minute')):
        try:
            return (datetime.strptime(value, fmt), precision)
        except ValueError:
            continue
    raise ValueError(f'Unsupported timestamp format: {value}')

def parse_cgm_timestamp(value):
    value = str(value).strip()
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%m/%d/%y %H:%M:%S', '%m/%d/%y %H:%M', '%m/%d/%y %I:%M:%S %p', '%m/%d/%y %I:%M %p'):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f'Unsupported CGM timestamp format: {value}')

def excel_serial_to_datetime(value):
    return datetime(1899, 12, 30) + timedelta(days=float(value))

def format_duration(total_seconds):
    total = max(0, int(round(total_seconds)))
    hours = total // 3600
    minutes = total % 3600 // 60
    seconds = total % 60
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

def moving_average(values, ratio):
    if not values:
        return []
    window = max(3, int(math.ceil(len(values) * max(0.0, ratio))))
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = []
    for idx in range(len(values)):
        left = max(0, idx - half)
        right = min(len(values), idx + half + 1)
        out.append(sum(values[left:right]) / max(1, right - left))
    return out

def toolkit_minmax(values):
    if not values:
        return []
    low = min(values)
    high = max(values)
    if abs(high - low) < 1e-12:
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]

def participant_id_from_rel_path(rel_path):
    parts = safe_rel_path(rel_path).split('/')
    for part in parts:
        prefix = part[:1].upper()
        if prefix in ('P', 'H'):
            digits = ''.join((ch for ch in part if ch.isdigit()))
            if digits:
                return f'{prefix}{int(digits):03d}'
    raise ValueError(f'Could not determine participant from {rel_path}')

def list_csv_files_under(root_dir):
    out = []
    if not os.path.exists(root_dir):
        return out
    for root, _, files in os.walk(root_dir):
        for name in files:
            if name.lower().endswith('.csv'):
                abs_path = os.path.join(root, name)
                out.append(os.path.relpath(abs_path, ROOT_DIR).replace(os.sep, '/'))
    return sorted(out)

def metadata_rel_for_csv(csv_rel_path):
    return safe_rel_path(csv_rel_path) + '.metadata.json'

def load_cgm_lookup(rel_path, start_dt, end_dt):
    participant = participant_id_from_rel_path(rel_path)
    cgm_path = os.path.join(CGM_DIR, f'{participant}_cleaned.csv')
    if not os.path.exists(cgm_path):
        return {'second': {}, 'minute': {}}
    second_lookup = {}
    minute_lookup = {}
    with open(cgm_path, newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_ts = (row.get('timestamp') or '').strip()
            raw_value = (row.get('glucose_mmol_L') or '').strip()
            if not raw_ts or not raw_value:
                continue
            dt = parse_cgm_timestamp(raw_ts)
            if dt < start_dt or dt > end_dt:
                continue
            value_lower = raw_value.lower()
            if value_lower == 'high':
                value = 27.8
            elif value_lower == 'low':
                value = 3.0
            else:
                value = float(raw_value)
            second_key = dt.strftime('%Y-%m-%d %H:%M:%S')
            minute_key = dt.strftime('%Y-%m-%d %H:%M')
            second_lookup[second_key] = value
            minute_lookup[minute_key] = value
    return {'second': second_lookup, 'minute': minute_lookup}

def load_cgm_points_in_range(rel_path, start_dt, end_dt):
    participant = participant_id_from_rel_path(rel_path)
    cgm_path = os.path.join(CGM_DIR, f'{participant}_cleaned.csv')
    if not os.path.exists(cgm_path):
        return []
    out = []
    with open(cgm_path, newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_ts = (row.get('timestamp') or '').strip()
            raw_value = (row.get('glucose_mmol_L') or '').strip()
            if not raw_ts or not raw_value:
                continue
            dt = parse_cgm_timestamp(raw_ts)
            if dt < start_dt or dt > end_dt:
                continue
            value_lower = raw_value.lower()
            if value_lower == 'high':
                value = 27.8
            elif value_lower == 'low':
                value = 3.0
            else:
                value = float(raw_value)
            out.append({'timestamp': dt.strftime('%Y-%m-%d %H:%M:%S'), 'value': value})
    return out

def xlsx_column_index(cell_ref):
    letters = ''.join((ch for ch in cell_ref if ch.isalpha()))
    index = 0
    for ch in letters:
        index = index * 26 + (ord(ch.upper()) - ord('A') + 1)
    return index - 1

def xlsx_cell_value(cell, shared_strings, namespace):
    cell_type = cell.attrib.get('t')
    if cell_type == 'inlineStr':
        return ''.join((text.text or '' for text in cell.findall('.//a:t', namespace)))
    value_node = cell.find('a:v', namespace)
    if value_node is None:
        return ''
    value = value_node.text or ''
    if cell_type == 's' and value:
        return shared_strings[int(value)]
    return value

def read_xlsx_dict_rows(abs_path):
    namespace = {'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(abs_path) as workbook:
        names = workbook.namelist()
        shared_strings = []
        if 'xl/sharedStrings.xml' in names:
            shared_root = ET.fromstring(workbook.read('xl/sharedStrings.xml'))
            for item in shared_root.findall('a:si', namespace):
                shared_strings.append(''.join((text.text or '' for text in item.findall('.//a:t', namespace))))
        sheet_names = [name for name in names if name.startswith('xl/worksheets/sheet') and name.endswith('.xml')]
        if not sheet_names:
            return []
        sheet_root = ET.fromstring(workbook.read(sorted(sheet_names)[0]))
        table_rows = []
        for row in sheet_root.findall('.//a:sheetData/a:row', namespace):
            values = {}
            max_index = -1
            for cell in row.findall('a:c', namespace):
                ref = cell.attrib.get('r', '')
                index = xlsx_column_index(ref)
                values[index] = xlsx_cell_value(cell, shared_strings, namespace)
                max_index = max(max_index, index)
            if max_index < 0:
                continue
            table_rows.append([values.get(idx, '') for idx in range(max_index + 1)])
    if not table_rows:
        return []
    headers = [str(value).strip() for value in table_rows[0]]
    if headers and (not headers[0]):
        headers[0] = 'Timestamp'
    rows = []
    for values in table_rows[1:]:
        rows.append({headers[idx] if idx < len(headers) and headers[idx] else f'Column{idx + 1}': values[idx] for idx in range(len(values))})
    return rows

def raw_dict_rows_from_file(rel_path):
    rel_path = safe_rel_path(rel_path)
    abs_path = abs_from_rel(rel_path)
    field_root = DATA_DIR if rel_path.startswith('raw_data/sensor/') else ROOT_DIR
    assert_under(abs_path, field_root, 'Path is outside the expected root.')
    if rel_path.lower().endswith('.xlsx'):
        return read_xlsx_dict_rows(abs_path)
    with open(abs_path, newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))

def standardized_rows_from_csv(rel_path):
    rows = []
    raw_rows = raw_dict_rows_from_file(rel_path)
    fieldnames = list(raw_rows[0].keys()) if raw_rows else []
    normalized = {normalize_header(name): name for name in fieldnames}
    ts_key = normalized.get('timestamp')
    ch1_key = normalized.get('channel1')
    ch2_key = normalized.get('channel2')
    ch3_key = normalized.get('channel3')
    d21_key = normalized.get('deltachannel2minuschannel1')
    d23_key = normalized.get('deltachannel2minuschannel3')
    nd21_key = normalized.get('normalizeddeltachannel2minuschannel1')
    nd23_key = normalized.get('normalizeddeltachannel2minuschannel3')
    ncgm_key = normalized.get('normalizedcgm')
    cgm_key = normalized.get('cgm')
    if not (ts_key and ch1_key and ch2_key and ch3_key):
        raise ValueError(f'Missing required columns in {rel_path}')
    for row in raw_rows:
        raw_ts = str(row.get(ts_key) or '').strip()
        raw_ch1 = str(row.get(ch1_key) or '').strip()
        raw_ch2 = str(row.get(ch2_key) or '').strip()
        raw_ch3 = str(row.get(ch3_key) or '').strip()
        if not raw_ts or not raw_ch1 or (not raw_ch2) or (not raw_ch3):
            continue
        dt, precision = parse_timestamp_info(raw_ts)
        channel1 = float(raw_ch1)
        channel2 = float(raw_ch2)
        channel3 = float(raw_ch3)
        delta21 = float((row.get(d21_key) or '').strip()) if d21_key and (row.get(d21_key) or '').strip() else channel2 - channel1
        delta23 = float((row.get(d23_key) or '').strip()) if d23_key and (row.get(d23_key) or '').strip() else channel2 - channel3
        cgm_raw = str(row.get(cgm_key) or '').strip() if cgm_key else ''
        cgm_value = None if not cgm_raw or cgm_raw.upper() == 'NA' else float(cgm_raw)
        normalized_row = {'Timestamp': dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], 'dt': dt, 'timestamp_precision': precision, 'Channel1': channel1, 'Channel2': channel2, 'Channel3': channel3, 'Delta_Channel2_minus_Channel1': delta21, 'Delta_Channel2_minus_Channel3': delta23, 'CGM': cgm_value}
        if nd21_key and (row.get(nd21_key) or '').strip():
            normalized_row['Normalized_Delta_Channel2_minus_Channel1'] = float((row.get(nd21_key) or '').strip())
        if nd23_key and (row.get(nd23_key) or '').strip():
            normalized_row['Normalized_Delta_Channel2_minus_Channel3'] = float((row.get(nd23_key) or '').strip())
        if ncgm_key and (row.get(ncgm_key) or '').strip():
            ncgm_raw = (row.get(ncgm_key) or '').strip()
            normalized_row['Normalized_CGM'] = None if ncgm_raw.upper() == 'NA' else float(ncgm_raw)
        rows.append(normalized_row)
    return rows

def attach_exact_cgm(rows, rel_path):
    if not rows:
        return rows
    lookup = load_cgm_lookup(rel_path, rows[0]['dt'], rows[-1]['dt'])
    out = []
    for row in rows:
        copied = dict(row)
        second_key = row['Timestamp'][:19]
        minute_key = row['Timestamp'][:16]
        if copied.get('timestamp_precision') == 'minute':
            copied['CGM'] = lookup['minute'].get(minute_key)
        else:
            copied['CGM'] = lookup['second'].get(second_key)
        out.append(copied)
    return out

def apply_normalized_columns(rows):
    out = [dict(row) for row in rows]
    d21 = toolkit_minmax([row['Delta_Channel2_minus_Channel1'] for row in out])
    d23 = toolkit_minmax([row['Delta_Channel2_minus_Channel3'] for row in out])
    cgm_values = [row['CGM'] for row in out if row.get('CGM') is not None]
    cgm_scaled = toolkit_minmax(cgm_values) if cgm_values else []
    cgm_iter = iter(cgm_scaled)
    for idx, row in enumerate(out):
        row['Normalized_Delta_Channel2_minus_Channel1'] = d21[idx] if d21 else None
        row['Normalized_Delta_Channel2_minus_Channel3'] = d23[idx] if d23 else None
        if row.get('CGM') is None:
            row['Normalized_CGM'] = None
        else:
            row['Normalized_CGM'] = next(cgm_iter)
    return out

def load_tool1_series(rel_path):
    rows = attach_exact_cgm(standardized_rows_from_csv(rel_path), rel_path)
    return apply_normalized_columns(rows)

def load_tool2_series(rel_path):
    rows = attach_exact_cgm(standardized_rows_from_csv(rel_path), rel_path)
    return apply_normalized_columns(rows)

def load_processed_series(rel_path):
    rows = standardized_rows_from_csv(rel_path)
    if rows and all((row.get('CGM') is None for row in rows)):
        metadata = load_metadata_json(metadata_rel_for_csv(rel_path))
        if metadata and metadata.get('tool') == 'tool2' and metadata.get('source_path'):
            source_path = metadata['source_path']
            start_index = int(metadata.get('start_index', 0))
            end_index = int(metadata.get('end_index', max(0, len(rows) - 1)))
            return tool2_process_rows(load_tool2_series(source_path), source_path, start_index, end_index)
    if all(('Normalized_Delta_Channel2_minus_Channel1' in row for row in rows)):
        return rows
    return apply_normalized_columns(rows)

def load_metadata_json(metadata_rel_path):
    metadata_abs_path = abs_from_rel(metadata_rel_path)
    if not os.path.exists(metadata_abs_path):
        return None
    with open(metadata_abs_path, encoding='utf-8') as handle:
        return json.load(handle)

def tool2_process_rows(raw_rows, rel_path, start_index, end_index):
    if not raw_rows:
        return []
    start_index = max(0, min(start_index, len(raw_rows) - 1))
    end_index = max(start_index, min(end_index, len(raw_rows) - 1))
    base21 = raw_rows[start_index]['Channel2'] - raw_rows[start_index]['Channel1']
    base23 = raw_rows[start_index]['Channel2'] - raw_rows[start_index]['Channel3']
    sliced = raw_rows[start_index:end_index + 1]
    processed = []
    for row in sliced:
        processed.append({'Timestamp': row['Timestamp'], 'dt': row['dt'], 'timestamp_precision': row.get('timestamp_precision'), 'Channel1': row['Channel1'], 'Channel2': row['Channel2'], 'Channel3': row['Channel3'], 'Delta_Channel2_minus_Channel1': row['Channel2'] - row['Channel1'] - base21 + 0.04, 'Delta_Channel2_minus_Channel3': row['Channel2'] - row['Channel3'] - base23})
    processed = attach_exact_cgm(processed, rel_path)
    return apply_normalized_columns(processed)


import cgm_feature_engineering as cgm_features
from server_instance import replacing_server

try:
    from analysis_results_dashboard import (
        NUMPY_AVAILABLE,
        SklearnTSNE,
        analysis_dashboard_script,
        fallback_tsne_embedding,
        html_page as analysis_results_page_html,
    )
except Exception:
    NUMPY_AVAILABLE = False
    SklearnTSNE = None
    analysis_dashboard_script = None
    fallback_tsne_embedding = None
    analysis_results_page_html = None

ALIGNMENT_DIR = str(DATA_ALIGNMENT_DIR)
RAW_ORIGINAL_DIR = DATA_DIR
SMOOTHED_DIR = os.path.join(ALIGNMENT_DIR, "smoothed_series")
LOCAL_PROCESSED_DIR = TOOL2_OUTPUT_DIR
CONVERTED_DIR = os.path.join(ALIGNMENT_DIR, "converted_time_series")
STATE_DIR = str(DATA_STATE_DIR)
SEGMENT_STATE_JSON = os.path.join(STATE_DIR, "segment_state.json")
PEAK_PAIR_JSON = os.path.join(STATE_DIR, "peak_pair_features.json")
ANNOTATION_STATE_JSON = os.path.join(STATE_DIR, "alignment_annotations.json")
DIET_INSTANCES_JSON = os.path.join(STATE_DIR, "diet_instances.json")
DIET_PHOTO_ANNOTATIONS_JSON = os.path.join(STATE_DIR, "diet_photo_annotations.json")
RANGE_LABELS_JSON = os.path.join(STATE_DIR, "range_label_selections.json")
RANGE_LABEL_HISTORY_JSON = os.path.join(STATE_DIR, "range_label_history.json")
PROJECTS_DIR = os.path.join(STATE_DIR, "projects")
ACTIVE_PROJECT_JSON = os.path.join(STATE_DIR, "active_project_session.json")
LAB_DATA_XLSX = os.path.join(ALIGNMENT_DIR, "Lab_data.xlsx")
HOST = "127.0.0.1"
PORT = 8768
DIET_TS_RE = re.compile(r"(\d{8}_\d{6})")
PATIENT_ID_RE = re.compile(r"\b([PH])(\d{1,3})\b", re.IGNORECASE)
ALLOWED_LABEL_OPTIONS = [
    "CGM peak",
    "CGM trough",
    "SEG peak",
    "SEG trough",
    "SEL peak",
    "SEL trough",
]
ALLOWED_LABEL_SET_LOWER = {label.lower(): label for label in ALLOWED_LABEL_OPTIONS}
LAB_FEATURE_CACHE = None
DIET_INSTANCES_CACHE = None
DIET_PHOTO_ANN_CACHE = None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, payload):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def ensure_unified_annotation_state():
    if os.path.exists(ANNOTATION_STATE_JSON):
        return
    seg = load_json(SEGMENT_STATE_JSON, {})
    if not isinstance(seg, dict):
        seg = {}
    pair = load_json(PEAK_PAIR_JSON, {"cases": {}})
    if not isinstance(pair, dict):
        pair = {"cases": {}}
    payload = {
        "meta": {"saved_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), "mode": "single_file"},
        "saved_segments": seg,
        "peak_pair_features": pair,
        "paired_features": pair,
    }
    ensure_dir(STATE_DIR)
    with open(ANNOTATION_STATE_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def list_project_files():
    ensure_dir(PROJECTS_DIR)
    out = []
    for name in os.listdir(PROJECTS_DIR):
        if not name.lower().endswith(".json"):
            continue
        abs_path = os.path.join(PROJECTS_DIR, name)
        if os.path.isfile(abs_path):
            out.append(name)
    return sorted(out)


def _sanitize_project_name(name):
    out = str(name or "").strip()
    if not out:
        out = "autosave_latest"
    out = out.replace("\\", "_").replace("/", "_")
    if not out.lower().endswith(".json"):
        out += ".json"
    return out


def load_active_project_session():
    payload = load_json(ACTIVE_PROJECT_JSON, {})
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("autosave_enabled", False)
    payload.setdefault("project_file", "")
    return payload


def save_active_project_session(payload):
    ensure_dir(STATE_DIR)
    with open(ACTIVE_PROJECT_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def set_active_project_file(project_file, autosave_enabled=True):
    session = load_active_project_session()
    session["project_file"] = _sanitize_project_name(project_file)
    session["autosave_enabled"] = bool(autosave_enabled)
    save_active_project_session(session)


def write_project_snapshot_file(project_file):
    ensure_dir(PROJECTS_DIR)
    name = _sanitize_project_name(project_file)
    payload = {
        "meta": {"name": name, "saved_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")},
        "saved_segments": load_segment_state(),
        "peak_pair_features": load_peak_pair_state(),
        "paired_features": load_peak_pair_state(),
    }
    with open(os.path.join(PROJECTS_DIR, name), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return name


def autosave_active_project_if_enabled():
    session = load_active_project_session()
    if not session.get("autosave_enabled"):
        return
    project_file = session.get("project_file", "")
    if not project_file:
        return
    write_project_snapshot_file(project_file)


def save_project_snapshot(project_name):
    name = _sanitize_project_name(project_name)
    return write_project_snapshot_file(name)


def build_project_payload(name_hint=""):
    pair = load_peak_pair_state()
    return {
        "meta": {"name": _sanitize_project_name(name_hint or "project"), "saved_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")},
        "saved_segments": load_segment_state(),
        "peak_pair_features": pair,
        "paired_features": pair,
    }


def build_current_analysis_payload():
    payload = load_json(ANNOTATION_STATE_JSON, {})
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("saved_segments", load_segment_state())
    peak = payload.get("peak_pair_features", payload.get("paired_features"))
    if not isinstance(peak, dict):
        peak = load_peak_pair_state()
    payload["peak_pair_features"] = peak
    payload["paired_features"] = peak
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    meta["source"] = "Current alignment annotations"
    meta["source_path"] = ANNOTATION_STATE_JSON
    payload["meta"] = meta
    return payload


def _canonical_json(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def project_dirty_status(project_file):
    name = _sanitize_project_name(project_file)
    abs_path = os.path.join(PROJECTS_DIR, name)
    if not os.path.exists(abs_path):
        return {"exists": False, "dirty": True}
    file_payload = json.load(open(abs_path, encoding="utf-8"))
    current_payload = build_project_payload(name)
    return {"exists": True, "dirty": _canonical_json(file_payload) != _canonical_json(current_payload)}


def load_project_snapshot(project_file):
    name = _sanitize_project_name(project_file)
    if not name:
        raise ValueError("project_file is required")
    abs_path = os.path.join(PROJECTS_DIR, name)
    if not os.path.exists(abs_path):
        raise ValueError("Project file not found.")
    payload = json.load(open(abs_path, encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Invalid project payload.")
    normalized = canonicalize_imported_project(payload)
    seg = normalized.get("saved_segments", {})
    pair = normalized.get("peak_pair_features", {"cases": {}})
    save_segment_state(seg)
    save_peak_pair_state(pair)
    return name


def create_new_project(project_name):
    name = str(project_name or "").strip()
    if not name:
        raise ValueError("project_name is required")
    save_segment_state({})
    save_peak_pair_state({"cases": {}})
    return save_project_snapshot(name)


def source_processed_files():
    return sorted(
        path
        for path in list_csv_files_under(LOCAL_PROCESSED_DIR)
        if path.lower().endswith(".csv") and not path.endswith(".metadata.json")
    )


def smoothed_rel_path(source_rel_path):
    prefix = os.path.relpath(TOOL2_OUTPUT_DIR, ROOT_DIR).replace(os.sep, "/") + "/"
    rel = safe_rel_path(source_rel_path)
    if not rel.startswith(prefix):
        raise ValueError("Expected a Tool 2 processed file path.")
    suffix = rel[len(prefix) :]
    return os.path.relpath(os.path.join(SMOOTHED_DIR, suffix), ROOT_DIR).replace(os.sep, "/")


def list_processed_files():
    return sorted(
        path
        for path in list_csv_files_under(LOCAL_PROCESSED_DIR)
        if path.lower().endswith(".csv") and not path.endswith(".metadata.json")
    )


def remap_case_path_to_current(path, known_paths):
    rel = safe_rel_path(path)
    if rel in known_paths:
        return rel
    marker = "annotated_data/series/"
    if marker in rel:
        suffix = rel.split(marker, 1)[1]
        for kp in known_paths:
            if marker in kp and kp.split(marker, 1)[1] == suffix:
                return kp
    # Fallback: match by trailing file path when imported roots differ.
    rel_norm = rel.replace("\\", "/")
    for kp in known_paths:
        kp_norm = kp.replace("\\", "/")
        if kp_norm.endswith(rel_norm):
            return kp
    # Fallback: match unique filename.
    base = os.path.basename(rel_norm)
    if base:
        matches = [kp for kp in known_paths if os.path.basename(kp.replace("\\", "/")) == base]
        if len(matches) == 1:
            return matches[0]
    return rel


def remap_state_dict_paths(payload):
    if not isinstance(payload, dict):
        return {}
    known = list_processed_files()
    out = {}
    for k, v in payload.items():
        if not isinstance(k, str):
            continue
        try:
            rk = remap_case_path_to_current(k, known)
        except Exception:
            continue
        out[rk] = v
    return out


def has_usable_segment_entries(seg):
    if not isinstance(seg, dict) or not seg:
        return False
    for _k, v in seg.items():
        if not isinstance(v, dict):
            continue
        if any(
            key in v
            for key in (
                "labels_glu_cgm",
                "labels_glu_lac",
                "region_pairs_glu_cgm",
                "region_pairs_glu_lac",
                "included_glu_cgm",
                "included_glu_lac",
            )
        ):
            return True
    return False


def has_usable_peak_pair_entries(pair):
    if not isinstance(pair, dict):
        return False
    cases = pair.get("cases", {})
    if not isinstance(cases, dict) or not cases:
        return False
    for _k, v in cases.items():
        if not isinstance(v, dict):
            continue
        if isinstance(v.get("paired_features"), list) and v.get("paired_features"):
            return True
        if isinstance(v.get("matched_peak_sets"), list) and v.get("matched_peak_sets"):
            return True
    return False


def count_segment_annotation_items(seg):
    if not isinstance(seg, dict):
        return 0
    total = 0
    for _k, v in seg.items():
        if not isinstance(v, dict):
            continue
        total += len(v.get("labels_glu_cgm") or [])
        total += len(v.get("labels_glu_lac") or [])
        total += len(v.get("region_pairs_glu_cgm") or [])
        total += len(v.get("region_pairs_glu_lac") or [])
    return total


def count_peak_pair_items(pair):
    if not isinstance(pair, dict):
        return 0
    cases = pair.get("cases", {})
    if not isinstance(cases, dict):
        return 0
    total = 0
    for _k, v in cases.items():
        if not isinstance(v, dict):
            continue
        total += len(v.get("paired_features") or [])
        total += len(v.get("matched_peak_sets") or [])
    return total


def canonicalize_imported_project(imported):
    if not isinstance(imported, dict):
        raise ValueError("imported_project must be a JSON object")

    # Segment-state formats:
    # 1) Wrapped: {"saved_segments": {...}} or {"segment_state": {...}}
    # 2) Plain case map: {"<case_path>": {"included_glu_cgm": ...}, ...}
    seg = imported.get("saved_segments", imported.get("segment_state"))
    if seg is None:
        is_plain_state = any(
            isinstance(k, str) and isinstance(v, dict) and any(
                key in v
                for key in (
                    "labels_glu_cgm",
                    "labels_glu_lac",
                    "region_pairs_glu_cgm",
                    "region_pairs_glu_lac",
                    "included_glu_cgm",
                    "included_glu_lac",
                )
            )
            for k, v in imported.items()
        )
        seg = imported if is_plain_state else {}

    # Peak-pair formats:
    # 1) Wrapped: {"peak_pair_features": {"cases": {...}}}
    # 2) Backward alias: {"paired_features": {"cases": {...}}}
    # 3) Wrapped cases: {"cases": {...}}
    # 4) Plain case map: {"<case_path>": {"paired_features":[...]}, ...}
    pair = imported.get("peak_pair_features")
    if pair is None:
        pair = imported.get("paired_features")
    if pair is None:
        if isinstance(imported.get("cases"), dict):
            pair = {"cases": imported.get("cases")}
        else:
            is_plain_pair_case_map = any(
                isinstance(k, str)
                and isinstance(v, dict)
                and ("paired_features" in v or "matched_peak_sets" in v)
                for k, v in imported.items()
            )
            pair = {"cases": imported} if is_plain_pair_case_map else {"cases": {}}
    elif isinstance(pair, dict) and "cases" not in pair:
        # If user provided {"paired_features": {"<case_path>": {...}}}, wrap it.
        looks_like_case_map = any(isinstance(k, str) and isinstance(v, dict) for k, v in pair.items())
        if looks_like_case_map:
            pair = {"cases": pair}

    if not isinstance(seg, dict):
        raise ValueError("saved_segments must be an object")
    if not isinstance(pair, dict):
        pair = {"cases": {}}
    if not isinstance(pair.get("cases"), dict):
        pair = {"cases": {}}

    seg = remap_state_dict_paths(seg)
    pair["cases"] = remap_state_dict_paths(pair.get("cases", {}))
    return {"saved_segments": seg, "peak_pair_features": pair}


def list_tool2_metadata():
    out = []
    for root, _, files in os.walk(SMOOTHED_DIR):
        for name in files:
            if name.endswith(".csv.metadata.json"):
                out.append(os.path.relpath(os.path.join(root, name), ROOT_DIR).replace(os.sep, "/"))
    return sorted(out)


def smooth_sparse(values, ratio):
    known = [value for value in values if value is not None]
    smoothed_known = moving_average(known, ratio) if known else []
    out = []
    idx = 0
    for value in values:
        if value is None:
            out.append(None)
        else:
            out.append(smoothed_known[idx])
            idx += 1
    return out


def write_smoothed_rows_csv(output_rel_path, rows):
    output_abs_path = os.path.join(ROOT_DIR, safe_rel_path(output_rel_path))
    ensure_dir(os.path.dirname(output_abs_path))
    fieldnames = [
        "Timestamp",
        "Channel1",
        "Channel2",
        "Channel3",
        "Delta_Channel2_minus_Channel1",
        "Delta_Channel2_minus_Channel3",
        "CGM",
        "Normalized_Delta_Channel2_minus_Channel1",
        "Normalized_Delta_Channel2_minus_Channel3",
        "Normalized_CGM",
    ]
    with open(output_abs_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Timestamp": row["Timestamp"],
                    "Channel1": f"{row['Channel1']:.6f}",
                    "Channel2": f"{row['Channel2']:.6f}",
                    "Channel3": f"{row['Channel3']:.6f}",
                    "Delta_Channel2_minus_Channel1": f"{row['Delta_Channel2_minus_Channel1']:.6f}",
                    "Delta_Channel2_minus_Channel3": f"{row['Delta_Channel2_minus_Channel3']:.6f}",
                    "CGM": "NA" if row.get("CGM") is None else f"{row['CGM']:.6f}",
                    "Normalized_Delta_Channel2_minus_Channel1": "NA" if row.get("Normalized_Delta_Channel2_minus_Channel1") is None else f"{row['Normalized_Delta_Channel2_minus_Channel1']:.6f}",
                    "Normalized_Delta_Channel2_minus_Channel3": "NA" if row.get("Normalized_Delta_Channel2_minus_Channel3") is None else f"{row['Normalized_Delta_Channel2_minus_Channel3']:.6f}",
                    "Normalized_CGM": "NA" if row.get("Normalized_CGM") is None else f"{row['Normalized_CGM']:.6f}",
                }
            )


def smooth_processed_rows(rows):
    if not rows:
        return []
    ch1 = moving_average([row["Channel1"] for row in rows], 0.02)
    ch2 = moving_average([row["Channel2"] for row in rows], 0.02)
    ch3 = moving_average([row["Channel3"] for row in rows], 0.02)
    d21 = moving_average([row["Delta_Channel2_minus_Channel1"] for row in rows], 0.03)
    d23 = moving_average([row["Delta_Channel2_minus_Channel3"] for row in rows], 0.03)
    cgm = [row.get("CGM") for row in rows]
    norm_d21 = minmax(d21)
    norm_d23 = minmax(d23)
    norm_cgm_known = minmax([value for value in cgm if value is not None])
    norm_cgm = []
    idx = 0
    for value in cgm:
        if value is None:
            norm_cgm.append(None)
        else:
            norm_cgm.append(norm_cgm_known[idx])
            idx += 1
    out = []
    for i, row in enumerate(rows):
        out.append(
            {
                "Timestamp": row["Timestamp"],
                "Channel1": ch1[i],
                "Channel2": ch2[i],
                "Channel3": ch3[i],
                "Delta_Channel2_minus_Channel1": d21[i],
                "Delta_Channel2_minus_Channel3": d23[i],
                "CGM": cgm[i],
                "Normalized_Delta_Channel2_minus_Channel1": norm_d21[i],
                "Normalized_Delta_Channel2_minus_Channel3": norm_d23[i],
                "Normalized_CGM": norm_cgm[i],
            }
        )
    return out


def ensure_smoothed_processed_files():
    ensure_dir(SMOOTHED_DIR)
    for source_rel in source_processed_files():
        smoothed_rel = smoothed_rel_path(source_rel)
        source_abs = os.path.join(ROOT_DIR, safe_rel_path(source_rel))
        smoothed_abs = os.path.join(ROOT_DIR, safe_rel_path(smoothed_rel))
        source_meta_rel = metadata_rel_for_csv(source_rel)
        smoothed_meta_rel = metadata_rel_for_csv(smoothed_rel)
        source_meta_abs = os.path.join(ROOT_DIR, source_meta_rel)
        smoothed_meta_abs = os.path.join(ROOT_DIR, smoothed_meta_rel)
        needs_update = not os.path.exists(smoothed_abs)
        if not needs_update:
            source_mtime = max(os.path.getmtime(source_abs), os.path.getmtime(source_meta_abs) if os.path.exists(source_meta_abs) else 0)
            target_mtime = os.path.getmtime(smoothed_abs)
            needs_update = target_mtime < source_mtime
        if not needs_update and os.path.exists(smoothed_meta_abs):
            existing_meta = load_metadata_json(smoothed_meta_rel) or {}
            smoothing = existing_meta.get("smoothing", {})
            needs_update = smoothing.get("CGM") not in (None, "raw", 0, 0.0)
        if not needs_update:
            continue
        rows = load_processed_series(source_rel)
        smoothed_rows = smooth_processed_rows(rows)
        write_smoothed_rows_csv(smoothed_rel, smoothed_rows)
        source_meta = load_metadata_json(source_meta_rel) or {}
        metadata = dict(source_meta)
        metadata["source_path"] = source_rel
        metadata["output_path"] = smoothed_rel
        metadata["smoothing"] = {
            "Channel1": 0.02,
            "Channel2": 0.02,
            "Channel3": 0.02,
            "Delta_Channel2_minus_Channel1": 0.03,
            "Delta_Channel2_minus_Channel3": 0.03,
            "CGM": "raw",
        }
        metadata["row_count"] = len(smoothed_rows)
        metadata["start_timestamp"] = smoothed_rows[0]["Timestamp"] if smoothed_rows else ""
        metadata["end_timestamp"] = smoothed_rows[-1]["Timestamp"] if smoothed_rows else ""
        with open(os.path.join(ROOT_DIR, smoothed_meta_rel), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)


def load_segment_state():
    # Single source of truth: unified annotation state file.
    source_path = ANNOTATION_STATE_JSON if os.path.exists(ANNOTATION_STATE_JSON) else SEGMENT_STATE_JSON
    if not os.path.exists(source_path):
        return {}
    payload = {}
    with open(source_path, encoding="utf-8") as handle:
        raw = handle.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Recover from files with trailing/extra JSON fragments by decoding only
        # the first valid JSON value, then backing up the malformed source.
        decoder = json.JSONDecoder()
        recovered = None
        for start in range(len(raw)):
            ch = raw[start]
            if ch not in "{[":
                continue
            try:
                obj, _end = decoder.raw_decode(raw[start:])
                recovered = obj
                break
            except Exception:
                continue
        if recovered is None:
            return {}
        payload = recovered
        try:
            backup = source_path + ".corrupt"
            with open(backup, "w", encoding="utf-8") as handle:
                handle.write(raw)
            # Rewrite canonical recovered JSON to unblock server usage.
            with open(source_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception:
            pass
    if not isinstance(payload, dict):
        return {}
    # Unified shape: {saved_segments, peak_pair_features}
    if "saved_segments" in payload and isinstance(payload.get("saved_segments"), dict):
        payload = payload.get("saved_segments", {})
    state = {}
    payload = remap_state_dict_paths(payload)
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, dict):
            try:
                rel = safe_rel_path(key)
            except Exception:
                continue
            if "start_index" in value and "end_index" in value:
                try:
                    start = int(value.get("start_index", 0))
                    end = int(value.get("end_index", 0))
                except Exception:
                    continue
                state[rel] = {
                    "included_glu_cgm": True,
                    "included_glu_lac": True,
                    "glu_cgm": {"start_index": start, "end_index": end},
                    "glu_lac": {"start_index": start, "end_index": end},
                }
                continue
            per_panel = {}
            included_glu_cgm = value.get("included_glu_cgm", value.get("included", True))
            included_glu_lac = value.get("included_glu_lac", value.get("included", True))
            per_panel["included_glu_cgm"] = bool(included_glu_cgm)
            per_panel["included_glu_lac"] = bool(included_glu_lac)
            for panel in ("glu_cgm", "glu_lac"):
                panel_value = value.get(panel)
                if not isinstance(panel_value, dict):
                    panel_pairs = []
                else:
                    try:
                        start = int(panel_value.get("start_index", 0))
                        end = int(panel_value.get("end_index", 0))
                        panel_pairs = [{"start_index": min(start, end), "end_index": max(start, end)}]
                    except Exception:
                        panel_pairs = []
                region_key = f"region_pairs_{panel}"
                raw_pairs = value.get(region_key, [])
                if isinstance(raw_pairs, list):
                    parsed_pairs = []
                    for entry in raw_pairs:
                        if not isinstance(entry, dict):
                            continue
                        try:
                            start = int(entry.get("start_index", 0))
                            end = int(entry.get("end_index", 0))
                        except Exception:
                            continue
                        parsed_pairs.append({"start_index": min(start, end), "end_index": max(start, end)})
                    if parsed_pairs:
                        panel_pairs = parsed_pairs
                per_panel[region_key] = panel_pairs
                label_key = f"labels_{panel}"
                raw_labels = value.get(label_key, [])
                label_entries = []
                if isinstance(raw_labels, list):
                    for i, entry in enumerate(raw_labels):
                        if not isinstance(entry, dict):
                            continue
                        name = str(entry.get("name", f"L{i+1}"))
                        try:
                            idx = int(entry.get("index", 0))
                        except Exception:
                            continue
                        try:
                            counter = int(entry.get("counter", i + 1))
                        except Exception:
                            counter = i + 1
                        # Preserve optional fields such as timestamp_ms,
                        # cgm_value, and origin_panel. Dropping them makes
                        # reloaded annotations look incomplete and can break
                        # panel-specific SEG isolation.
                        preserved = dict(entry)
                        preserved["name"] = name
                        preserved["index"] = idx
                        preserved["counter"] = max(1, counter)
                        label_entries.append(preserved)
                per_panel[label_key] = label_entries
                if panel_pairs:
                    per_panel[panel] = {"start_index": panel_pairs[-1]["start_index"], "end_index": panel_pairs[-1]["end_index"]}
            if per_panel.get("region_pairs_glu_cgm") is not None or per_panel.get("region_pairs_glu_lac") is not None:
                state[rel] = per_panel
    return state


def save_segment_state(state):
    for rel_path, value in list(state.items()):
        if not isinstance(rel_path, str) or not isinstance(value, dict):
            continue
        try:
            markers = load_diet_markers(rel_path)
        except Exception:
            markers = []
        value["dietary_timestamps"] = [marker.get("timestamp", "") for marker in markers if isinstance(marker, dict)]
        pid = participant_id_from_rel(rel_path)
        features = load_lab_features_by_participant().get(pid, {}) if pid else {}
        if features:
            value["lab_features"] = features
    ensure_dir(STATE_DIR)
    # Preserve existing peak-pair entries while writing unified file.
    existing = load_json(ANNOTATION_STATE_JSON, {})
    if not isinstance(existing, dict):
        existing = {}
    peak = existing.get("peak_pair_features")
    if not isinstance(peak, dict):
        peak = load_peak_pair_state()
    unified = {
        "meta": {"saved_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), "mode": "single_file"},
        "saved_segments": state,
        "peak_pair_features": peak if isinstance(peak, dict) else {"cases": {}},
        "paired_features": peak if isinstance(peak, dict) else {"cases": {}},
    }
    with open(ANNOTATION_STATE_JSON, "w", encoding="utf-8") as handle:
        json.dump(unified, handle, indent=2)


def load_peak_pair_state():
    payload = {}
    if os.path.exists(ANNOTATION_STATE_JSON):
        unified = load_json(ANNOTATION_STATE_JSON, {})
        if isinstance(unified, dict):
            payload = unified.get("peak_pair_features", unified.get("paired_features", {}))
    elif os.path.exists(PEAK_PAIR_JSON):
        with open(PEAK_PAIR_JSON, encoding="utf-8") as handle:
            payload = json.load(handle)
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("cases", {})
    return payload


def save_peak_pair_state(payload):
    ensure_dir(STATE_DIR)
    existing = load_json(ANNOTATION_STATE_JSON, {})
    if not isinstance(existing, dict):
        existing = {}
    seg = existing.get("saved_segments")
    if not isinstance(seg, dict):
        seg = load_segment_state()
    unified = {
        "meta": {"saved_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), "mode": "single_file"},
        "saved_segments": seg if isinstance(seg, dict) else {},
        "peak_pair_features": payload if isinstance(payload, dict) else {"cases": {}},
        "paired_features": payload if isinstance(payload, dict) else {"cases": {}},
    }
    with open(ANNOTATION_STATE_JSON, "w", encoding="utf-8") as handle:
        json.dump(unified, handle, indent=2)


def _signal_kind_from_label_name(name):
    up = str(name or "").upper()
    signal = None
    if up.startswith("SEG "):
        signal = "seg"
    elif up.startswith("SEL "):
        signal = "sel"
    elif up.startswith("CGM "):
        signal = "cgm"
    kind = "peak" if "PEAK" in up else ("trough" if "TROUGH" in up else None)
    return signal, kind


def _label_identity(entry):
    if not isinstance(entry, dict):
        return None
    signal, kind = _signal_kind_from_label_name(entry.get("name"))
    if not signal or not kind:
        return None
    return (
        signal,
        kind,
        str(entry.get("name", "")),
        int(entry.get("counter", 1) or 1),
        int(entry.get("index", 0) or 0),
        str(entry.get("timestamp", "") or ""),
        _to_finite_float(entry.get("timestamp_ms")),
    )


def visible_labels_for_panel(file_state, panel):
    if not isinstance(file_state, dict):
        return []
    key = panel_label_key(panel)
    labels = file_state.get(key, [])
    if not isinstance(labels, list):
        return []
    if panel != "glu_lac":
        return labels
    cgm_labels = file_state.get("labels_glu_cgm", [])
    if not isinstance(cgm_labels, list):
        cgm_labels = []
    cgm_seg_identities = {
        ident
        for ident in (_label_identity(entry) for entry in cgm_labels)
        if ident and ident[0] == "seg"
    }
    visible = []
    for entry in labels:
        ident = _label_identity(entry)
        origin = str(entry.get("origin_panel", "") or "")
        if ident and ident[0] == "seg" and ident in cgm_seg_identities and origin != "glu_lac":
            continue
        visible.append(entry)
    return visible


def _dedup_labels_for_aggregation(file_state, label_keys=None):
    merged = []
    keys = tuple(label_keys or ("labels_glu_cgm", "labels_glu_lac"))
    for key in keys:
        if key == "labels_glu_cgm":
            entries = visible_labels_for_panel(file_state, "glu_cgm")
        elif key == "labels_glu_lac":
            entries = visible_labels_for_panel(file_state, "glu_lac")
        else:
            entries = file_state.get(key, []) or []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            signal, kind = _signal_kind_from_label_name(entry.get("name"))
            if not signal or not kind:
                continue
            merged.append(
                {
                    "name": str(entry.get("name", "")),
                    "counter": int(entry.get("counter", 1) or 1),
                    "index": int(entry.get("index", 0) or 0),
                    "timestamp": str(entry.get("timestamp", "") or ""),
                    "timestamp_ms": _to_finite_float(entry.get("timestamp_ms")),
                    "signal": signal,
                    "kind": kind,
                }
            )
    uniq = {}
    for x in merged:
        uniq[(x["name"], x["counter"], x["index"], x["signal"], x["kind"])] = x
    return sorted(uniq.values(), key=lambda x: (x.get("timestamp_ms") if x.get("timestamp_ms") is not None else x["index"], x["name"]))


def compute_aggregated_peaks_for_case(path, file_state, rows, label_keys=None):
    labels = _dedup_labels_for_aggregation(file_state if isinstance(file_state, dict) else {}, label_keys=label_keys)
    buckets = {"seg": {"peak": [], "trough": []}, "sel": {"peak": [], "trough": []}, "cgm": {"peak": [], "trough": []}}
    for lb in labels:
        buckets[lb["signal"]][lb["kind"]].append(lb)
    for sig in ("seg", "sel", "cgm"):
        buckets[sig]["peak"].sort(key=lambda x: (x.get("timestamp_ms") if x.get("timestamp_ms") is not None else x["index"]))
        buckets[sig]["trough"].sort(key=lambda x: (x.get("timestamp_ms") if x.get("timestamp_ms") is not None else x["index"]))
    def label_timestamp(label, fallback_index):
        explicit = str(label.get("timestamp", "") or "")
        if explicit:
            return explicit
        return rows[fallback_index]["Timestamp"] if 0 <= fallback_index < len(rows) else ""
    def label_order(label):
        explicit = label.get("timestamp_ms")
        return explicit if explicit is not None else int(label.get("index", 0) or 0)
    out = []
    for sig in ("seg", "sel", "cgm"):
        troughs = buckets[sig]["trough"]
        for peak in buckets[sig]["peak"]:
            pi = int(peak["index"])
            po = label_order(peak)
            left = None
            right = None
            for t in troughs:
                ti = int(t["index"])
                to = label_order(t)
                if to < po:
                    left = t
                elif to > po:
                    right = t
                    break
            if left is None or right is None:
                continue
            li = int(left["index"])
            ri = int(right["index"])
            if li < 0 or ri >= len(rows) or ri <= li:
                continue
            out.append(
                {
                    "id": f"{sig}_p{pi}_l{li}_r{ri}",
                    "signal": sig,
                    "peak": {"index": pi, "name": peak["name"], "counter": peak["counter"], "timestamp": label_timestamp(peak, pi)},
                    "left_trough": {"index": li, "name": left["name"], "counter": left["counter"], "timestamp": label_timestamp(left, li)},
                    "right_trough": {"index": ri, "name": right["name"], "counter": right["counter"], "timestamp": label_timestamp(right, ri)},
                    "window": {"start_index": li, "end_index": ri, "duration_points": ri - li},
                }
            )
    return out


def canonical_label_name(name):
    key = str(name or "").strip().lower()
    if key not in ALLOWED_LABEL_SET_LOWER:
        raise ValueError("Invalid label. Use one of the predefined options.")
    return ALLOWED_LABEL_SET_LOWER[key]


def next_label_counter(labels, base_name):
    max_counter = 0
    for entry in labels:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("name", "")).strip() != base_name:
            continue
        try:
            counter = int(entry.get("counter", 0))
        except Exception:
            counter = 0
        if counter > max_counter:
            max_counter = counter
    return max_counter + 1


def is_seg_label(name):
    return str(name or "").strip().lower().startswith("seg ")


def seg_label_sync_enabled(file_state):
    # Labeling lanes are intentionally isolated: Glu-CGM labels should not
    # mutate Glu-Lac labels, and vice versa.
    return False


def aggregation_label_keys_for_pair_category(file_state, category):
    if seg_label_sync_enabled(file_state):
        return None
    if category == "CGM_vs_SEG":
        return ("labels_glu_cgm",)
    if category == "SEG_vs_SEL":
        return ("labels_glu_lac",)
    return ("labels_glu_cgm", "labels_glu_lac")


def label_panel_for_pair_signal(category, signal):
    category = str(category or "")
    signal = str(signal or "").strip().lower()
    if category == "CGM_vs_SEG":
        if signal in ("seg", "cgm"):
            return "glu_cgm"
        if signal == "sel":
            return "glu_lac"
        return None
    if category == "SEG_vs_SEL":
        return "glu_lac" if signal in ("seg", "sel") else None
    if category == "CGM_vs_SEL":
        if signal == "cgm":
            return "glu_cgm"
        if signal == "sel":
            return "glu_lac"
    return None


def anchor_signal_for_pair_category(category):
    category = str(category or "")
    if category == "CGM_vs_SEG":
        return "cgm"
    if category == "SEG_vs_SEL":
        return "sel"
    if category == "CGM_vs_SEL":
        return "cgm"
    return None


def apply_pair_context_to_peak_meta(meta, category):
    if not isinstance(meta, dict):
        return meta
    signal = str(meta.get("signal") or "").strip().lower()
    meta["pair_context"] = str(category or "")
    panel = label_panel_for_pair_signal(category, signal)
    if panel:
        meta["label_panel"] = panel
    return meta


def apply_pair_preceding_instances(rel_path, pair):
    if not isinstance(pair, dict):
        return pair
    selected = pair.get("selected_peak_metadata")
    if not isinstance(selected, dict):
        return pair
    category = str(pair.get("category", "") or "")
    anchor_signal = anchor_signal_for_pair_category(category)
    anchor_meta = selected.get(anchor_signal) if anchor_signal else None
    anchor_peak_ts = anchor_meta.get("peak_timestamp", "") if isinstance(anchor_meta, dict) else ""
    shared_preceding = find_preceding_diet_instance_for_peak(rel_path, anchor_peak_ts) if anchor_peak_ts else None
    for signal, meta in selected.items():
        if not isinstance(meta, dict):
            continue
        apply_pair_context_to_peak_meta(meta, category)
        peak_ts = meta.get("peak_timestamp", "")
        meta["preceding_instance"] = decorated_preceding_instance_for_peak(
            shared_preceding,
            peak_ts,
            anchor_signal=anchor_signal,
            anchor_peak_timestamp=anchor_peak_ts,
        )
    return pair


def panel_label_key(panel):
    return "labels_glu_cgm" if panel == "glu_cgm" else "labels_glu_lac"


def other_panel(panel):
    return "glu_lac" if panel == "glu_cgm" else "glu_cgm"


def participant_id_from_rel(rel_path):
    match = PATIENT_ID_RE.search(str(rel_path))
    if not match:
        return None
    return f"{match.group(1).upper()}{int(match.group(2)):02d}"


def day_id_from_rel(rel_path):
    parts = safe_rel_path(rel_path).split("/")
    for part in parts:
        if re.match(r"^Day\d{1,2}$", str(part), re.IGNORECASE):
            m = re.match(r"^Day(\d{1,2})$", str(part), re.IGNORECASE)
            if m:
                return f"Day{int(m.group(1)):02d}"
            return str(part)
    return None


def load_diet_instances():
    global DIET_INSTANCES_CACHE
    if DIET_INSTANCES_CACHE is not None:
        return DIET_INSTANCES_CACHE
    payload = load_json(DIET_INSTANCES_JSON, {})
    if not isinstance(payload, dict):
        payload = {}
    instances = payload.get("instances", {})
    if not isinstance(instances, dict):
        instances = {}
    DIET_INSTANCES_CACHE = instances
    return instances


def load_diet_photo_annotations():
    global DIET_PHOTO_ANN_CACHE
    if DIET_PHOTO_ANN_CACHE is not None:
        return DIET_PHOTO_ANN_CACHE
    payload = load_json(DIET_PHOTO_ANNOTATIONS_JSON, {})
    if not isinstance(payload, dict):
        payload = {}
    DIET_PHOTO_ANN_CACHE = payload
    return payload


def _normalize_patient_id(text):
    s = str(text or "").strip()
    m = re.match(r"^([PH])?(\d{1,3})$", s, re.IGNORECASE)
    if not m:
        return s
    prefix = (m.group(1) or "P").upper()
    return f"{prefix}{int(m.group(2)):02d}"


def _normalize_day_id(text):
    s = str(text or "").strip()
    m = re.match(r"^Day(\d{1,2})$", s, re.IGNORECASE)
    if not m:
        return s
    return f"Day{int(m.group(1)):02d}"


def find_preceding_diet_instance_for_peak(rel_path, peak_timestamp):
    peak_dt = parse_timestamp_flexible(peak_timestamp)
    if peak_dt is None:
        return None
    pid = participant_id_from_rel(rel_path)
    day = day_id_from_rel(rel_path)
    if not pid or not day:
        return None
    instances = load_diet_instances()
    photo_ann = load_diet_photo_annotations()
    best = None
    for inst_id, inst in instances.items():
        if not isinstance(inst, dict):
            continue
        if _normalize_patient_id(inst.get("patient")) != pid:
            continue
        if _normalize_day_id(inst.get("day")) != day:
            continue
        ts_list = inst.get("timestamps", [])
        if not isinstance(ts_list, list):
            ts_list = []
        dt_candidates = []
        for ts in ts_list:
            dt = parse_timestamp_flexible(ts)
            if dt is not None:
                dt_candidates.append(dt)
        if not dt_candidates:
            continue
        # Use the latest timestamp in the instance as the instance anchor.
        anchor_dt = max(dt_candidates)
        if anchor_dt > peak_dt:
            continue
        if best is None or anchor_dt > best["anchor_dt"]:
            label_types = set()
            photo_rels = inst.get("photo_rels", [])
            if not isinstance(photo_rels, list):
                photo_rels = []
            for rel in photo_rels:
                info = photo_ann.get(str(rel), {})
                if not isinstance(info, dict):
                    continue
                t = str(info.get("label_type", "")).strip().lower()
                if t:
                    label_types.add(t)
            if not label_types:
                t = str(inst.get("label_type", "")).strip().lower()
                if t:
                    label_types.add(t)
            if len(label_types) == 1:
                meal_med = next(iter(label_types))
            elif len(label_types) > 1:
                meal_med = "mixed:" + ",".join(sorted(label_types))
            else:
                meal_med = "unknown"
            mode = str(inst.get("mode", "")).strip().lower()
            total_cal = inst.get("total_calories_intake")
            calories = None
            if mode == "aggregate":
                calories = total_cal
            else:
                if total_cal is not None:
                    calories = total_cal
                else:
                    cals = inst.get("calories", [])
                    if isinstance(cals, list) and cals:
                        calories = cals[0]
            best = {
                "instance_id": str(inst_id),
                "anchor_dt": anchor_dt,
                "instance_timestamp": anchor_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "meal_med": meal_med,
                "calories": calories,
                "mode": mode or None,
            }
    if best is None:
        return None
    best["instance_to_peak_seconds"] = max(0, int((peak_dt - best["anchor_dt"]).total_seconds()))
    best.pop("anchor_dt", None)
    return best


def with_instance_duration_for_peak(preceding_instance, peak_timestamp):
    if not isinstance(preceding_instance, dict):
        return None
    out = dict(preceding_instance)
    inst_ts = out.get("instance_timestamp", "")
    inst_dt = parse_timestamp_flexible(inst_ts)
    peak_dt = parse_timestamp_flexible(peak_timestamp)
    if inst_dt is None or peak_dt is None:
        out["instance_to_peak_seconds"] = None
        return out
    out["instance_to_peak_seconds"] = max(0, int((peak_dt - inst_dt).total_seconds()))
    return out


def preceding_instance_placeholder(peak_timestamp, anchor_signal=None, anchor_peak_timestamp="", reason="no_matching_preceding_instance"):
    return {
        "instance_id": None,
        "instance_timestamp": "",
        "meal_med": "",
        "calories": None,
        "mode": None,
        "instance_to_peak_seconds": None,
        "has_preceding_instance": 0,
        "lookup_status": reason,
        "anchor_signal": anchor_signal,
        "anchor_peak_timestamp": anchor_peak_timestamp or "",
        "target_peak_timestamp": peak_timestamp or "",
    }


def decorated_preceding_instance_for_peak(preceding_instance, peak_timestamp, anchor_signal=None, anchor_peak_timestamp=""):
    if not isinstance(preceding_instance, dict):
        return preceding_instance_placeholder(
            peak_timestamp,
            anchor_signal=anchor_signal,
            anchor_peak_timestamp=anchor_peak_timestamp,
        )
    out = with_instance_duration_for_peak(preceding_instance, peak_timestamp)
    if not isinstance(out, dict):
        return preceding_instance_placeholder(
            peak_timestamp,
            anchor_signal=anchor_signal,
            anchor_peak_timestamp=anchor_peak_timestamp,
        )
    out["has_preceding_instance"] = 1
    out["lookup_status"] = "matched"
    out["anchor_signal"] = anchor_signal
    out["anchor_peak_timestamp"] = anchor_peak_timestamp or ""
    out["target_peak_timestamp"] = peak_timestamp or ""
    return out


def diet_instances_for_case(rel_path):
    pid = participant_id_from_rel(rel_path)
    day = day_id_from_rel(rel_path)
    if not pid or not day:
        return []
    instances = load_diet_instances()
    photo_ann = load_diet_photo_annotations()
    out = []
    for inst_id, inst in instances.items():
        if not isinstance(inst, dict):
            continue
        if _normalize_patient_id(inst.get("patient")) != pid:
            continue
        if _normalize_day_id(inst.get("day")) != day:
            continue
        ts_list = inst.get("timestamps", [])
        if not isinstance(ts_list, list):
            ts_list = []
        dt_candidates = []
        for ts in ts_list:
            dt = parse_timestamp_flexible(ts)
            if dt is not None:
                dt_candidates.append(dt)
        if not dt_candidates:
            continue
        anchor_dt = max(dt_candidates)
        label_types = set()
        photo_rels = inst.get("photo_rels", [])
        if not isinstance(photo_rels, list):
            photo_rels = []
        for rel in photo_rels:
            info = photo_ann.get(str(rel), {})
            if not isinstance(info, dict):
                continue
            t = str(info.get("label_type", "")).strip().lower()
            if t:
                label_types.add(t)
        if not label_types:
            t = str(inst.get("label_type", "")).strip().lower()
            if t:
                label_types.add(t)
        if len(label_types) == 1:
            meal_med = next(iter(label_types))
        elif len(label_types) > 1:
            meal_med = "mixed"
        else:
            meal_med = "unknown"
        mode = str(inst.get("mode", "")).strip().lower()
        total_cal = inst.get("total_calories_intake")
        calories = None
        if mode == "aggregate":
            calories = total_cal
        elif total_cal is not None:
            calories = total_cal
        else:
            cals = inst.get("calories", [])
            if isinstance(cals, list) and cals:
                calories = cals[0]
        out.append(
            {
                "instance_id": str(inst_id),
                "timestamp": anchor_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "meal_med": meal_med,
                "mode": mode or None,
                "calories": calories,
                "label": f"{meal_med} | {('kcal ' + str(calories)) if calories is not None else 'kcal NA'}",
            }
        )
    out.sort(key=lambda x: str(x.get("timestamp", "")))
    return out


def _xlsx_col_to_idx(col_ref):
    idx = 0
    for ch in col_ref:
        if not ("A" <= ch <= "Z"):
            break
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _load_xlsx_rows(path):
    if not os.path.exists(path):
        return []
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path, "r") as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in shared_root.findall("x:si", ns):
                parts = []
                for t in si.findall(".//x:t", ns):
                    parts.append(t.text or "")
                shared.append("".join(parts))
        wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
        first_sheet = wb_root.find("x:sheets/x:sheet", ns)
        if first_sheet is None:
            return []
        rel_id = first_sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        rel_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in rel_root:
            if rel.attrib.get("Id") == rel_id:
                target = rel.attrib.get("Target")
                break
        if not target:
            return []
        if not target.startswith("xl/"):
            target = "xl/" + target.lstrip("/")
        sheet_root = ET.fromstring(zf.read(target))
        rows = []
        for row in sheet_root.findall(".//x:sheetData/x:row", ns):
            values = {}
            max_idx = -1
            for c in row.findall("x:c", ns):
                ref = c.attrib.get("r", "")
                col_ref = "".join(ch for ch in ref if ch.isalpha())
                if not col_ref:
                    continue
                idx = _xlsx_col_to_idx(col_ref)
                max_idx = max(max_idx, idx)
                ctype = c.attrib.get("t")
                v = c.find("x:v", ns)
                if ctype == "s" and v is not None and v.text is not None:
                    try:
                        val = shared[int(v.text)]
                    except Exception:
                        val = v.text
                elif ctype == "inlineStr":
                    t = c.find(".//x:t", ns)
                    val = t.text if t is not None else ""
                else:
                    val = v.text if v is not None else ""
                values[idx] = val
            if max_idx < 0:
                continue
            row_list = [values.get(i, "") for i in range(max_idx + 1)]
            rows.append(row_list)
        return rows


def load_lab_features_by_participant():
    global LAB_FEATURE_CACHE
    if LAB_FEATURE_CACHE is not None:
        return LAB_FEATURE_CACHE
    rows = _load_xlsx_rows(LAB_DATA_XLSX)
    if not rows:
        LAB_FEATURE_CACHE = {}
        return LAB_FEATURE_CACHE
    headers = [str(h).strip() for h in rows[0]]
    id_col = None
    for i, h in enumerate(headers):
        low = h.lower()
        if any(token in low for token in ("participant", "patient", "subject", "id")):
            id_col = i
            break
    if id_col is None:
        for i in range(len(headers)):
            for row in rows[1:]:
                if i < len(row) and PATIENT_ID_RE.search(str(row[i])):
                    id_col = i
                    break
            if id_col is not None:
                break
    id_mode = "patient_string"
    if id_col is not None:
        header_name = str(headers[id_col]).strip().lower()
        if header_name in ("record_id", "recordid", "participant_id", "participantid", "subject_id", "subjectid"):
            id_mode = "numeric_record"
    features = {}
    if id_col is None:
        LAB_FEATURE_CACHE = features
        return features
    for row in rows[1:]:
        if id_col >= len(row):
            continue
        raw_id = str(row[id_col]).strip()
        if id_mode == "numeric_record":
            prefixed = re.match(r"^([PH])\s*0*(\d{1,3})$", raw_id, re.IGNORECASE)
            if prefixed:
                pid = f"{prefixed.group(1).upper()}{int(prefixed.group(2)):02d}"
            else:
                try:
                    pid_num = int(float(raw_id))
                except Exception:
                    continue
                if pid_num <= 0:
                    continue
                pid = f"P{pid_num:02d}"
        else:
            pid_match = PATIENT_ID_RE.search(raw_id)
            if not pid_match:
                continue
            pid = f"{pid_match.group(1).upper()}{int(pid_match.group(2)):02d}"
        payload = {}
        for i, h in enumerate(headers):
            if i == id_col:
                continue
            key = str(h).strip()
            if not key:
                continue
            value = row[i] if i < len(row) else ""
            text = str(value).strip()
            if text == "":
                continue
            payload[key] = text
        features[pid] = payload
    LAB_FEATURE_CACHE = features
    return features


def summarize_processed_file(rel_path):
    rows = load_processed_series(rel_path)
    cgm_matches = sum(1 for row in rows if row.get("CGM") is not None)
    return {
        "path": rel_path,
        "row_count": len(rows),
        "duration_hms": format_duration(len(rows)),
        "start": rows[0]["Timestamp"] if rows else "",
        "end": rows[-1]["Timestamp"] if rows else "",
        "matched_cgm_points": cgm_matches,
    }


def summarize_processed_file_fast(rel_path):
    metadata = load_metadata_json(metadata_rel_for_csv(rel_path))
    if metadata:
        return {
            "path": rel_path,
            "row_count": int(metadata.get("row_count", 0)),
            "duration_hms": format_duration(int(metadata.get("row_count", 0))),
            "start": metadata.get("start_timestamp", ""),
            "end": metadata.get("end_timestamp", ""),
            "matched_cgm_points": None,
        }
    return summarize_processed_file(rel_path)


def pearson(xs, ys):
    n = min(len(xs), len(ys))
    if n < 2:
        return None
    xs = xs[:n]
    ys = ys[:n]
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def minmax(values):
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def rmse(xs, ys):
    n = min(len(xs), len(ys))
    if n < 1:
        return None
    xs = xs[:n]
    ys = ys[:n]
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(xs, ys)) / n)


def direction_agreement(xs, ys):
    n = min(len(xs), len(ys))
    if n < 2:
        return None
    agree = 0
    total = 0
    for i in range(1, n):
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        sx = 0 if abs(dx) < 1e-12 else (1 if dx > 0 else -1)
        sy = 0 if abs(dy) < 1e-12 else (1 if dy > 0 else -1)
        if sx == sy:
            agree += 1
        total += 1
    return (agree / total) if total else None


def lagged_pairs(xs, ys, lag_steps):
    pairs_x = []
    pairs_y = []
    n = min(len(xs), len(ys))
    if lag_steps >= 0:
        start_x = 0
        start_y = lag_steps
        length = n - lag_steps
    else:
        start_x = -lag_steps
        start_y = 0
        length = n + lag_steps
    if length <= 0:
        return [], []
    for offset in range(length):
        x = xs[start_x + offset]
        y = ys[start_y + offset]
        if x is None or y is None:
            continue
        pairs_x.append(x)
        pairs_y.append(y)
    return pairs_x, pairs_y


def best_lag_metric(xs, ys, max_lag_steps, metric_fn, min_pairs=3):
    best_value = None
    best_lag = 0
    for lag in range(-max_lag_steps, max_lag_steps + 1):
        lag_xs, lag_ys = lagged_pairs(xs, ys, lag)
        if len(lag_xs) < min_pairs:
            continue
        value = metric_fn(lag_xs, lag_ys)
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_lag = lag
    return best_value, best_lag


def compute_alignment_metrics(rows, max_lag_steps=0):
    matched = [row for row in rows if row.get("CGM") is not None]
    glucose_all = [row["Delta_Channel2_minus_Channel1"] for row in rows]
    lactate_all = [row["Delta_Channel2_minus_Channel3"] for row in rows]
    cgm_all = [row.get("CGM") for row in rows]
    lag_corr_glucose_cgm = None
    lag_corr_glucose_cgm_steps = 0
    lag_direction_glucose_cgm = None
    lag_direction_glucose_cgm_steps = 0
    lag_corr_glucose_lactate = None
    lag_corr_glucose_lactate_steps = 0
    lag_direction_glucose_lactate = None
    lag_direction_glucose_lactate_steps = 0
    if max_lag_steps > 0:
        lag_corr_glucose_cgm, lag_corr_glucose_cgm_steps = best_lag_metric(glucose_all, cgm_all, max_lag_steps, pearson)
        lag_direction_glucose_cgm, lag_direction_glucose_cgm_steps = best_lag_metric(glucose_all, cgm_all, max_lag_steps, direction_agreement)
        lag_corr_glucose_lactate, lag_corr_glucose_lactate_steps = best_lag_metric(glucose_all, lactate_all, max_lag_steps, pearson)
        lag_direction_glucose_lactate, lag_direction_glucose_lactate_steps = best_lag_metric(glucose_all, lactate_all, max_lag_steps, direction_agreement)
    if len(matched) < 3:
        return {
            "matched_cgm_points": len(matched),
            "corr_glucose_cgm": None,
            "corr_lactate_cgm": None,
            "corr_glucose_lactate": None,
            "rmse_glucose_cgm": None,
            "rmse_lactate_cgm": None,
            "direction_glucose_cgm": None,
            "direction_lactate_cgm": None,
            "best_lag_corr_glucose_cgm": lag_corr_glucose_cgm,
            "best_lag_corr_glucose_cgm_steps": lag_corr_glucose_cgm_steps,
            "best_lag_direction_glucose_cgm": lag_direction_glucose_cgm,
            "best_lag_direction_glucose_cgm_steps": lag_direction_glucose_cgm_steps,
            "best_lag_corr_glucose_lactate": lag_corr_glucose_lactate,
            "best_lag_corr_glucose_lactate_steps": lag_corr_glucose_lactate_steps,
            "best_lag_direction_glucose_lactate": lag_direction_glucose_lactate,
            "best_lag_direction_glucose_lactate_steps": lag_direction_glucose_lactate_steps,
        }
    glucose = [row["Delta_Channel2_minus_Channel1"] for row in matched]
    lactate = [row["Delta_Channel2_minus_Channel3"] for row in matched]
    cgm = [row["CGM"] for row in matched]
    glucose_norm = minmax(glucose)
    lactate_norm = minmax(lactate)
    cgm_norm = minmax(cgm)
    return {
        "matched_cgm_points": len(matched),
        "corr_glucose_cgm": pearson(glucose, cgm),
        "corr_lactate_cgm": pearson(lactate, cgm),
        "corr_glucose_lactate": pearson(glucose, lactate),
        "rmse_glucose_cgm": rmse(glucose_norm, cgm_norm),
        "rmse_lactate_cgm": rmse(lactate_norm, cgm_norm),
        "direction_glucose_cgm": direction_agreement(glucose, cgm),
        "direction_lactate_cgm": direction_agreement(lactate, cgm),
        "best_lag_corr_glucose_cgm": lag_corr_glucose_cgm,
        "best_lag_corr_glucose_cgm_steps": lag_corr_glucose_cgm_steps,
        "best_lag_direction_glucose_cgm": lag_direction_glucose_cgm,
        "best_lag_direction_glucose_cgm_steps": lag_direction_glucose_cgm_steps,
        "best_lag_corr_glucose_lactate": lag_corr_glucose_lactate,
        "best_lag_corr_glucose_lactate_steps": lag_corr_glucose_lactate_steps,
        "best_lag_direction_glucose_lactate": lag_direction_glucose_lactate,
        "best_lag_direction_glucose_lactate_steps": lag_direction_glucose_lactate_steps,
    }


def apply_segment(rows, segment):
    if not rows:
        return rows
    if not segment:
        return rows
    start = max(0, min(int(segment.get("start_index", 0)), len(rows) - 1))
    end = max(start, min(int(segment.get("end_index", len(rows) - 1)), len(rows) - 1))
    return rows[start : end + 1]


def aggregate_metric(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"mean": None, "median": None}
    return {
        "mean": sum(values) / len(values),
        "median": median(values),
    }


def panel_segment(segment_state, rel_path, panel, row_count):
    file_state = segment_state.get(rel_path, {}) if isinstance(segment_state.get(rel_path, {}), dict) else {}
    segment = file_state.get(panel)
    if not segment:
        return {"start_index": 0, "end_index": max(0, row_count - 1)}
    return segment


def file_included(segment_state, rel_path, panel):
    file_state = segment_state.get(rel_path, {}) if isinstance(segment_state.get(rel_path, {}), dict) else {}
    key = "included_glu_cgm" if panel == "glu_cgm" else "included_glu_lac"
    return bool(file_state.get(key, True))


def cohort_summary(max_lag_steps=0):
    state = load_segment_state()
    per_file = []
    for rel_path in list_processed_files():
        rows = load_processed_series(rel_path)
        included_glu_cgm = file_included(state, rel_path, "glu_cgm")
        included_glu_lac = file_included(state, rel_path, "glu_lac")
        glu_cgm_segment = panel_segment(state, rel_path, "glu_cgm", len(rows))
        glu_lac_segment = panel_segment(state, rel_path, "glu_lac", len(rows))
        glu_cgm_rows = apply_segment(rows, glu_cgm_segment)
        glu_lac_rows = apply_segment(rows, glu_lac_segment)
        glu_cgm_metrics = compute_alignment_metrics(glu_cgm_rows, max_lag_steps=max_lag_steps) if included_glu_cgm else compute_alignment_metrics([], max_lag_steps=max_lag_steps)
        glu_lac_metrics = compute_alignment_metrics(glu_lac_rows, max_lag_steps=max_lag_steps) if included_glu_lac else compute_alignment_metrics([], max_lag_steps=max_lag_steps)
        per_file.append(
            {
                "path": rel_path,
                "included_glu_cgm": included_glu_cgm,
                "included_glu_lac": included_glu_lac,
                "glu_cgm_segment_start_index": glu_cgm_segment["start_index"],
                "glu_cgm_segment_end_index": glu_cgm_segment["end_index"],
                "glu_cgm_segment_row_count": len(glu_cgm_rows),
                "glu_cgm_segment_duration_hms": format_duration(len(glu_cgm_rows)),
                "glu_cgm_segment_start_time": glu_cgm_rows[0]["Timestamp"] if glu_cgm_rows else "",
                "glu_cgm_segment_end_time": glu_cgm_rows[-1]["Timestamp"] if glu_cgm_rows else "",
                "glu_lac_segment_start_index": glu_lac_segment["start_index"],
                "glu_lac_segment_end_index": glu_lac_segment["end_index"],
                "glu_lac_segment_row_count": len(glu_lac_rows),
                "glu_lac_segment_duration_hms": format_duration(len(glu_lac_rows)),
                "glu_lac_segment_start_time": glu_lac_rows[0]["Timestamp"] if glu_lac_rows else "",
                "glu_lac_segment_end_time": glu_lac_rows[-1]["Timestamp"] if glu_lac_rows else "",
                "matched_cgm_points": glu_cgm_metrics["matched_cgm_points"],
                "corr_glucose_cgm": glu_cgm_metrics["corr_glucose_cgm"],
                "rmse_glucose_cgm": glu_cgm_metrics["rmse_glucose_cgm"],
                "direction_glucose_cgm": glu_cgm_metrics["direction_glucose_cgm"],
                "corr_lactate_cgm": glu_cgm_metrics["corr_lactate_cgm"],
                "rmse_lactate_cgm": glu_cgm_metrics["rmse_lactate_cgm"],
                "direction_lactate_cgm": glu_cgm_metrics["direction_lactate_cgm"],
                "corr_glucose_lactate": glu_lac_metrics["corr_glucose_lactate"],
                "best_lag_corr_glucose_cgm": glu_cgm_metrics["best_lag_corr_glucose_cgm"],
                "best_lag_corr_glucose_cgm_steps": glu_cgm_metrics["best_lag_corr_glucose_cgm_steps"],
                "best_lag_direction_glucose_cgm": glu_cgm_metrics["best_lag_direction_glucose_cgm"],
                "best_lag_direction_glucose_cgm_steps": glu_cgm_metrics["best_lag_direction_glucose_cgm_steps"],
                "best_lag_corr_glucose_lactate": glu_lac_metrics["best_lag_corr_glucose_lactate"],
                "best_lag_corr_glucose_lactate_steps": glu_lac_metrics["best_lag_corr_glucose_lactate_steps"],
                "best_lag_direction_glucose_lactate": glu_lac_metrics["best_lag_direction_glucose_lactate"],
                "best_lag_direction_glucose_lactate_steps": glu_lac_metrics["best_lag_direction_glucose_lactate_steps"],
            }
        )
    included_glu_cgm_files = [item for item in per_file if item["included_glu_cgm"]]
    included_glu_lac_files = [item for item in per_file if item["included_glu_lac"]]
    summary = {
        "included_glu_cgm_file_count": len(included_glu_cgm_files),
        "excluded_glu_cgm_file_count": len(per_file) - len(included_glu_cgm_files),
        "included_glu_lac_file_count": len(included_glu_lac_files),
        "excluded_glu_lac_file_count": len(per_file) - len(included_glu_lac_files),
        "files_with_cgm": sum(1 for item in included_glu_cgm_files if item["matched_cgm_points"] >= 3),
        "total_matched_cgm_points": sum(item["matched_cgm_points"] for item in included_glu_cgm_files),
        "corr_glucose_cgm": aggregate_metric([item["corr_glucose_cgm"] for item in included_glu_cgm_files]),
        "corr_lactate_cgm": aggregate_metric([item["corr_lactate_cgm"] for item in included_glu_cgm_files]),
        "corr_glucose_lactate": aggregate_metric([item["corr_glucose_lactate"] for item in included_glu_lac_files]),
        "rmse_glucose_cgm": aggregate_metric([item["rmse_glucose_cgm"] for item in included_glu_cgm_files]),
        "rmse_lactate_cgm": aggregate_metric([item["rmse_lactate_cgm"] for item in included_glu_cgm_files]),
        "direction_glucose_cgm": aggregate_metric([item["direction_glucose_cgm"] for item in included_glu_cgm_files]),
        "direction_lactate_cgm": aggregate_metric([item["direction_lactate_cgm"] for item in included_glu_cgm_files]),
        "best_lag_corr_glucose_cgm": aggregate_metric([item["best_lag_corr_glucose_cgm"] for item in included_glu_cgm_files]),
        "best_lag_direction_glucose_cgm": aggregate_metric([item["best_lag_direction_glucose_cgm"] for item in included_glu_cgm_files]),
        "best_lag_corr_glucose_lactate": aggregate_metric([item["best_lag_corr_glucose_lactate"] for item in included_glu_lac_files]),
        "best_lag_direction_glucose_lactate": aggregate_metric([item["best_lag_direction_glucose_lactate"] for item in included_glu_lac_files]),
        "per_file": per_file,
    }
    return summary


def rows_for_browser(rows):
    out = []
    for row in rows:
        out.append(
            {
                "Timestamp": row["Timestamp"],
                "Channel1": row["Channel1"],
                "Channel2": row["Channel2"],
                "Channel3": row["Channel3"],
                "Delta_Channel2_minus_Channel1": row["Delta_Channel2_minus_Channel1"],
                "Delta_Channel2_minus_Channel3": row["Delta_Channel2_minus_Channel3"],
                "CGM": row.get("CGM"),
                "Normalized_Delta_Channel2_minus_Channel1": row.get("Normalized_Delta_Channel2_minus_Channel1"),
                "Normalized_Delta_Channel2_minus_Channel3": row.get("Normalized_Delta_Channel2_minus_Channel3"),
                "Normalized_CGM": row.get("Normalized_CGM"),
            }
        )
    return out


def source_rel_from_processed_path(rel_path):
    return cgm_features.source_rel_from_processed_path(rel_path)


def independent_cgm_series_for_case(rel_path, processed_rows=None):
    return cgm_features.independent_cgm_series_for_case(
        rel_path,
        processed_rows=processed_rows,
        parse_timestamp=parse_timestamp_flexible,
    )


def _to_finite_float(value):
    return cgm_features.to_finite_float(value)


def signal_value_column(signal):
    return cgm_features.signal_value_column(signal)


def signal_points_for_interval(rel_path, rows, signal, start_ts, end_ts, cgm_points=None):
    return cgm_features.signal_points_for_interval(
        rows,
        signal,
        start_ts,
        end_ts,
        parse_timestamp_flexible,
        cgm_points=cgm_points,
    )


def normalized_interval_derivative_summary(points, start_ts, end_ts):
    return cgm_features.normalized_interval_derivative_summary(
        points,
        start_ts,
        end_ts,
        parse_timestamp_flexible,
    )


def nearest_signal_value_at_timestamp(rows, signal, timestamp):
    return cgm_features.nearest_signal_value_at_timestamp(
        rows,
        signal,
        timestamp,
        parse_timestamp_flexible,
    )


def aggregate_point_indices(aggregate_id):
    return cgm_features.aggregate_point_indices(aggregate_id)


def signal_value_at_index(rows, signal, index):
    return cgm_features.signal_value_at_index(rows, signal, index)


def point_value_features_for_peak(rows, peak_meta, cgm_points=None):
    return cgm_features.point_value_features_for_peak(
        rows,
        peak_meta,
        parse_timestamp_flexible,
        concentration_from_delta,
        cgm_points=cgm_points,
    )


def trough_to_trough_auc_features_for_peak(rows, peak_meta, cgm_points=None):
    return cgm_features.trough_to_trough_auc_features_for_peak(
        rows,
        peak_meta,
        parse_timestamp_flexible,
        concentration_from_delta,
        cgm_points=cgm_points,
    )


def interval_derivative_features_for_peak(rel_path, rows, peak_meta, cgm_points=None):
    return cgm_features.interval_derivative_features_for_peak(
        rows,
        peak_meta,
        parse_timestamp_flexible,
        cgm_points=cgm_points,
    )

def _timestamp_duration_seconds(start_ts, end_ts):
    start_dt = parse_timestamp_flexible(start_ts)
    end_dt = parse_timestamp_flexible(end_ts)
    if start_dt is None or end_dt is None:
        return None
    return (end_dt - start_dt).total_seconds()


def augment_peak_metadata_with_durations(peak_meta):
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
        seconds = _timestamp_duration_seconds(start_ts, end_ts)
        if seconds is None:
            peak_meta[f"{prefix}_seconds"] = None
        else:
            seconds = max(0.0, seconds)
            peak_meta[f"{prefix}_seconds"] = seconds
    return peak_meta


def augment_peak_metadata_with_preceding_peak_count(peak_meta, aggregates):
    if not isinstance(peak_meta, dict):
        return peak_meta
    preceding = peak_meta.get("preceding_instance")
    instance_ts = preceding.get("instance_timestamp", "") if isinstance(preceding, dict) else ""
    instance_dt = parse_timestamp_flexible(instance_ts)
    peak_dt = parse_timestamp_flexible(peak_meta.get("peak_timestamp", ""))
    signal = str(peak_meta.get("signal") or "").strip().lower()
    count = None
    if instance_dt is not None and peak_dt is not None and signal:
        count = 0
        for agg in aggregates or []:
            if not isinstance(agg, dict):
                continue
            if str(agg.get("signal", "")).strip().lower() != signal:
                continue
            other_peak_dt = parse_timestamp_flexible(agg.get("peak", {}).get("timestamp", ""))
            if other_peak_dt is None:
                continue
            if instance_dt < other_peak_dt < peak_dt:
                count += 1
    peak_meta["num_preceding_peaks_since_instance"] = count
    # Keep the requested typo-compatible feature name available in flattened analysis rows.
    peak_meta["num_presceding_peak"] = count
    return peak_meta


def augment_pair_peak_time_differences(pair):
    if not isinstance(pair, dict):
        return pair
    selected = pair.get("selected_peak_metadata")
    if not isinstance(selected, dict):
        return pair

    def peak_ts(signal):
        meta = selected.get(signal)
        return meta.get("peak_timestamp", "") if isinstance(meta, dict) else ""

    pairs = {
        "cgm_seg_peak_time_difference": ("cgm", "seg"),
        "cgm_sel_peak_time_difference": ("cgm", "sel"),
        "seg_sel_peak_time_difference": ("seg", "sel"),
    }
    for prefix, (left_signal, right_signal) in pairs.items():
        seconds = _timestamp_duration_seconds(peak_ts(left_signal), peak_ts(right_signal))
        if seconds is None:
            pair.pop(f"{prefix}_seconds", None)
            pair.pop(f"abs_{prefix}_seconds", None)
            continue
        pair[f"{prefix}_seconds"] = seconds
        pair[f"abs_{prefix}_seconds"] = abs(seconds)

    # Backward-compatible generic names for the selected two-signal pair.
    category = str(pair.get("category", ""))
    generic_prefix = None
    if category == "CGM_vs_SEG":
        generic_prefix = "cgm_seg_peak_time_difference"
    elif category == "CGM_vs_SEL":
        generic_prefix = "cgm_sel_peak_time_difference"
    elif category == "SEG_vs_SEL":
        generic_prefix = "seg_sel_peak_time_difference"
    if generic_prefix:
        pair["peak_time_difference_seconds"] = pair.get(f"{generic_prefix}_seconds")
        pair["abs_peak_time_difference_seconds"] = pair.get(f"abs_{generic_prefix}_seconds")

    delay_aliases = {
        "CGM_vs_SEG": ("seg", "cgm"),
        "CGM_vs_SEL": ("sel", "cgm"),
        "SEG_vs_SEL": ("sel", "seg"),
    }
    alias = delay_aliases.get(category)
    if alias:
        later_signal, earlier_signal = alias
        seconds = _timestamp_duration_seconds(peak_ts(earlier_signal), peak_ts(later_signal))
        signed_key = f"delay_{later_signal}_minus_{earlier_signal}_peak_seconds"
        abs_key = f"delay_abs_{later_signal}_{earlier_signal}_peak_seconds"
        if seconds is None:
            pair[signed_key] = None
            pair[abs_key] = None
        else:
            rounded_seconds = int(round(seconds))
            pair[signed_key] = rounded_seconds
            pair[abs_key] = abs(rounded_seconds)
    return pair


def clear_sel_from_triplet_pair(pair):
    if not isinstance(pair, dict):
        return pair
    selected = pair.get("selected_peak_metadata")
    if isinstance(selected, dict):
        selected.pop("sel", None)
    for key in (
        "sel_id",
        "triplet_complete",
        "sel_attached_at_utc",
        "cgm_sel_peak_time_difference_seconds",
        "abs_cgm_sel_peak_time_difference_seconds",
        "seg_sel_peak_time_difference_seconds",
        "abs_seg_sel_peak_time_difference_seconds",
        "delay_sel_minus_cgm_peak_seconds",
        "delay_abs_sel_cgm_peak_seconds",
        "delay_sel_minus_seg_peak_seconds",
        "delay_abs_sel_seg_peak_seconds",
    ):
        pair.pop(key, None)
    for key in list(pair.keys()):
        key_l = str(key).lower()
        if (
            key_l.startswith("cgm_sel")
            or key_l.startswith("abs_cgm_sel")
            or key_l.startswith("seg_sel")
            or key_l.startswith("abs_seg_sel")
            or "sel_minus" in key_l
            or "_sel_" in key_l
        ):
            pair.pop(key, None)
    return pair


def augment_peak_metadata_with_derivatives(rel_path, rows, peak_meta, cgm_points=None, aggregates=None):
    if not isinstance(peak_meta, dict):
        return peak_meta
    augment_peak_metadata_with_durations(peak_meta)
    augment_peak_metadata_with_preceding_peak_count(peak_meta, aggregates)
    peak_meta["point_values"] = point_value_features_for_peak(rows, peak_meta, cgm_points=cgm_points)
    peak_meta["trough1_to_trough2_auc"] = trough_to_trough_auc_features_for_peak(rows, peak_meta, cgm_points=cgm_points)
    peak_meta["interval_derivative_features"] = interval_derivative_features_for_peak(rel_path, rows, peak_meta, cgm_points=cgm_points)
    return peak_meta


def augment_peak_metadata_with_point_values(rows, peak_meta, cgm_points=None, aggregates=None):
    if not isinstance(peak_meta, dict):
        return peak_meta
    augment_peak_metadata_with_durations(peak_meta)
    augment_peak_metadata_with_preceding_peak_count(peak_meta, aggregates)
    peak_meta["point_values"] = point_value_features_for_peak(rows, peak_meta, cgm_points=cgm_points)
    peak_meta["trough1_to_trough2_auc"] = trough_to_trough_auc_features_for_peak(rows, peak_meta, cgm_points=cgm_points)
    return peak_meta


def local_processed_abs_path(rel_path):
    rel = safe_rel_path(rel_path)
    if not rel.startswith("annotated_data/series/"):
        rel = "annotated_data/series/" + rel
    return abs_from_rel(rel)


def load_rows_for_point_value_backfill(rel_path):
    abs_path = local_processed_abs_path(rel_path)
    rows = []
    with open(abs_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "Timestamp": row.get("Timestamp", ""),
                    "Delta_Channel2_minus_Channel1": row.get("Delta_Channel2_minus_Channel1"),
                    "Delta_Channel2_minus_Channel3": row.get("Delta_Channel2_minus_Channel3"),
                    "CGM": row.get("CGM"),
                }
            )
    return rows


def backfill_peak_pair_metadata_features():
    peak_state = load_peak_pair_state()
    cases = peak_state.get("cases", {}) if isinstance(peak_state, dict) else {}
    if not isinstance(cases, dict) or not cases:
        return 0
    updated = 0
    rows_cache = {}
    known_paths = None
    for case_path, case in cases.items():
        if not isinstance(case, dict):
            continue
        pairs = case.get("paired_features", [])
        if not isinstance(pairs, list) or not pairs:
            continue
        try:
            try:
                current_path = safe_rel_path(case_path)
                if current_path not in rows_cache:
                    rows_cache[current_path] = load_rows_for_point_value_backfill(current_path)
            except Exception:
                if known_paths is None:
                    known_paths = list_processed_files()
                current_path = remap_case_path_to_current(case_path, known_paths)
                if current_path not in rows_cache:
                    rows_cache[current_path] = load_rows_for_point_value_backfill(current_path)
            if current_path not in rows_cache:
                rows_cache[current_path] = load_rows_for_point_value_backfill(current_path)
            rows = rows_cache[current_path]
        except Exception:
            continue
        segment_state = load_segment_state()
        file_state = segment_state.get(current_path, {})
        if not isinstance(file_state, dict) or not file_state:
            file_state = segment_state.get(case_path, {})
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            selected = pair.get("selected_peak_metadata")
            if not isinstance(selected, dict):
                continue
            before_pair = json.dumps(
                {
                    "duration_fields": pair,
                },
                sort_keys=True,
                default=str,
            )
            category = str(pair.get("category", "") or "")
            file_state_for_agg = file_state if isinstance(file_state, dict) else {}
            aggregates = compute_aggregated_peaks_for_case(
                current_path,
                file_state_for_agg,
                rows,
                label_keys=aggregation_label_keys_for_pair_category(file_state_for_agg, category),
            )
            apply_pair_preceding_instances(current_path, pair)
            for signal in ("seg", "sel", "cgm"):
                meta = selected.get(signal)
                if isinstance(meta, dict):
                    before = json.dumps(
                        {
                            "durations": {
                                "duration_trough1_to_peak_seconds": meta.get("duration_trough1_to_peak_seconds"),
                                "duration_peak_to_trough2_seconds": meta.get("duration_peak_to_trough2_seconds"),
                                "duration_trough1_to_trough2_seconds": meta.get("duration_trough1_to_trough2_seconds"),
                            },
                            "point_values": meta.get("point_values", {}),
                            "trough1_to_trough2_auc": meta.get("trough1_to_trough2_auc", {}),
                            "preceding_instance": meta.get("preceding_instance"),
                            "pair_context": meta.get("pair_context"),
                            "label_panel": meta.get("label_panel"),
                            "num_preceding_peaks_since_instance": meta.get("num_preceding_peaks_since_instance"),
                            "num_presceding_peak": meta.get("num_presceding_peak"),
                        },
                        sort_keys=True,
                        default=str,
                    )
                    cgm_points = independent_cgm_series_for_case(current_path, rows) if str(meta.get("signal", "")).lower() == "cgm" else None
                    augment_peak_metadata_with_derivatives(current_path, rows, meta, cgm_points=cgm_points, aggregates=aggregates)
                    after = json.dumps(
                        {
                            "durations": {
                                "duration_trough1_to_peak_seconds": meta.get("duration_trough1_to_peak_seconds"),
                                "duration_peak_to_trough2_seconds": meta.get("duration_peak_to_trough2_seconds"),
                                "duration_trough1_to_trough2_seconds": meta.get("duration_trough1_to_trough2_seconds"),
                            },
                            "point_values": meta.get("point_values", {}),
                            "trough1_to_trough2_auc": meta.get("trough1_to_trough2_auc", {}),
                            "preceding_instance": meta.get("preceding_instance"),
                            "pair_context": meta.get("pair_context"),
                            "label_panel": meta.get("label_panel"),
                            "num_preceding_peaks_since_instance": meta.get("num_preceding_peaks_since_instance"),
                            "num_presceding_peak": meta.get("num_presceding_peak"),
                        },
                        sort_keys=True,
                        default=str,
                    )
                    if before != after:
                        updated += 1
            augment_pair_peak_time_differences(pair)
            after_pair = json.dumps(
                {
                    "duration_fields": pair,
                },
                sort_keys=True,
                default=str,
            )
            if before_pair != after_pair:
                updated += 1
    if updated:
        save_peak_pair_state(peak_state)
    return updated


def conversion_source_column(source):
    normalized = str(source or "").strip().lower()
    if normalized in ("sel", "lactate", "seb_lactate", "delta_channel2_minus_channel3"):
        return "Delta_Channel2_minus_Channel3", "SEL_lactate"
    return "Delta_Channel2_minus_Channel1", "SEG_glucose"


def concentration_from_delta(delta_value):
    if delta_value is None:
        return None
    try:
        return (float(delta_value) - 0.0188) / 0.00185
    except Exception:
        return None


def seg_converted_concentration_um(delta_value):
    value = _to_finite_float(delta_value)
    if value is None:
        return None
    return (value - 0.00453) / 0.00128


def sel_converted_concentration_um(delta_value):
    value = _to_finite_float(delta_value)
    if value is None:
        return None
    return (value - 0.00638) / 2.9145e-4


def load_range_label_state():
    payload = load_json(RANGE_LABELS_JSON, {"version": 1, "ranges": []})
    if not isinstance(payload, dict):
        payload = {"version": 1, "ranges": []}
    ranges = payload.get("ranges")
    if not isinstance(ranges, list):
        payload["ranges"] = []
    payload.setdefault("version", 1)
    changed = False
    for entry in payload.get("ranges", []):
        if not isinstance(entry, dict):
            continue
        if "smoothing_ratio" not in entry:
            entry["smoothing_ratio"] = 0.08
            changed = True
        if not entry.get("smoothed_values"):
            try:
                path = safe_rel_path(entry.get("path", ""))
                rows = load_processed_series(path)
                start = max(0, min(int(entry.get("start_index", 0)), len(rows) - 1))
                end = max(0, min(int(entry.get("end_index", start)), len(rows) - 1))
                if end < start:
                    start, end = end, start
                entry["smoothed_values"] = smoothed_range_rows_for_export(rows, start, end, smoothing_ratio=entry.get("smoothing_ratio", 0.08))
                changed = True
            except Exception:
                pass
        try:
            if refresh_range_selection_meals(entry):
                changed = True
        except Exception:
            pass
    if changed:
        save_json(RANGE_LABELS_JSON, payload)
        try:
            refresh_range_history_from_selection_state()
        except Exception:
            pass
    return payload


def save_range_label_state(payload):
    save_json(RANGE_LABELS_JSON, payload)


def load_range_label_history(path_filter=None):
    try:
        load_range_label_state()
    except Exception:
        pass
    payload = load_json(RANGE_LABEL_HISTORY_JSON, {"version": 1, "history": []})
    if not isinstance(payload, dict):
        payload = {"version": 1, "history": []}
    history = payload.get("history")
    if not isinstance(history, list):
        history = []
        payload["history"] = history
    payload.setdefault("version", 1)
    if not history and os.path.exists(RANGE_LABELS_JSON):
        source = load_json(RANGE_LABELS_JSON, {"ranges": []})
        migrated = []
        if isinstance(source, dict):
            for selection in source.get("ranges", []):
                if not isinstance(selection, dict):
                    continue
                migrated.append(
                    {
                        "id": selection.get("id", ""),
                        "saved_at_utc": selection.get("saved_at_utc", ""),
                        "path": selection.get("path", ""),
                        "start_timestamp": selection.get("start_timestamp", ""),
                        "end_timestamp": selection.get("end_timestamp", ""),
                        "smoothing_ratio": selection.get("smoothing_ratio"),
                        "meal_timestamps": selection.get("meal_timestamps", []),
                        "note": selection.get("note", ""),
                        "label_source": selection.get("label_source", ""),
                    }
                )
        if migrated:
            payload["history"] = migrated
            history = migrated
            save_range_label_history(payload)
    if path_filter:
        path = safe_rel_path(path_filter)
        return {
            "version": payload.get("version", 1),
            "history": [entry for entry in history if isinstance(entry, dict) and entry.get("path") == path],
        }
    return payload


def save_range_label_history(payload):
    save_json(RANGE_LABEL_HISTORY_JSON, payload)


def append_range_label_history(selection):
    if not isinstance(selection, dict):
        return None
    history_state = load_range_label_history()
    history = history_state.setdefault("history", [])
    entry = {
        "id": selection.get("id") or datetime.utcnow().strftime("range_%Y%m%d_%H%M%S_%f"),
        "saved_at_utc": selection.get("saved_at_utc") or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "path": selection.get("path", ""),
        "start_timestamp": selection.get("start_timestamp", ""),
        "end_timestamp": selection.get("end_timestamp", ""),
        "smoothing_ratio": selection.get("smoothing_ratio"),
        "meal_timestamps": selection.get("meal_timestamps", []),
        "note": selection.get("note", ""),
        "label_source": selection.get("label_source", ""),
    }
    history.append(entry)
    save_range_label_history(history_state)
    return entry


def range_entry_matches(entry, target_id="", path="", start_timestamp="", end_timestamp=""):
    if not isinstance(entry, dict):
        return False
    if target_id and entry.get("id") == target_id:
        return True
    if not path:
        return False
    return (
        entry.get("path") == path
        and entry.get("start_timestamp") == start_timestamp
        and entry.get("end_timestamp") == end_timestamp
    )


def delete_range_label_entry(target_id="", path="", start_timestamp="", end_timestamp=""):
    path = safe_rel_path(path) if path else ""
    removed_history = 0
    history_state = load_range_label_history()
    history = history_state.get("history", [])
    if not isinstance(history, list):
        history = []
    kept_history = []
    for entry in history:
        if range_entry_matches(entry, target_id=target_id, path=path, start_timestamp=start_timestamp, end_timestamp=end_timestamp):
            removed_history += 1
        else:
            kept_history.append(entry)
    history_state["history"] = kept_history
    save_range_label_history(history_state)

    removed_ranges = 0
    range_state = load_range_label_state()
    ranges = range_state.get("ranges", [])
    if not isinstance(ranges, list):
        ranges = []
    kept_ranges = []
    for entry in ranges:
        if range_entry_matches(entry, target_id=target_id, path=path, start_timestamp=start_timestamp, end_timestamp=end_timestamp):
            removed_ranges += 1
        else:
            kept_ranges.append(entry)
    range_state["ranges"] = kept_ranges
    save_range_label_state(range_state)
    return removed_history, removed_ranges


def range_rows_for_export(rows, start_index, end_index):
    out = []
    for idx in range(start_index, end_index + 1):
        row = rows[idx]
        seg_delta = _to_finite_float(row.get("Delta_Channel2_minus_Channel1"))
        sel_delta = _to_finite_float(row.get("Delta_Channel2_minus_Channel3"))
        out.append(
            {
                "index": idx,
                "Timestamp": row.get("Timestamp", ""),
                "Channel1": _to_finite_float(row.get("Channel1")),
                "Channel2": _to_finite_float(row.get("Channel2")),
                "Channel3": _to_finite_float(row.get("Channel3")),
                "Delta_Channel2_minus_Channel1": seg_delta,
                "Delta_Channel2_minus_Channel3": sel_delta,
                "SEG_converted_concentration_uM": seg_converted_concentration_um(seg_delta),
                "SEL_converted_concentration_uM": sel_converted_concentration_um(sel_delta),
                "CGM": _to_finite_float(row.get("CGM")),
            }
        )
    return out


def smooth_dense_for_range(values, ratio=0.08):
    try:
        ratio = max(0.0, min(0.30, float(ratio)))
    except Exception:
        ratio = 0.08
    if ratio <= 0:
        return list(values)
    if not values:
        return []
    window = max(5, int(math.ceil(len(values) * max(0.0, ratio))))
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = []
    for idx in range(len(values)):
        left = max(0, idx - half)
        right = min(len(values), idx + half + 1)
        known = [value for value in values[left:right] if value is not None]
        out.append((sum(known) / len(known)) if known else values[idx])
    return out


def smoothed_range_rows_for_export(rows, start_index, end_index, smoothing_ratio=0.08):
    cgm_smoothed = smooth_dense_for_range([_to_finite_float(row.get("CGM")) for row in rows], smoothing_ratio)
    seg_delta_smoothed = smooth_dense_for_range([_to_finite_float(row.get("Delta_Channel2_minus_Channel1")) for row in rows], smoothing_ratio)
    sel_delta_smoothed = smooth_dense_for_range([_to_finite_float(row.get("Delta_Channel2_minus_Channel3")) for row in rows], smoothing_ratio)
    out = []
    for idx in range(start_index, end_index + 1):
        seg_delta = seg_delta_smoothed[idx] if idx < len(seg_delta_smoothed) else None
        sel_delta = sel_delta_smoothed[idx] if idx < len(sel_delta_smoothed) else None
        out.append(
            {
                "index": idx,
                "Timestamp": rows[idx].get("Timestamp", ""),
                "CGM": cgm_smoothed[idx] if idx < len(cgm_smoothed) else None,
                "Delta_Channel2_minus_Channel1": seg_delta,
                "Delta_Channel2_minus_Channel3": sel_delta,
                "SEG_converted_concentration_uM": seg_converted_concentration_um(seg_delta),
                "SEL_converted_concentration_uM": sel_converted_concentration_um(sel_delta),
            }
        )
    return out


def meal_markers_for_range(path, start_dt, end_dt):
    if start_dt is None or end_dt is None:
        return []
    markers_by_ts = {}
    for inst in diet_instances_for_case(path):
        ts = inst.get("timestamp", "")
        marker_dt = parse_timestamp_flexible(ts)
        if marker_dt is None or marker_dt < start_dt or marker_dt > end_dt:
            continue
        markers_by_ts[ts] = {
            "timestamp": ts,
            "label": inst.get("label", ts),
            "source": "diet_instances",
            "instance_id": inst.get("instance_id", ""),
            "meal_med": inst.get("meal_med", ""),
            "mode": inst.get("mode"),
            "calories": inst.get("calories"),
        }
    for marker in load_diet_markers(path):
        ts = marker.get("timestamp", "")
        marker_dt = parse_timestamp_flexible(ts)
        if marker_dt is None or marker_dt < start_dt or marker_dt > end_dt:
            continue
        if ts in markers_by_ts:
            existing = markers_by_ts[ts]
            existing.setdefault("raw_files", [])
            if marker.get("file"):
                existing["raw_files"].append(marker.get("file"))
            continue
        enriched = dict(marker)
        enriched["source"] = "diet_folder"
        markers_by_ts[ts] = enriched
    return [markers_by_ts[ts] for ts in sorted(markers_by_ts)]


def refresh_range_selection_meals(selection):
    if not isinstance(selection, dict):
        return False
    path = selection.get("path", "")
    start_dt = parse_timestamp_flexible(selection.get("start_timestamp", ""))
    end_dt = parse_timestamp_flexible(selection.get("end_timestamp", ""))
    if not path or start_dt is None or end_dt is None:
        return False
    markers = meal_markers_for_range(path, start_dt, end_dt)
    meal_timestamps = [marker.get("timestamp", "") for marker in markers]
    if selection.get("meal_timestamps") == meal_timestamps and selection.get("meal_markers") == markers:
        return False
    selection["meal_timestamps"] = meal_timestamps
    selection["meal_markers"] = markers
    return True


def refresh_range_history_from_selection_state():
    if not os.path.exists(RANGE_LABEL_HISTORY_JSON):
        return False
    history_state = load_json(RANGE_LABEL_HISTORY_JSON, {"version": 1, "history": []})
    range_state = load_json(RANGE_LABELS_JSON, {"ranges": []})
    history = history_state.get("history", []) if isinstance(history_state, dict) else []
    ranges = range_state.get("ranges", []) if isinstance(range_state, dict) else []
    if not isinstance(history, list) or not isinstance(ranges, list):
        return False
    by_id = {entry.get("id"): entry for entry in ranges if isinstance(entry, dict) and entry.get("id")}
    changed = False
    for entry in history:
        if not isinstance(entry, dict):
            continue
        source = by_id.get(entry.get("id"))
        if not source:
            continue
        meal_timestamps = source.get("meal_timestamps", [])
        if entry.get("meal_timestamps") != meal_timestamps:
            entry["meal_timestamps"] = meal_timestamps
            changed = True
    if changed:
        save_json(RANGE_LABEL_HISTORY_JSON, history_state)
    return changed


def save_range_label_selection(rel_path, start_index, end_index, label_source="range_plot", note="", smoothing_ratio=0.08):
    path = safe_rel_path(rel_path)
    rows = load_processed_series(path)
    if not rows:
        raise ValueError("No rows found for selected file.")
    start = max(0, min(int(start_index), len(rows) - 1))
    end = max(0, min(int(end_index), len(rows) - 1))
    if end < start:
        start, end = end, start
    try:
        smoothing_ratio = max(0.0, min(0.30, float(smoothing_ratio)))
    except Exception:
        smoothing_ratio = 0.08
    start_ts = rows[start].get("Timestamp", "")
    end_ts = rows[end].get("Timestamp", "")
    start_dt = parse_timestamp_flexible(start_ts)
    end_dt = parse_timestamp_flexible(end_ts)
    markers = meal_markers_for_range(path, start_dt, end_dt)
    selection = {
        "id": datetime.utcnow().strftime("range_%Y%m%d_%H%M%S_%f"),
        "saved_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "path": path,
        "label_source": str(label_source or "range_plot"),
        "note": str(note or ""),
        "start_index": start,
        "end_index": end,
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "smoothing_ratio": smoothing_ratio,
        "meal_timestamps": [marker.get("timestamp", "") for marker in markers],
        "meal_markers": markers,
        "values": range_rows_for_export(rows, start, end),
        "smoothed_values": smoothed_range_rows_for_export(rows, start, end, smoothing_ratio=smoothing_ratio),
        "formulas": {
            "SEG_converted_concentration_uM": "(Delta_Channel2_minus_Channel1 - 0.00453) / 0.00128",
            "SEL_converted_concentration_uM": "(Delta_Channel2_minus_Channel3 - 0.00638) / 2.9145e-4",
        },
    }
    state = load_range_label_state()
    ranges = state.setdefault("ranges", [])
    ranges.append(selection)
    save_range_label_state(state)
    append_range_label_history(selection)
    return selection, state




def converted_rows_for_export(rel_path, source="seg", smoothing_ratio=0.08):
    path = safe_rel_path(rel_path)
    rows = load_processed_series(path)
    column, label = conversion_source_column(source)
    ratio = max(0.0, min(0.30, float(smoothing_ratio)))
    raw_delta = []
    for row in rows:
        value = row.get(column)
        if value is None:
            raw_delta.append(None)
            continue
        try:
            raw_delta.append(float(value))
        except Exception:
            raw_delta.append(None)
    smoothed = moving_average([v for v in raw_delta if v is not None], ratio) if ratio > 0 else [v for v in raw_delta if v is not None]
    smoothed_full = []
    ptr = 0
    for value in raw_delta:
        if value is None:
            smoothed_full.append(None)
        else:
            smoothed_full.append(smoothed[ptr] if ptr < len(smoothed) else value)
            ptr += 1
    out = []
    for idx, row in enumerate(rows):
        out.append(
            {
                "Timestamp": row.get("Timestamp", ""),
                "source": label,
                "source_delta": raw_delta[idx],
                "smoothed_delta": smoothed_full[idx],
                "concentration": concentration_from_delta(smoothed_full[idx]),
                "CGM": row.get("CGM"),
            }
        )
    return out, label, ratio


def export_converted_time_series(rel_path, source="seg", smoothing_ratio=0.08):
    path = safe_rel_path(rel_path)
    rows, label, ratio = converted_rows_for_export(path, source=source, smoothing_ratio=smoothing_ratio)
    suffix = path
    marker = "annotated_data/series/"
    if marker in suffix:
        suffix = suffix.split(marker, 1)[1]
    stem, _ext = os.path.splitext(suffix)
    out_rel = os.path.join("converted_time_series", stem + f"__{label}__converted.csv").replace(os.sep, "/")
    out_abs = os.path.join(ALIGNMENT_DIR, out_rel)
    ensure_dir(os.path.dirname(out_abs))
    with open(out_abs, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Timestamp",
                "source",
                "smoothing_ratio",
                "source_delta",
                "smoothed_delta",
                "concentration",
                "CGM",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Timestamp": row["Timestamp"],
                    "source": row["source"],
                    "smoothing_ratio": f"{ratio:.4f}",
                    "source_delta": "" if row["source_delta"] is None else f"{row['source_delta']:.8f}",
                    "smoothed_delta": "" if row["smoothed_delta"] is None else f"{row['smoothed_delta']:.8f}",
                    "concentration": "" if row["concentration"] is None else f"{row['concentration']:.8f}",
                    "CGM": "" if row["CGM"] is None else row["CGM"],
                }
            )
    return out_rel, len(rows), label, ratio


def cgm_day_ranges_from_original(rel_path, max_days=3):
    rel_path = safe_rel_path(rel_path)
    parts = rel_path.split("/")
    pid = None
    for part in parts:
        if re.match(r"^P\d{1,3}$", part, re.IGNORECASE):
            pid = f"P{int(part[1:]):03d}"
            break
    if not pid:
        return []
    participant_dir = os.path.join(RAW_ORIGINAL_DIR, pid)
    if not os.path.isdir(participant_dir):
        return []

    out = []
    day_dirs = []
    for name in os.listdir(participant_dir):
        m = re.match(r"^Day(\d{1,2})$", name, re.IGNORECASE)
        if not m:
            continue
        day_dirs.append((int(m.group(1)), name))
    day_dirs.sort(key=lambda x: x[0])
    for day_num, day_name in day_dirs[: max(1, int(max_days))]:
        day_abs = os.path.join(participant_dir, day_name)
        csv_candidates = [
            n
            for n in os.listdir(day_abs)
            if n.lower().endswith(".csv") and "annotated" not in n.lower()
        ]
        if not csv_candidates:
            continue
        csv_candidates.sort()
        raw_csv_abs = os.path.join(day_abs, csv_candidates[0])
        raw_rel = os.path.relpath(raw_csv_abs, ROOT_DIR).replace(os.sep, "/")
        rows = load_tool1_series(raw_rel)
        vals = []
        day_date = None
        for row in rows:
            ts = str(row.get("Timestamp", "")).strip()
            if len(ts) >= 10 and day_date is None:
                day_date = ts[:10]
            cgm = row.get("CGM")
            if cgm is None:
                continue
            try:
                vals.append(float(cgm))
            except Exception:
                continue
        if not vals:
            continue
        out.append(
            {
                "label": f"Day {day_num}",
                "date": (day_date or ""),
                "min": min(vals),
                "max": max(vals),
            }
        )
    return out


def parse_timestamp_flexible(text):
    s = str(text or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def raw_original_time_window(rel_path):
    rel_path = safe_rel_path(rel_path)
    parts = rel_path.split("/")
    pid = None
    day = None
    fname = None
    for i, part in enumerate(parts):
        if re.match(r"^P\d{1,3}$", part, re.IGNORECASE):
            pid = f"P{int(part[1:]):03d}"
        if re.match(r"^Day\d{1,2}$", part, re.IGNORECASE):
            day = part
            if i + 1 < len(parts):
                fname = parts[i + 1]
    if not pid or not day:
        return None
    day_dir = os.path.join(RAW_ORIGINAL_DIR, pid, day)
    if not os.path.isdir(day_dir):
        return None
    candidate = os.path.join(day_dir, fname or "")
    if not fname or not os.path.exists(candidate):
        csvs = sorted(
            n for n in os.listdir(day_dir)
            if n.lower().endswith(".csv") and "annotated" not in n.lower()
        )
        if not csvs:
            return None
        candidate = os.path.join(day_dir, csvs[0])
    raw_rel = os.path.relpath(candidate, ROOT_DIR).replace(os.sep, "/")
    rows = load_tool1_series(raw_rel)
    if not rows:
        return None
    return {"start": rows[0]["Timestamp"], "end": rows[-1]["Timestamp"]}


def organized_day_dir_from_rel_path(rel_path):
    source = cgm_features.source_rel_from_processed_path(rel_path)
    parts = safe_rel_path(source).split("/")
    if parts[:2] != ["raw_data", "sensor"] or len(parts) < 5:
        return None
    return os.path.join(ROOT_DIR, *parts[:4])


def load_diet_markers(rel_path):
    day_dir = organized_day_dir_from_rel_path(rel_path)
    if not day_dir:
        return []
    diet_dir = os.path.join(day_dir, "Diet")
    if not os.path.isdir(diet_dir):
        return []
    markers = []
    for name in sorted(os.listdir(diet_dir)):
        match = DIET_TS_RE.search(name)
        if not match:
            continue
        try:
            dt = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        markers.append(
            {
                "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "label": match.group(1),
                "file": name,
            }
        )
    return markers


def page_html():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Alignment Dashboard</title>
  <style>
    body { font-family: Helvetica, Arial, sans-serif; margin: 0; background: #f6f4ee; color: #222; }
    .layout { display: grid; grid-template-columns: 360px minmax(720px, 1fr) 360px; height: 100vh; }
    .sidebar { border-right: 1px solid #ddd; padding: 18px; background: #fffdf9; display: grid; grid-template-rows: auto minmax(0, 1fr); gap: 8px; overflow: hidden; }
    .sidebar-top { overflow-y: auto; padding-right: 4px; }
    .sidebar-bottom { min-height: 0; display: grid; grid-template-rows: auto minmax(0, 1fr); gap: 8px; overflow: hidden; padding-right: 4px; }
    .file-list-header { position: sticky; top: 0; z-index: 2; background: #fffdf9; border-bottom: 1px solid #eee4d7; padding-bottom: 8px; }
    .file-list-header h3 { margin: 0 0 8px 0; }
    #dataset-list { min-height: 0; overflow-y: auto; padding-right: 4px; }
    .center-panel { padding: 18px; overflow-y: auto; }
    .panel-controls { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 8px; }
    .panel-status { font-size: 12px; color: #666; margin-top: 6px; }
    .right-panel { border-left: 1px solid #ddd; padding: 18px; overflow-y: auto; background: #fffdf9; }
    .card { background: white; border: 1px solid #ddd; border-radius: 14px; padding: 14px; margin-bottom: 14px; }
    .dataset-item { border: 1px solid #ddd; border-radius: 10px; padding: 10px; margin-bottom: 8px; background: white; }
    .dataset-item.active { border-color: #173f5f; box-shadow: inset 0 0 0 1px #173f5f; }
    .toolbar { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
    button { padding: 8px 12px; border-radius: 9px; border: 1px solid #bbb; background: white; cursor: pointer; font-size: 14px; }
    .primary { background: #173f5f; color: white; border-color: #173f5f; }
    canvas { width: 100%; border: 1px solid #ddd; border-radius: 12px; background: white; display: block; }
    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }
    .stat { border: 1px solid #e4dfd4; border-radius: 10px; padding: 10px; background: #fcfbf7; }
    .stat .label { font-size: 12px; color: #666; }
    .stat .value { font-size: 18px; margin-top: 4px; color: #173f5f; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 8px; border-bottom: 1px solid #eee; text-align: left; }
    th { background: #faf8f2; }
    .small { color: #666; font-size: 12px; }
    .checkline { display: flex; align-items: center; gap: 8px; margin-top: 6px; font-size: 12px; color: #444; }
    .section-row th { background: #f1ede3; color: #173f5f; font-weight: 700; text-transform: uppercase; letter-spacing: 0.4px; font-size: 11px; }
    .corr-strong { color: #1b6f3a; font-weight: 700; }
    .corr-mid { color: #8a6a00; font-weight: 700; }
    .corr-weak { color: #8b2f2f; font-weight: 700; }
    .log-list { max-height: 220px; overflow-y: auto; border: 1px solid #eee; border-radius: 10px; padding: 8px; background: #fff; }
    .log-row { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 12px; }
    .log-row.active { border: 1px solid #173f5f; border-radius: 8px; padding: 4px; background: #eef5fb; }
    .tiny-btn { font-size: 11px; padding: 4px 7px; border-radius: 7px; }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <section class="sidebar-top">
        <h1>Alignment Dashboard</h1>
        <p>Cohort-level alignment review for processed delta files. Use drag selection independently on each panel to inspect alignment segments.</p>
        <h3>Time Series Controls</h3>
        <div class="card">
          <div class="checkline">
            <label for="lag-minutes-input">Lag search window (minutes)</label>
            <input id="lag-minutes-input" type="number" min="0" max="180" step="1" value="30" style="width:80px;" />
            <button id="apply-lag">Apply Lag Setting</button>
          </div>
          <div class="checkline">
            <label for="smooth-ratio-input">Smoothing extent (0-0.30)</label>
            <input id="smooth-ratio-input" type="number" min="0" max="0.30" step="0.01" value="0.08" style="width:80px;" />
            <button id="apply-smoothing">Apply Smoothing</button>
          </div>
          <div class="small" id="status"></div>
        </div>
        <h3>Annotation Storage</h3>
        <div class="card">
          <div class="small">Project snapshot/version-control workflow is disabled to avoid accidental overwrite.</div>
          <div class="small">All annotations now auto-save into <code>annotated_data/alignment_annotations.json</code>.</div>
          <div class="checkline"><button id="open-analysis-results" type="button">Open Analysis Results</button></div>
        </div>
      </section>
      <section class="sidebar-bottom">
        <div class="file-list-header">
          <h3>File List</h3>
          <label class="checkline" for="file-label-filter">
            <span>Show</span>
            <select id="file-label-filter" style="width:100%;">
              <option value="all">All files</option>
              <option value="labeled">Files with labels</option>
              <option value="unlabeled">Files without labels</option>
            </select>
          </label>
          <div class="small" id="file-filter-summary"></div>
        </div>
        <div id="dataset-list"></div>
      </section>
    </aside>
    <main class="center-panel">
      <div class="card">
        <h2 id="dataset-title">No dataset loaded</h2>
        <div class="small" id="dataset-subtitle"></div>
      </div>
      <div class="card">
        <h3>Raw Channels (Before Delta Calculation)</h3>
        <canvas id="plot-raw" width="1280" height="440"></canvas>
        <div class="panel-controls">
          <button id="reset-raw-zoom">Reset Raw Zoom</button>
          <div class="small" id="raw-hover-detail">Drag a box to zoom. Hover to inspect nearest point.</div>
        </div>
        <div class="small">This panel shows Channel1, Channel2, and Channel3 directly from the loaded processed rows.</div>
      </div>
      <div class="card">
        <h3>SEB Glu vs CGM</h3>
        <canvas id="plot-glu-cgm" width="1280" height="520"></canvas>
        <div class="panel-controls">
          <select id="new-label-name-glu-cgm" style="width:140px;">
            <option>CGM peak</option>
            <option>CGM trough</option>
            <option>SEG peak</option>
            <option>SEG trough</option>
            <option>SEL peak</option>
            <option>SEL trough</option>
          </select>
          <button id="add-label-glu-cgm">Add Label</button>
          <button id="set-start-glu-cgm">Set Start Marker</button>
          <button id="set-end-glu-cgm">Set End Marker</button>
          <button id="add-region-glu-cgm" class="primary">Add Region</button>
          <button id="clear-regions-glu-cgm">Clear Regions</button>
        </div>
        <div class="panel-status" id="panel-status-glu-cgm"></div>
        <div class="small">Click to place cursor. Set start/end markers, then add region. You can drag marker toggles at the bottom.</div>
      </div>
      <div class="card">
        <h3>SEB Glu vs Lac</h3>
        <canvas id="plot-glu-lac" width="1280" height="520"></canvas>
        <div class="panel-controls">
          <select id="new-label-name-glu-lac" style="width:140px;">
            <option>CGM peak</option>
            <option>CGM trough</option>
            <option>SEG peak</option>
            <option>SEG trough</option>
            <option>SEL peak</option>
            <option>SEL trough</option>
          </select>
          <button id="add-label-glu-lac">Add Label</button>
          <button id="set-start-glu-lac">Set Start Marker</button>
          <button id="set-end-glu-lac">Set End Marker</button>
          <button id="add-region-glu-lac" class="primary">Add Region</button>
          <button id="clear-regions-glu-lac">Clear Regions</button>
        </div>
        <div class="panel-status" id="panel-status-glu-lac"></div>
        <div class="small" id="selection-label">Click to place cursor. Set start/end markers, then add region. You can drag marker toggles at the bottom.</div>
      </div>
      <div class="card">
        <h3>Converted Concentration vs CGM</h3>
        <canvas id="plot-converted-cgm" width="1280" height="520"></canvas>
        <div class="small" id="converted-hover-detail">Hover over the plot to inspect converted concentration and CGM values.</div>
        <div class="small">Dual-axis view of converted concentration with CGM over the same datetime axis.</div>
      </div>
      <div class="card">
        <h3>Range Labeling: CGM / SEG uM / SEL uM</h3>
        <canvas id="plot-range-cgm" width="1280" height="300"></canvas>
        <canvas id="plot-range-seg" width="1280" height="300" style="margin-top:10px;"></canvas>
        <canvas id="plot-range-sel" width="1280" height="300" style="margin-top:10px;"></canvas>
        <div class="panel-controls">
          <button id="clear-range-selection">Clear Range</button>
          <button id="save-range-selection" class="primary">Save Range JSON</button>
          <input id="range-label-note" type="text" placeholder="Optional note" style="min-width:240px; flex:1;" />
        </div>
        <div class="panel-status" id="range-label-status">Drag across any one of the three plots to select a shared time range. Display follows the smoothing extent control.</div>
      </div>
    </main>
    <aside class="right-panel">
      <div class="card">
        <h3>Delta Conversion</h3>
        <div class="checkline">
          <label for="conversion-source" style="min-width:90px;">Delta source</label>
          <select id="conversion-source" style="width:100%;">
            <option value="seg">SEG glucose delta (Ch2-Ch1)</option>
            <option value="sel">SEL lactate delta (Ch2-Ch3)</option>
          </select>
        </div>
        <div class="checkline">
          <label for="conversion-smoothing" style="min-width:90px;">Smoothing</label>
          <input id="conversion-smoothing" type="number" min="0" max="0.30" step="0.01" value="0.08" style="width:80px;" />
          <button id="apply-conversion">Apply</button>
        </div>
        <label class="checkline"><input id="export-conversion-toggle" type="checkbox" /> Export converted time series on apply</label>
        <label class="checkline"><input id="show-negative-regions-toggle" type="checkbox" checked /> Show negative concentration regions</label>
        <div class="small" id="conversion-status">Preview only. Export toggle is off.</div>
      </div>
      <div class="card">
        <h3>Selected Segment Summary</h3>
        <label class="checkline"><input id="enable-segment-summary" type="checkbox" /> Enable segment summary calculation</label>
        <div class="small" id="segment-summary-status">Summary disabled. Turn on the switch to calculate.</div>
        <table>
          <tbody id="segment-table"></tbody>
        </table>
      </div>
      <div class="card">
        <h3>Region Logs (Current Case)</h3>
        <div class="log-list" id="region-log-list"></div>
      </div>
      <div class="card">
        <h3>Range Label History</h3>
        <div class="small">Reloads saved start/end timestamps for the current file without loading saved range values.</div>
        <div class="log-list" id="range-label-history-list"></div>
      </div>
      <div class="card">
        <h3>Labeling Mode</h3>
        <label class="checkline"><input id="label-mode-glu-cgm" type="checkbox" checked /> Label SEB Glu vs CGM</label>
        <label class="checkline"><input id="label-mode-glu-lac" type="checkbox" /> Label SEB Glu vs Lac</label>
        <div class="small" id="label-mode-status">Active labeling lane: SEB Glu vs CGM.</div>
        <div class="small" style="margin:10px 0 8px;">Display Filters</div>
        <label class="checkline"><input id="show-labels-toggle" type="checkbox" checked /> Show labels</label>
        <label class="checkline"><input id="show-label-cgm" type="checkbox" checked /> CGM</label>
        <label class="checkline"><input id="show-label-seg" type="checkbox" checked /> SEG</label>
        <label class="checkline"><input id="show-label-sel" type="checkbox" checked /> SEL</label>
        <label class="checkline"><input id="show-label-peak" type="checkbox" checked /> Peak</label>
        <label class="checkline"><input id="show-label-trough" type="checkbox" checked /> Trough</label>
        <div class="small" style="margin:10px 0 8px;">Glu-Lac View</div>
        <label class="checkline"><input id="glu-lac-split-view-toggle" type="checkbox" /> Split top/bottom view</label>
      </div>
      <div class="card mode-panel" data-mode-panel="glu_cgm">
        <h3>Label Logs: Glu-CGM</h3>
        <div class="log-list" id="label-log-list-glu-cgm"></div>
      </div>
      <div class="card mode-panel" data-mode-panel="glu_lac" style="display:none;">
        <h3>Label Logs: Glu-Lac</h3>
        <div class="log-list" id="label-log-list-glu-lac"></div>
      </div>
      <div class="card mode-panel" data-mode-panel="glu_cgm">
        <h3>Peak Pair Features: Glu-CGM</h3>
        <div class="small">Uses Glu-CGM peak/trough labels only.</div>
        <div class="checkline"><button id="refresh-peak-aggregates-glu-cgm">Refresh Aggregates</button></div>
        <div class="small" id="peak-agg-summary-glu-cgm"></div>
        <div class="checkline"><label style="min-width:90px;">SEG</label><select id="pair-seg-glu-cgm" style="width:100%;"></select></div>
        <div class="checkline"><label style="min-width:90px;">CGM</label><select id="pair-cgm-glu-cgm" style="width:100%;"></select></div>
        <div class="checkline"><button id="save-peak-pair-glu-cgm">Save Glu-CGM Pair</button></div>
        <div class="log-list" id="peak-pair-list-glu-cgm"></div>
      </div>
      <div class="card mode-panel" data-mode-panel="glu_lac" style="display:none;">
        <h3>Triplet Features: Add SEL</h3>
        <div class="small">Choose a fixed CGM-SEG peak instance, then attach the matching SEL aggregate. SEG guides are displayed as fixed intervals.</div>
        <div class="checkline"><button id="refresh-peak-aggregates-glu-lac">Refresh Aggregates</button></div>
        <div class="small" id="peak-agg-summary-glu-lac"></div>
        <div class="checkline"><label style="min-width:90px;">CGM-SEG</label><select id="pair-seg-glu-lac" style="width:100%;"></select></div>
        <div class="checkline"><label style="min-width:90px;">SEL</label><select id="pair-sel-glu-lac" style="width:100%;"></select></div>
        <div class="checkline"><button id="save-peak-pair-glu-lac">Attach SEL to Triplet</button></div>
        <div class="log-list" id="peak-pair-list-glu-lac"></div>
      </div>
    </aside>
  </div>
  <script>
    const state = {
      datasets: [],
      currentIndex: 0,
      loaded: new Map(),
      savedSegments: {},
      pendingSegments: { glu_cgm: null, glu_lac: null },
      regionPairs: { glu_cgm: [], glu_lac: [] },
      labelPoints: { glu_cgm: [], glu_lac: [] },
      markerIndex: {
        glu_cgm: { start: null, end: null },
        glu_lac: { start: null, end: null }
      },
      cursorIndex: { glu_cgm: 0, glu_lac: 0 },
      cursorTimeMs: { glu_cgm: null, glu_lac: null },
      activePanel: 'glu_cgm',
      fileLabelFilter: 'all',
      smoothRatio: 0.08,
      renderCache: new Map(),
      dragRafPending: false,
      markerDrag: { panel: null, kind: null },
      labelDrag: { panel: null, index: null },
      lagMinutes: 30,
      segmentSummaryEnabled: false,
      segLabelSync: false,
      labelFilter: {
        show: true,
        cgm: true,
        seg: true,
        sel: true,
        peak: true,
        trough: true
      },
      peakAgg: [],
      peakAggByPanel: { glu_cgm: [], glu_lac: [] },
      peakPairs: [],
      selectedLabelRef: { panel: null, index: null },
      selectedPeakPairIndex: { glu_cgm: null, glu_lac: null },
      rawZoom: { startMs: null, endMs: null },
      rawHoverIndex: null,
      rawDrag: { active: false, startX: null, currentX: null },
      conversionSource: 'seg',
      conversionSmoothing: 0.08,
      exportConversionOnApply: false,
      showNegativeRegions: true,
      convertedHoverIndex: null,
      rangeLabelSelection: { start: null, end: null },
      rangeLabelDrag: { active: false, source: null, startX: null, currentX: null },
      rangeLabelHoverIndex: null,
      rangeLabelHistory: [],
      gluLacSplitView: false,
      gluLacOverlayHeight: 520,
      gluLacSplitHeight: 980,
      bounds: { left: 70, right: 1230, top: 80, bottom: 440 }
    };
    const LABEL_OPTIONS = ['CGM peak', 'CGM trough', 'SEG peak', 'SEG trough', 'SEL peak', 'SEL trough'];
    const listEl = document.getElementById('dataset-list');
    const fileFilterSummaryEl = document.getElementById('file-filter-summary');
    const regionLogListEl = document.getElementById('region-log-list');
    const labelLogListEls = {
      glu_cgm: document.getElementById('label-log-list-glu-cgm'),
      glu_lac: document.getElementById('label-log-list-glu-lac')
    };
    const plotRaw = document.getElementById('plot-raw');
    const plotRawCtx = plotRaw.getContext('2d');
    const plotConvertedCgm = document.getElementById('plot-converted-cgm');
    const plotConvertedCgmCtx = plotConvertedCgm.getContext('2d');
    const rangeCanvases = {
      cgm: document.getElementById('plot-range-cgm'),
      seg: document.getElementById('plot-range-seg'),
      sel: document.getElementById('plot-range-sel')
    };
    const rangeContexts = {
      cgm: rangeCanvases.cgm.getContext('2d'),
      seg: rangeCanvases.seg.getContext('2d'),
      sel: rangeCanvases.sel.getContext('2d')
    };
    const plotGluCgm = document.getElementById('plot-glu-cgm');
    const plotGluCgmCtx = plotGluCgm.getContext('2d');
    const plotGluLac = document.getElementById('plot-glu-lac');
    const plotGluLacCtx = plotGluLac.getContext('2d');
    function syncGluLacCanvasHeight() {
      const targetHeight = state.gluLacSplitView ? state.gluLacSplitHeight : state.gluLacOverlayHeight;
      if (plotGluLac.height !== targetHeight) {
        plotGluLac.height = targetHeight;
      }
      plotGluLac.style.height = targetHeight + 'px';
    }
    function gluLacSplitBounds() {
      const base = state.bounds;
      const topMargin = base.top;
      const bottomMargin = 80;
      const gap = 70;
      const drawableBottom = Math.max(topMargin + 260, plotGluLac.height - bottomMargin);
      const panelHeight = Math.max(260, Math.floor((drawableBottom - topMargin - gap) / 2));
      return {
        top: { left: base.left, right: base.right, top: topMargin, bottom: topMargin + panelHeight },
        bottom: { left: base.left, right: base.right, top: topMargin + panelHeight + gap, bottom: topMargin + panelHeight + gap + panelHeight }
      };
    }
    function setStatus(message) { document.getElementById('status').textContent = message; }
    function setProjectStatus(message) { const el = document.getElementById('project-status'); if (el) el.textContent = message || ''; }
    function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
    function parseMs(value) { return new Date(value.replace(' ', 'T')).getTime(); }
    function minmax(values) {
      if (!values.length) return [];
      const low = Math.min(...values);
      const high = Math.max(...values);
      if (Math.abs(high - low) < 1e-12) return values.map(() => 0.5);
      return values.map((value) => (value - low) / (high - low));
    }
    function downsample(values, maxPoints = 900) {
      if (values.length <= maxPoints) return values.map((value, idx) => [idx, value]);
      const step = Math.max(1, Math.ceil(values.length / maxPoints));
      const out = [];
      for (let idx = 0; idx < values.length; idx += step) out.push([idx, values[idx]]);
      if (out[out.length - 1][0] !== values.length - 1) out.push([values.length - 1, values[values.length - 1]]);
      return out;
    }
    function downsampleSparse(values, maxPoints = 900) {
      const present = [];
      values.forEach((value, idx) => {
        if (value != null && !Number.isNaN(value)) present.push([idx, value]);
      });
      if (present.length <= maxPoints) return present;
      const step = Math.max(1, Math.ceil(present.length / maxPoints));
      const out = [];
      for (let idx = 0; idx < present.length; idx += step) out.push(present[idx]);
      if (out[out.length - 1][0] !== present[present.length - 1][0]) out.push(present[present.length - 1]);
      return out;
    }
    async function fetchJson(url, options) {
      const opts = options ? { ...options } : {};
      if (!opts.method || String(opts.method).toUpperCase() === 'GET') {
        opts.cache = 'no-store';
      }
      const response = await fetch(url, opts);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Request failed');
      return data;
    }
    function currentMeta() { return state.datasets[state.currentIndex] || null; }
    function currentRows() {
      const meta = currentMeta();
      return meta ? (state.loaded.get(meta.path)?.rows || []) : [];
    }
    function currentDietMarkers() {
      const meta = currentMeta();
      return meta ? (state.loaded.get(meta.path)?.diet_instances || []) : [];
    }
    function currentCgmSeries() {
      const meta = currentMeta();
      return meta ? (state.loaded.get(meta.path)?.cgm_series || []) : [];
    }
    function currentOriginalCgmDayRanges() {
      const meta = currentMeta();
      return meta ? (state.loaded.get(meta.path)?.cgm_day_ranges_original || []) : [];
    }
    function currentEffectiveSegment(panel) {
      const meta = currentMeta();
      if (!meta) return null;
      const pairs = state.regionPairs[panel] || [];
      if (pairs.length) {
        const last = pairs[pairs.length - 1];
        return { start_index: last.start_index, end_index: last.end_index };
      }
      const fileSegments = state.savedSegments[meta.path] || {};
      return state.pendingSegments[panel] || fileSegments[panel] || { start_index: 0, end_index: Math.max(0, meta.row_count - 1) };
    }
    function currentIncluded(panel) {
      const meta = currentMeta();
      if (!meta) return true;
      const fileSegments = state.savedSegments[meta.path] || {};
      const key = panel === 'glu_cgm' ? 'included_glu_cgm' : 'included_glu_lac';
      return fileSegments[key] !== false;
    }
    function regionKey(panel) {
      return panel === 'glu_cgm' ? 'region_pairs_glu_cgm' : 'region_pairs_glu_lac';
    }
    function labelKey(panel) {
      return panel === 'glu_cgm' ? 'labels_glu_cgm' : 'labels_glu_lac';
    }
    function savedRegionPairs(path, panel) {
      const fileSegments = state.savedSegments[path] || {};
      const key = regionKey(panel);
      const pairs = fileSegments[key];
      return Array.isArray(pairs) ? pairs : [];
    }
    function savedLabels(path, panel) {
      const fileSegments = state.savedSegments[path] || {};
      const key = labelKey(panel);
      let labels = fileSegments[key];
      if (!Array.isArray(labels)) return [];
      if (panel === 'glu_lac') {
        const cgmLabels = Array.isArray(fileSegments.labels_glu_cgm) ? fileSegments.labels_glu_cgm : [];
        const identity = (entry) => {
          const name = String(entry?.name || '');
          const upper = name.toUpperCase();
          const signal = upper.startsWith('SEG ') ? 'seg' : (upper.startsWith('CGM ') ? 'cgm' : (upper.startsWith('SEL ') ? 'sel' : 'other'));
          const kind = upper.includes('TROUGH') ? 'trough' : (upper.includes('PEAK') ? 'peak' : 'other');
          const tsMs = entry?.timestamp_ms == null || entry?.timestamp_ms === '' ? '' : String(Number(entry.timestamp_ms));
          return [signal, kind, name, Number(entry?.counter || 1), Number(entry?.index || 0), String(entry?.timestamp || ''), tsMs].join('|');
        };
        const cgmSegIds = new Set(
          cgmLabels
            .filter((entry) => String(entry?.name || '').toUpperCase().startsWith('SEG '))
            .map(identity)
        );
        labels = labels
          .map((entry, sourceIndex) => ({ entry, sourceIndex }))
          .filter(({ entry }) => {
            const isSeg = String(entry?.name || '').toUpperCase().startsWith('SEG ');
            const origin = String(entry?.origin_panel || '');
            return !(isSeg && cgmSegIds.has(identity(entry)) && origin !== 'glu_lac');
          });
      } else {
        labels = labels.map((entry, sourceIndex) => ({ entry, sourceIndex }));
      }
      const optionalNumber = (value) => {
        if (value === null || value === undefined || value === '') return null;
        const n = Number(value);
        return Number.isFinite(n) ? n : null;
      };
      return labels.map(({ entry, sourceIndex }, idx) => ({
        name: String(entry?.name || ('L' + (idx + 1))),
        index: Number(entry?.index || 0),
        counter: Math.max(1, Number(entry?.counter || (idx + 1))),
        source_index: Number(sourceIndex),
        origin_panel: entry?.origin_panel || '',
        timestamp: entry?.timestamp || '',
        timestamp_ms: optionalNumber(entry?.timestamp_ms),
        cgm_value: optionalNumber(entry?.cgm_value)
      }));
    }
    function refreshLabelStateForPanel(path, panel) {
      state.labelPoints[panel] = savedLabels(path, panel);
      if (state.segLabelSync) {
        const other = panel === 'glu_cgm' ? 'glu_lac' : 'glu_cgm';
        state.labelPoints[other] = savedLabels(path, other);
        drawCurrentPlot();
      } else {
        drawCurrentPlot(panel);
      }
    }
    function labelDisplay(entry, idxFallback = 1) {
      const counter = Math.max(1, Number(entry?.counter || idxFallback));
      return String(entry?.name || 'Label') + ' ' + counter;
    }
    function labelVisual(entry) {
      const raw = String(entry?.name || '');
      const upper = raw.toUpperCase();
      let color = '#6b7280';
      let signal = 'other';
      if (upper.startsWith('CGM ')) color = '#6c757d';
      else if (upper.startsWith('SEG ')) color = '#9d4edd';
      else if (upper.startsWith('SEL ')) color = '#3a86ff';
      if (upper.startsWith('CGM ')) signal = 'cgm';
      else if (upper.startsWith('SEG ')) signal = 'seg';
      else if (upper.startsWith('SEL ')) signal = 'sel';
      const kind = upper.includes('TROUGH') ? 'trough' : (upper.includes('PEAK') ? 'peak' : 'other');
      return { color, kind, signal };
    }
    function labelPassesFilter(vis) {
      const f = state.labelFilter || {};
      if (!f.show) return false;
      if (vis.signal === 'cgm' && !f.cgm) return false;
      if (vis.signal === 'seg' && !f.seg) return false;
      if (vis.signal === 'sel' && !f.sel) return false;
      if (vis.kind === 'peak' && !f.peak) return false;
      if (vis.kind === 'trough' && !f.trough) return false;
      return true;
    }
    function isCgmLabelName(name) {
      return String(name || '').trim().toUpperCase().startsWith('CGM ');
    }
    function labelTimeMs(entry, timeMs) {
      const explicit = (entry?.timestamp_ms === null || entry?.timestamp_ms === undefined || entry?.timestamp_ms === '') ? NaN : Number(entry.timestamp_ms);
      if (Number.isFinite(explicit)) return explicit;
      const parsed = parseMs(entry?.timestamp || '');
      if (!Number.isNaN(parsed)) return parsed;
      const cIdx = clamp(entry?.index ?? 0, 0, timeMs.length - 1);
      return timeMs[cIdx];
    }
    function drawStar(ctx, cx, cy, outerR, innerR, color) {
      ctx.save();
      ctx.beginPath();
      let angle = -Math.PI / 2;
      const step = Math.PI / 5;
      for (let i = 0; i < 10; i += 1) {
        const r = i % 2 === 0 ? outerR : innerR;
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        angle += step;
      }
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
      ctx.restore();
    }
    function drawSavedLabels(ctx, bounds, timeMs, xForTime, labels) {
      labels.forEach((entry, idx) => {
        const labelMs = labelTimeMs(entry, timeMs);
        if (!Number.isFinite(labelMs)) return;
        const x = xForTime(labelMs);
        const vis = labelVisual(entry);
        if (!labelPassesFilter(vis)) return;
        ctx.save();
        ctx.strokeStyle = vis.color;
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(x, bounds.top);
        ctx.lineTo(x, bounds.bottom);
        ctx.stroke();
        ctx.restore();
        if (vis.kind === 'trough') drawStar(ctx, x, bounds.bottom + 40, 6, 2.8, vis.color);
        else {
          ctx.fillStyle = vis.color;
          ctx.beginPath();
          ctx.arc(x, bounds.bottom + 40, 5, 0, Math.PI * 2);
          ctx.fill();
        }
        const name = labelDisplay(entry, idx + 1);
        ctx.save();
        ctx.translate(x + 3, bounds.bottom + 6);
        ctx.rotate(Math.PI / 2);
        ctx.fillStyle = vis.color;
        ctx.font = '11px Helvetica';
        ctx.textBaseline = 'top';
        ctx.fillText(name, 0, 0);
        ctx.restore();
      });
    }
    function labelOptionsHtml(selectedName) {
      return LABEL_OPTIONS.map((opt) => {
        const selected = opt === selectedName ? ' selected' : '';
        return '<option' + selected + '>' + opt + '</option>';
      }).join('');
    }
    async function loadFiles() {
      const data = await fetchJson('/api/files');
      state.datasets = data.datasets;
      state.savedSegments = data.saved_segments || {};
      state.currentIndex = 0;
      renderList();
      await loadCurrentDataset();
    }
    async function loadCurrentDataset() {
      const meta = currentMeta();
      if (!meta) return;
      if (!state.loaded.has(meta.path)) {
        const data = await fetchJson('/api/series?path=' + encodeURIComponent(meta.path));
        state.loaded.set(meta.path, data);
      }
      state.rawZoom = { startMs: null, endMs: null };
      state.rawHoverIndex = null;
      state.rawDrag = { active: false, startX: null, currentX: null };
      state.convertedHoverIndex = null;
      state.rangeLabelSelection = { start: null, end: null };
      state.rangeLabelDrag = { active: false, source: null, startX: null, currentX: null };
      state.rangeLabelHoverIndex = null;
      state.rangeLabelHistory = [];
      state.selectedPeakPairIndex = { glu_cgm: null, glu_lac: null };
      updateRawHoverDetail(null);
      updateConvertedHoverDetail(null);
      updateRangeLabelStatus();
      state.pendingSegments = { glu_cgm: null, glu_lac: null };
      state.regionPairs = {
        glu_cgm: savedRegionPairs(meta.path, 'glu_cgm').map((pair) => ({ start_index: pair.start_index, end_index: pair.end_index })),
        glu_lac: savedRegionPairs(meta.path, 'glu_lac').map((pair) => ({ start_index: pair.start_index, end_index: pair.end_index }))
      };
      state.labelPoints = {
        glu_cgm: savedLabels(meta.path, 'glu_cgm'),
        glu_lac: savedLabels(meta.path, 'glu_lac')
      };
      state.segLabelSync = false;
      state.markerIndex = {
        glu_cgm: { start: null, end: null },
        glu_lac: { start: null, end: null }
      };
      state.cursorIndex = { glu_cgm: 0, glu_lac: 0 };
      state.cursorTimeMs = { glu_cgm: null, glu_lac: null };
      setActivePanel('glu_cgm');
      document.getElementById('dataset-title').textContent = meta.path.split('/').slice(-3).join('/');
      document.getElementById('dataset-subtitle').textContent = meta.path + ' | Duration: ' + meta.duration_hms + ' | Rows: ' + meta.row_count + ' | Matched CGM: ' + meta.matched_cgm_points;
      state.renderCache.delete(meta.path + '|' + state.smoothRatio.toFixed(4));
      renderList();
      drawCurrentPlot();
      updateSegmentTable();
      updateRegionStatus();
      renderRegionLogs();
      renderLabelLogs();
      await loadRangeLabelHistory();
      await loadPeakFeatureData();
    }
    function panelLabel(panel) {
      return panel === 'glu_cgm' ? 'SEB Glu vs CGM' : 'SEB Glu vs Lac';
    }
    function updateModePanelVisibility() {
      document.querySelectorAll('[data-mode-panel]').forEach((el) => {
        el.style.display = el.getAttribute('data-mode-panel') === state.activePanel ? '' : 'none';
      });
    }
    function setActivePanel(panel) {
      if (panel !== 'glu_cgm' && panel !== 'glu_lac') return;
      const changed = state.activePanel !== panel;
      state.activePanel = panel;
      document.getElementById('label-mode-glu-cgm').checked = panel === 'glu_cgm';
      document.getElementById('label-mode-glu-lac').checked = panel === 'glu_lac';
      const status = document.getElementById('label-mode-status');
      if (status) status.textContent = 'Active labeling lane: ' + panelLabel(panel) + '.';
      updateModePanelVisibility();
      if (changed) renderPeakPairControls(panel);
    }
    function updateRegionStatus() {
      ['glu_cgm', 'glu_lac'].forEach((panel) => {
        const rows = currentRows();
        const pairs = state.regionPairs[panel] || [];
        const cursor = state.cursorIndex[panel];
        const startIdx = state.markerIndex[panel].start;
        const endIdx = state.markerIndex[panel].end;
        const cursorTs = rows[cursor]?.Timestamp || 'NA';
        const startTs = startIdx == null ? 'NA' : (rows[startIdx]?.Timestamp || 'NA');
        const endTs = endIdx == null ? 'NA' : (rows[endIdx]?.Timestamp || 'NA');
        let text = panelLabel(panel) + ' | Cursor: ' + cursorTs + ' (idx ' + cursor + ') | Regions: ' + pairs.length;
        text += ' | Start: ' + startTs + (startIdx == null ? '' : (' (idx ' + startIdx + ')'));
        text += ' | End: ' + endTs + (endIdx == null ? '' : (' (idx ' + endIdx + ')'));
        document.getElementById(panel === 'glu_cgm' ? 'panel-status-glu-cgm' : 'panel-status-glu-lac').textContent = text;
      });
    }
    function renderRegionLogs() {
      const meta = currentMeta();
      if (!meta) {
        regionLogListEl.innerHTML = '<div class="small">No case loaded.</div>';
        return;
      }
      const rows = currentRows();
      const buildPanelRows = (panel) => {
        const pairs = state.regionPairs[panel] || [];
        if (!pairs.length) return '<div class="small"><b>' + panelLabel(panel) + ':</b> no regions</div>';
        return pairs.map((pair, idx) => {
          const startTs = rows[pair.start_index]?.Timestamp || ('idx ' + pair.start_index);
          const endTs = rows[pair.end_index]?.Timestamp || ('idx ' + pair.end_index);
          return '<div class="log-row">' +
            '<span><b>' + panelLabel(panel) + '</b> #' + (idx + 1) + ': ' + pair.start_index + '-' + pair.end_index + '<br><span class="small">' + startTs + ' to ' + endTs + '</span></span>' +
            '<button class="tiny-btn" data-remove-panel="' + panel + '" data-remove-index="' + idx + '">Remove</button>' +
          '</div>';
        }).join('');
      };
      regionLogListEl.innerHTML = buildPanelRows('glu_cgm') + buildPanelRows('glu_lac');
      regionLogListEl.querySelectorAll('button[data-remove-panel]').forEach((button) => {
        button.addEventListener('click', async () => {
          const panel = button.getAttribute('data-remove-panel');
          const index = Number(button.getAttribute('data-remove-index'));
          const current = currentMeta();
          if (!current) return;
          const data = await fetchJson('/api/regions/remove', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: current.path, panel, region_index: index })
          });
          state.savedSegments = data.saved_segments || state.savedSegments;
          state.regionPairs[panel] = savedRegionPairs(current.path, panel).map((pair) => ({ start_index: pair.start_index, end_index: pair.end_index }));
          drawCurrentPlot(panel);
          updateSegmentTable();
          updateRegionStatus();
          renderRegionLogs();
          renderList();
          setStatus('Removed region #' + (index + 1) + ' from ' + panelLabel(panel) + '.');
        });
      });
    }
    function renderLabelLogs(panelFilter = null) {
      if (!panelFilter) {
        renderLabelLogs('glu_cgm');
        renderLabelLogs('glu_lac');
        return;
      }
      const labelLogListEl = labelLogListEls[panelFilter];
      if (!labelLogListEl) return;
      const meta = currentMeta();
      if (!meta) {
        labelLogListEl.innerHTML = '<div class="small">No case loaded.</div>';
        return;
      }
      const rows = currentRows();
      const entries = [];
      [panelFilter].forEach((panel) => {
        const labels = state.labelPoints[panel] || [];
        labels.forEach((entry, idx) => {
          entries.push({ panel, idx, entry, ts: rows[entry.index]?.Timestamp || ('idx ' + entry.index) });
        });
      });
      if (!entries.length) {
        labelLogListEl.innerHTML = '<div class="small">No labels.</div>';
        return;
      }
      entries.sort((a, b) => {
        const ai = Number(a.entry?.index || 0);
        const bi = Number(b.entry?.index || 0);
        return ai - bi;
      });
      labelLogListEl.innerHTML = entries.map((item) => {
        const panel = item.panel;
        const idx = item.idx;
        const entry = item.entry;
        const sourceIndex = Number.isFinite(Number(entry.source_index)) ? Number(entry.source_index) : idx;
        const ts = item.ts;
        const active = state.selectedLabelRef.panel === panel && Number(state.selectedLabelRef.index) === Number(idx);
        const rowId = 'label-row-' + panel + '-' + idx;
        return '<div class="log-row">' +
          '<div id="' + rowId + '" class="log-row' + (active ? ' active' : '') + '">' +
          '<span><b>' + labelDisplay(entry, idx + 1) + '</b>: <span class="small">' + ts + ' (idx ' + entry.index + ')</span></span>' +
          '<span>' +
            '<select data-label-panel="' + panel + '" data-label-index="' + sourceIndex + '" style="width:120px;">' + labelOptionsHtml(String(entry.name || 'CGM peak')) + '</select> ' +
            '<button class="tiny-btn" data-label-save-panel="' + panel + '" data-label-save-index="' + sourceIndex + '">Rename</button> ' +
            '<button class="tiny-btn" data-label-remove-panel="' + panel + '" data-label-remove-index="' + sourceIndex + '">Remove</button>' +
          '</span>' +
          '</div>' +
        '</div>';
      }).join('');
      labelLogListEl.querySelectorAll('div[id^="label-row-"]').forEach((row) => {
        row.addEventListener('click', () => {
          const id = row.id.replace('label-row-', '');
          const split = id.split('-');
          const panel = split[0] + '_' + split[1];
          const idx = Number(split[2] || 0);
          state.selectedLabelRef = { panel, index: idx };
          renderLabelLogs();
        });
      });
      labelLogListEl.querySelectorAll('button[data-label-save-panel]').forEach((button) => {
        button.addEventListener('click', async () => {
          const panel = button.getAttribute('data-label-save-panel');
          const index = Number(button.getAttribute('data-label-save-index'));
          const input = labelLogListEl.querySelector('select[data-label-panel="' + panel + '"][data-label-index="' + index + '"]');
          const name = (input?.value || 'CGM peak').trim();
          const current = currentMeta();
          if (!current) return;
          const data = await fetchJson('/api/labels/rename', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: current.path, panel, label_index: index, name })
          });
          state.savedSegments = data.saved_segments || state.savedSegments;
          refreshLabelStateForPanel(current.path, panel);
          renderLabelLogs(panel);
          renderList();
          setStatus('Renamed label #' + (index + 1) + ' on ' + panelLabel(panel) + '.');
        });
      });
      labelLogListEl.querySelectorAll('button[data-label-remove-panel]').forEach((button) => {
        button.addEventListener('click', async () => {
          const panel = button.getAttribute('data-label-remove-panel');
          const index = Number(button.getAttribute('data-label-remove-index'));
          const current = currentMeta();
          if (!current) return;
          const data = await fetchJson('/api/labels/remove', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: current.path, panel, label_index: index })
          });
          state.savedSegments = data.saved_segments || state.savedSegments;
          refreshLabelStateForPanel(current.path, panel);
          renderLabelLogs(panel);
          renderList();
          setStatus('Removed label #' + (index + 1) + ' from ' + panelLabel(panel) + '.');
        });
      });
    }
    function focusLabelLog(panel, index) {
      state.selectedLabelRef = { panel, index };
      renderLabelLogs();
      const row = document.getElementById('label-row-' + panel + '-' + index);
      if (row && row.scrollIntoView) row.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    function currentPeakAggPool(panel = state.activePanel) {
      return (state.peakAggByPanel && state.peakAggByPanel[panel]) || [];
    }
    function peakAggOptions(signal, panel = state.activePanel) {
      const options = ['<option value="">(none)</option>'];
      currentPeakAggPool(panel).filter((x) => x.signal === signal).forEach((x) => {
        options.push('<option value="' + x.id + '">' + x.id + ' @' + x.peak.index + '</option>');
      });
      return options.join('');
    }
    function pairControlIds(panel) {
      return {
        seg: 'pair-seg-' + panel.replace('_', '-'),
        sel: 'pair-sel-' + panel.replace('_', '-'),
        cgm: 'pair-cgm-' + panel.replace('_', '-')
      };
    }
    function pairCategoryForPanel(panel) {
      return 'CGM_vs_SEG';
    }
    function peakPairsForPanel(panel) {
      return (state.peakPairs || [])
        .map((item, index) => ({ item, index }))
        .filter(({ item }) => item && item.category === 'CGM_vs_SEG' && item.seg_id && item.cgm_id);
    }
    function fixedTripletOptions() {
      const options = ['<option value="">(choose CGM-SEG instance)</option>'];
      peakPairsForPanel('glu_lac').forEach(({ item, index }) => {
        const selState = item.sel_id ? 'SEL=' + item.sel_id : 'SEL not attached';
        const label = (item.name || ('CGM_vs_SEG_' + (index + 1))) + ' | SEG=' + (item.seg_id || 'NA') + ' | CGM=' + (item.cgm_id || 'NA') + ' | ' + selState;
        options.push('<option value="' + index + '">' + label + '</option>');
      });
      return options.join('');
    }
    function selectPeakPair(item, idx, panel = state.activePanel) {
      if (!item) return false;
      const ids = pairControlIds(panel);
      const segSel = document.getElementById(ids.seg);
      const selSel = document.getElementById(ids.sel);
      const cgmSel = document.getElementById(ids.cgm);
      const canSet = (selectEl, value) => !value || Array.from(selectEl.options).some((opt) => opt.value === value);
      if (!segSel) return false;
      if (panel === 'glu_lac') {
        if (!canSet(segSel, String(idx))) return false;
        if (selSel && !canSet(selSel, item.sel_id || '')) return false;
        segSel.value = String(idx);
        if (selSel) selSel.value = item.sel_id || '';
        state.selectedPeakPairIndex[panel] = idx;
        return true;
      }
      if (segSel && !canSet(segSel, item.seg_id || '')) return false;
      if (selSel && !canSet(selSel, item.sel_id || '')) return false;
      if (cgmSel && !canSet(cgmSel, item.cgm_id || '')) return false;
      if (segSel) segSel.value = item.seg_id || '';
      if (selSel) selSel.value = item.sel_id || '';
      if (cgmSel) cgmSel.value = item.cgm_id || '';
      state.selectedPeakPairIndex[panel] = idx;
      return true;
    }
    function autoSelectPeakPairIfNeeded(panel = state.activePanel) {
      const current = state.selectedPeakPairIndex[panel];
      const existing = current != null ? (state.peakPairs || [])[current] : null;
      if (existing && existing.category === pairCategoryForPanel(panel) && selectPeakPair(existing, current, panel)) return;
      state.selectedPeakPairIndex[panel] = null;
      for (const { item, index } of peakPairsForPanel(panel)) {
        if (selectPeakPair(item, index, panel)) return;
      }
    }
    function renderPeakPairList(panel = state.activePanel) {
      const el = document.getElementById('peak-pair-list-' + panel.replace('_', '-'));
      if (!el) return;
      const items = peakPairsForPanel(panel);
      if (!items.length) {
        state.selectedPeakPairIndex[panel] = null;
        el.innerHTML = '<div class="small">No paired peak features saved.</div>';
        return;
      }
      el.innerHTML = items.map(({ item: x, index: i }) => (
        '<div class="log-row item' + (state.selectedPeakPairIndex[panel] === i ? ' active' : '') + '" data-peak-pair-panel="' + panel + '" data-peak-pair-index="' + i + '">' +
        '<span><b>' + (x.name || ('pair_' + (i + 1))) + '</b><br><span class="small">cat=' + (x.category || 'NA') +
        ' | SEG=' + (x.seg_id || 'NA') + ' | SEL=' + (x.sel_id || 'NA') + ' | CGM=' + (x.cgm_id || 'NA') + '</span></span>' +
        (panel === 'glu_cgm' ? '<button class="tiny-btn" data-peak-pair-remove-panel="' + panel + '" data-peak-pair-remove-index="' + i + '">Delete</button>' : '') +
        (panel === 'glu_lac' && x.sel_id ? '<button class="tiny-btn" data-peak-pair-clear-sel-index="' + i + '">Remove SEL</button>' : '') +
        '</div>'
      )).join('');
      el.querySelectorAll('[data-peak-pair-index]').forEach((row) => {
        row.addEventListener('click', () => {
          const rowPanel = row.getAttribute('data-peak-pair-panel') || panel;
          const idx = Number(row.getAttribute('data-peak-pair-index') || -1);
          const item = (state.peakPairs || [])[idx];
          if (!item) return;
          setActivePanel(rowPanel);
          selectPeakPair(item, idx, rowPanel);
          renderPeakPairList(rowPanel);
          drawCurrentPlot();
          setStatus('Loaded peak pair feature: ' + (item.name || ('pair_' + (idx + 1))));
        });
      });
      el.querySelectorAll('[data-peak-pair-remove-index]').forEach((btn) => {
        btn.addEventListener('click', async (event) => {
          event.stopPropagation();
          const btnPanel = btn.getAttribute('data-peak-pair-remove-panel') || panel;
          const idx = Number(btn.getAttribute('data-peak-pair-remove-index') || -1);
          const meta = currentMeta();
          if (!meta || idx < 0) return;
          try {
            const data = await fetchJson('/api/peak/pairs/remove', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ path: meta.path, pair_index: idx })
            });
            state.peakPairs = data.paired_features || [];
            renderPeakPairList(btnPanel);
            drawCurrentPlot();
            setStatus('Deleted paired peak feature #' + (idx + 1) + '.');
          } catch (error) {
            setStatus('Failed to delete paired peak feature: ' + error.message);
          }
        });
      });
      el.querySelectorAll('[data-peak-pair-clear-sel-index]').forEach((btn) => {
        btn.addEventListener('click', async (event) => {
          event.stopPropagation();
          const idx = Number(btn.getAttribute('data-peak-pair-clear-sel-index') || -1);
          const meta = currentMeta();
          if (!meta || idx < 0) return;
          try {
            const data = await fetchJson('/api/peak/pairs/clear-sel', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ path: meta.path, pair_index: idx })
            });
            state.peakPairs = data.paired_features || [];
            renderPeakPairControls();
            drawCurrentPlot();
            setStatus('Removed SEL match from the selected triplet.');
          } catch (error) {
            setStatus('Failed to remove SEL match: ' + error.message);
          }
        });
      });
    }
    function renderPeakPairControls(panel = null) {
      if (!panel) {
        renderPeakPairControls('glu_cgm');
        renderPeakPairControls('glu_lac');
        return;
      }
      const ids = pairControlIds(panel);
      const segEl = document.getElementById(ids.seg);
      const selEl = document.getElementById(ids.sel);
      const cgmEl = document.getElementById(ids.cgm);
      if (segEl) segEl.innerHTML = panel === 'glu_lac' ? fixedTripletOptions() : peakAggOptions('seg', panel);
      if (selEl) selEl.innerHTML = peakAggOptions('sel', panel);
      if (cgmEl) cgmEl.innerHTML = peakAggOptions('cgm', panel);
      const pool = currentPeakAggPool(panel);
      const segN = panel === 'glu_lac' ? peakPairsForPanel('glu_lac').length : pool.filter((x) => x.signal === 'seg').length;
      const selN = pool.filter((x) => x.signal === 'sel').length;
      const cgmN = pool.filter((x) => x.signal === 'cgm').length;
      document.getElementById('peak-agg-summary-' + panel.replace('_', '-')).textContent = panel === 'glu_lac'
        ? 'Fixed CGM-SEG instances=' + segN + ', SEL aggregates from Glu-Lac labels=' + selN
        : 'Aggregates (' + panelLabel(panel) + ' labels only): SEG=' + segN + ', SEL=' + selN + ', CGM=' + cgmN;
      autoSelectPeakPairIfNeeded(panel);
      renderPeakPairList(panel);
    }
    async function loadPeakFeatureData() {
      const meta = currentMeta();
      if (!meta) return;
      const data = await fetchJson('/api/peak/features?path=' + encodeURIComponent(meta.path));
      state.peakAgg = data.aggregated_peaks || [];
      state.peakAggByPanel = data.aggregated_peaks_by_panel || { glu_cgm: state.peakAgg, glu_lac: state.peakAgg };
      state.peakPairs = data.paired_features || [];
      renderPeakPairControls();
    }
    function getRenderSeries(rows) {
      const meta = currentMeta();
      if (!meta) return null;
      const key = meta.path + '|' + state.smoothRatio.toFixed(4) + '|' + state.conversionSource + '|' + state.conversionSmoothing.toFixed(4);
      if (state.renderCache.has(key)) return state.renderCache.get(key);
      const norm = normalizedSeries(rows);
      const payload = {
        rawCh1: smoothDense(rows.map((row) => parseMaybeNumber(row.Channel1)), state.smoothRatio),
        rawCh2: smoothDense(rows.map((row) => parseMaybeNumber(row.Channel2)), state.smoothRatio),
        rawCh3: smoothDense(rows.map((row) => parseMaybeNumber(row.Channel3)), state.smoothRatio),
        gluRaw: smoothDense(rows.map((row) => parseMaybeNumber(row.Delta_Channel2_minus_Channel1)), state.smoothRatio),
        cgmRaw: rows.map((row) => parseMaybeNumber(row.CGM)),
        cgmPoints: currentCgmSeries()
          .map((row) => ({ ms: parseMs(row.Timestamp), value: parseMaybeNumber(row.CGM), timestamp: row.Timestamp }))
          .filter((point) => !Number.isNaN(point.ms) && point.value != null),
        convertedConc: convertedConcentrationSeries(rows),
        glucoseNorm: norm.glucose,
        lactateNorm: norm.lactate
      };
      state.renderCache.set(key, payload);
      return payload;
    }
    function selectedPeakAggBySignal(panel = state.activePanel) {
      const lookup = {};
      currentPeakAggPool(panel).forEach((x) => { lookup[x.id] = x; });
      const ids = pairControlIds(panel);
      const segId = document.getElementById(ids.seg)?.value || '';
      const selId = document.getElementById(ids.sel)?.value || '';
      const cgmId = document.getElementById(ids.cgm)?.value || '';
      return {
        seg: lookup[segId] || null,
        sel: lookup[selId] || null,
        cgm: lookup[cgmId] || null
      };
    }
    function drawPeakAggHighlight(ctx, bounds, timeMs, xForTime, agg, colorFill, colorPeak) {
      if (!agg || !agg.window) return;
      const l = clamp(Number(agg.window.start_index || 0), 0, timeMs.length - 1);
      const r = clamp(Number(agg.window.end_index || 0), 0, timeMs.length - 1);
      const p = clamp(Number((agg.peak && agg.peak.index) || 0), 0, timeMs.length - 1);
      const left = xForTime(timeMs[Math.min(l, r)]);
      const right = xForTime(timeMs[Math.max(l, r)]);
      ctx.fillStyle = colorFill;
      ctx.fillRect(left, bounds.top, Math.max(1, right - left), bounds.bottom - bounds.top);
      const peakX = xForTime(timeMs[p]);
      ctx.save();
      ctx.strokeStyle = colorPeak;
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 2]);
      ctx.beginPath();
      ctx.moveTo(peakX, bounds.top);
      ctx.lineTo(peakX, bounds.bottom);
      ctx.stroke();
      ctx.restore();
    }
    function drawFixedSegTripletGuides(ctx, bounds, timeMs, xForTime, options = {}) {
      const guides = peakPairsForPanel('glu_lac')
        .map(({ item, index }) => ({ item, index, seg: item?.selected_peak_metadata?.seg }))
        .filter(({ seg }) => seg && seg.trough1_timestamp && seg.trough2_timestamp);
      if (!guides.length) return;
      const showGuideLabel = options.showLabel !== false;
      const selectedIdx = state.selectedPeakPairIndex.glu_lac;
      guides.forEach(({ item, index, seg }) => {
        const leftMs = parseMs(seg.trough1_timestamp);
        const rightMs = parseMs(seg.trough2_timestamp);
        if (Number.isNaN(leftMs) || Number.isNaN(rightMs)) return;
        const startMs = Math.min(leftMs, rightMs);
        const endMs = Math.max(leftMs, rightMs);
        const left = xForTime(startMs);
        const right = xForTime(endMs);
        const selected = selectedIdx === index;
        ctx.save();
        ctx.fillStyle = selected ? 'rgba(157,78,221,0.24)' : 'rgba(157,78,221,0.11)';
        ctx.fillRect(left, bounds.top, Math.max(1, right - left), bounds.bottom - bounds.top);
        ctx.strokeStyle = selected ? 'rgba(157,78,221,0.75)' : 'rgba(157,78,221,0.35)';
        ctx.lineWidth = selected ? 1.8 : 1;
        ctx.strokeRect(left, bounds.top, Math.max(1, right - left), bounds.bottom - bounds.top);
        if (showGuideLabel) {
          ctx.fillStyle = selected ? '#6f2dbd' : '#8b5bbf';
          ctx.font = '11px Helvetica';
          ctx.fillText(item.name || ('CGM_vs_SEG_' + (index + 1)), left + 4, bounds.top + 14);
          const peakMs = parseMs(seg.peak_timestamp || '');
          if (!Number.isNaN(peakMs)) {
            const peakX = xForTime(peakMs);
            const peakY = bounds.top + 30 + ((index % 3) * 16);
            ctx.fillStyle = selected ? '#5a189a' : '#7b2cbf';
            ctx.beginPath();
            ctx.arc(peakX, peakY, selected ? 4.5 : 3.8, 0, Math.PI * 2);
            ctx.fill();
            ctx.save();
            ctx.translate(peakX + 5, peakY + 3);
            ctx.rotate(-Math.PI / 2);
            ctx.font = selected ? 'bold 11px Helvetica' : '11px Helvetica';
            ctx.fillStyle = selected ? '#5a189a' : '#7b2cbf';
            ctx.fillText('SEG peak', 0, 0);
            ctx.restore();
          }
        }
        ctx.restore();
      });
    }
    function renderList() {
      listEl.innerHTML = '';
      const labelCountForPath = (path) => {
        const saved = state.savedSegments[path] || {};
        return ((saved.labels_glu_cgm || []).length || 0) + ((saved.labels_glu_lac || []).length || 0);
      };
      const visibleItems = state.datasets
        .map((item, index) => ({ item, index, labelCount: labelCountForPath(item.path) }))
        .filter(({ labelCount }) => {
          if (state.fileLabelFilter === 'labeled') return labelCount > 0;
          if (state.fileLabelFilter === 'unlabeled') return labelCount === 0;
          return true;
        });
      const labeledCount = state.datasets.reduce((total, item) => total + (labelCountForPath(item.path) > 0 ? 1 : 0), 0);
      const unlabeledCount = state.datasets.length - labeledCount;
      if (fileFilterSummaryEl) {
        fileFilterSummaryEl.textContent = 'Showing ' + visibleItems.length + ' / ' + state.datasets.length + ' files | labeled ' + labeledCount + ' | unlabeled ' + unlabeledCount;
      }
      if (!visibleItems.length) {
        listEl.innerHTML = '<div class="small">No files match the current filter.</div>';
        return;
      }
      visibleItems.forEach(({ item, index }) => {
        const saved = state.savedSegments[item.path];
        const div = document.createElement('div');
        div.className = 'dataset-item' + (index === state.currentIndex ? ' active' : '');
        div.innerHTML =
          '<button type="button">' + item.path.split('/').slice(-3).join('/') + '</button>' +
          '<div class="small">' + item.path + '</div>' +
          '<div class="small">Duration: ' + item.duration_hms + ' | CGM matches: ' + (item.matched_cgm_points == null ? 'loading later' : item.matched_cgm_points) + '</div>' +
          '<label class="checkline"><input type="checkbox" class="include-toggle-glu-cgm" ' + ((saved?.included_glu_cgm !== false) ? 'checked' : '') + ' /> Include in Glu-CGM cohort</label>' +
          '<label class="checkline"><input type="checkbox" class="include-toggle-glu-lac" ' + ((saved?.included_glu_lac !== false) ? 'checked' : '') + ' /> Include in Glu-Lac cohort</label>' +
          '<div class="small">Region logs: Glu-CGM ' + ((saved?.region_pairs_glu_cgm?.length) || 0) + ' | Glu-Lac ' + ((saved?.region_pairs_glu_lac?.length) || 0) + '</div>' +
          '<div class="small">Labels: Glu-CGM ' + ((saved?.labels_glu_cgm?.length) || 0) + ' | Glu-Lac ' + ((saved?.labels_glu_lac?.length) || 0) + '</div>' +
          '<div class="small">' + (saved && saved.glu_cgm ? ('Glu-CGM: ' + saved.glu_cgm.start_index + ' to ' + saved.glu_cgm.end_index) : 'Glu-CGM: full file') + '</div>' +
          '<div class="small">' + (saved && saved.glu_lac ? ('Glu-Lac: ' + saved.glu_lac.start_index + ' to ' + saved.glu_lac.end_index) : 'Glu-Lac: full file') + '</div>';
        div.querySelector('button').addEventListener('click', async () => {
          state.currentIndex = index;
          await loadCurrentDataset();
        });
        div.querySelector('.include-toggle-glu-cgm').addEventListener('change', async (event) => {
          const checked = !!event.target.checked;
          try {
            const data = await fetchJson('/api/include', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ path: item.path, panel: 'glu_cgm', included: checked })
            });
            state.savedSegments = data.saved_segments || state.savedSegments;
            renderList();
            updateSegmentTable();
            renderRegionLogs();
            renderLabelLogs();
            setStatus((checked ? 'Included ' : 'Excluded ') + item.path.split('/').slice(-3).join('/') + ' in the Glu-CGM cohort.');
          } catch (error) {
            event.target.checked = !checked;
            setStatus('Failed to update include flag (Glu-CGM): ' + error.message);
          }
        });
        div.querySelector('.include-toggle-glu-lac').addEventListener('change', async (event) => {
          const checked = !!event.target.checked;
          try {
            const data = await fetchJson('/api/include', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ path: item.path, panel: 'glu_lac', included: checked })
            });
            state.savedSegments = data.saved_segments || state.savedSegments;
            renderList();
            updateSegmentTable();
            renderRegionLogs();
            renderLabelLogs();
            setStatus((checked ? 'Included ' : 'Excluded ') + item.path.split('/').slice(-3).join('/') + ' in the Glu-Lac cohort.');
          } catch (error) {
            event.target.checked = !checked;
            setStatus('Failed to update include flag (Glu-Lac): ' + error.message);
          }
        });
        listEl.appendChild(div);
      });
    }
    function yRange(seriesList) {
      let min = Infinity;
      let max = -Infinity;
      seriesList.forEach((values) => values.forEach((value) => {
        if (value == null || Number.isNaN(value)) return;
        if (value < min) min = value;
        if (value > max) max = value;
      }));
      if (!isFinite(min) || !isFinite(max)) return { min: -1, max: 1 };
      if (Math.abs(max - min) < 1e-12) { min -= 1; max += 1; }
      const pad = (max - min) * 0.08;
      return { min: min - pad, max: max + pad };
    }
    function parseMaybeNumber(value) {
      if (value == null) return null;
      if (typeof value === 'string') {
        const trimmed = value.trim();
        if (!trimmed || trimmed.toUpperCase() === 'NA') return null;
      }
      const n = Number(value);
      return Number.isNaN(n) ? null : n;
    }
    function visibleRawTimeWindow(timeMs) {
      if (!timeMs.length) return { startMs: 0, endMs: 1 };
      const fullStart = timeMs[0];
      const fullEnd = timeMs[timeMs.length - 1];
      const zStart = Number(state.rawZoom.startMs);
      const zEnd = Number(state.rawZoom.endMs);
      if (Number.isFinite(zStart) && Number.isFinite(zEnd) && zEnd > zStart) {
        return { startMs: Math.max(fullStart, zStart), endMs: Math.min(fullEnd, zEnd) };
      }
      return { startMs: fullStart, endMs: fullEnd };
    }
    function rawPointFromEvent(event) {
      const rect = plotRaw.getBoundingClientRect();
      return {
        x: (event.clientX - rect.left) * (plotRaw.width / rect.width),
        y: (event.clientY - rect.top) * (plotRaw.height / rect.height)
      };
    }
    function rawTimeForX(x) {
      const rows = currentRows();
      if (!rows.length) return null;
      const timeMs = rows.map((row) => parseMs(row.Timestamp));
      const bounds = state.bounds;
      const win = visibleRawTimeWindow(timeMs);
      const frac = clamp((x - bounds.left) / Math.max(1, bounds.right - bounds.left), 0, 1);
      return win.startMs + frac * (win.endMs - win.startMs);
    }
    function nearestRawIndexForX(x) {
      const rows = currentRows();
      if (!rows.length) return null;
      const targetMs = rawTimeForX(x);
      if (targetMs == null) return null;
      const timeMs = rows.map((row) => parseMs(row.Timestamp));
      const win = visibleRawTimeWindow(timeMs);
      let bestIdx = null;
      let bestDist = Infinity;
      for (let i = 0; i < timeMs.length; i += 1) {
        if (timeMs[i] < win.startMs || timeMs[i] > win.endMs) continue;
        const d = Math.abs(timeMs[i] - targetMs);
        if (d < bestDist) {
          bestDist = d;
          bestIdx = i;
        }
      }
      return bestIdx;
    }
    function updateRawHoverDetail(index) {
      const el = document.getElementById('raw-hover-detail');
      if (!el) return;
      const rows = currentRows();
      if (index == null || !rows[index]) {
        el.textContent = 'Drag a box to zoom. Hover to inspect nearest point.';
        return;
      }
      const row = rows[index];
      const parts = [
        'idx ' + index,
        row.Timestamp,
        'Ch1=' + formatMetric(parseMaybeNumber(row.Channel1)),
        'Ch2=' + formatMetric(parseMaybeNumber(row.Channel2)),
        'Ch3=' + formatMetric(parseMaybeNumber(row.Channel3))
      ];
      el.textContent = parts.join(' | ');
    }
    function nearestIndexForCanvasX(canvas, x) {
      const rows = currentRows();
      if (!rows.length) return null;
      const times = rows.map((row) => parseMs(row.Timestamp));
      const startMs = times[0];
      const endMs = times[times.length - 1];
      const frac = clamp((x - state.bounds.left) / Math.max(1, state.bounds.right - state.bounds.left), 0, 1);
      const targetMs = startMs + frac * (endMs - startMs);
      let bestIdx = 0;
      let bestDist = Infinity;
      for (let i = 0; i < times.length; i += 1) {
        const d = Math.abs(times[i] - targetMs);
        if (d < bestDist) {
          bestDist = d;
          bestIdx = i;
        }
      }
      return bestIdx;
    }
    function canvasPoint(canvas, event) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: (event.clientX - rect.left) * (canvas.width / rect.width),
        y: (event.clientY - rect.top) * (canvas.height / rect.height)
      };
    }
    function updateConvertedHoverDetail(index) {
      const el = document.getElementById('converted-hover-detail');
      if (!el) return;
      const rows = currentRows();
      if (index == null || !rows[index]) {
        el.textContent = 'Hover over the plot to inspect converted concentration and CGM values.';
        return;
      }
      const series = getRenderSeries(rows);
      const concentration = series?.convertedConc?.[index];
      const rowMs = parseMs(rows[index].Timestamp);
      let cgm = series?.cgmRaw?.[index];
      if (series?.cgmPoints?.length) {
        let best = null;
        let bestDist = Infinity;
        series.cgmPoints.forEach((point) => {
          const dist = Math.abs(point.ms - rowMs);
          if (dist < bestDist) {
            bestDist = dist;
            best = point;
          }
        });
        if (best) cgm = best.value;
      }
      el.textContent = [
        'idx ' + index,
        rows[index].Timestamp,
        'Concentration=' + formatMetric(concentration),
        'CGM=' + formatMetric(cgm)
      ].join(' | ');
    }
    function smoothDense(values, ratio = 0.08) {
      if (ratio <= 0) return values.slice();
      if (!values.length) return [];
      const window = Math.max(5, Math.ceil(values.length * Math.max(0, ratio)));
      const oddWindow = window % 2 === 0 ? window + 1 : window;
      const half = Math.floor(oddWindow / 2);
      const out = [];
      for (let idx = 0; idx < values.length; idx += 1) {
        let total = 0;
        let count = 0;
        const left = Math.max(0, idx - half);
        const right = Math.min(values.length, idx + half + 1);
        for (let j = left; j < right; j += 1) {
          const value = values[j];
          if (value == null || Number.isNaN(value)) continue;
          total += value;
          count += 1;
        }
        out.push(count ? total / count : values[idx]);
      }
      return out;
    }
    function conversionSourceColumn() {
      return state.conversionSource === 'sel' ? 'Delta_Channel2_minus_Channel3' : 'Delta_Channel2_minus_Channel1';
    }
    function conversionSourceLabel() {
      return state.conversionSource === 'sel' ? 'SEL lactate converted concentration' : 'SEG glucose converted concentration';
    }
    function concentrationFromDelta(deltaValue) {
      if (deltaValue == null || Number.isNaN(deltaValue)) return null;
      return (Number(deltaValue) - 0.0188) / 0.00185;
    }
    function segConvertedConcentrationUm(deltaValue) {
      if (deltaValue == null || Number.isNaN(deltaValue)) return null;
      return (Number(deltaValue) - 0.00453) / 0.00128;
    }
    function selConvertedConcentrationUm(deltaValue) {
      if (deltaValue == null || Number.isNaN(deltaValue)) return null;
      return (Number(deltaValue) - 0.00638) / 2.9145e-4;
    }
    function convertedConcentrationSeries(rows) {
      const column = conversionSourceColumn();
      const delta = rows.map((row) => parseMaybeNumber(row[column]));
      const smoothedDelta = smoothDense(delta, state.conversionSmoothing);
      return smoothedDelta.map((value) => concentrationFromDelta(value));
    }
    function rangeLabelSeries(rows) {
      const cgmRaw = rows.map((row) => parseMaybeNumber(row.CGM));
      const segDelta = rows.map((row) => parseMaybeNumber(row.Delta_Channel2_minus_Channel1));
      const selDelta = rows.map((row) => parseMaybeNumber(row.Delta_Channel2_minus_Channel3));
      const cgmSmoothed = smoothDense(cgmRaw, state.smoothRatio);
      const segDeltaSmoothed = smoothDense(segDelta, state.smoothRatio);
      const selDeltaSmoothed = smoothDense(selDelta, state.smoothRatio);
      return {
        cgm: cgmSmoothed,
        seg: segDeltaSmoothed.map((value) => segConvertedConcentrationUm(value)),
        sel: selDeltaSmoothed.map((value) => selConvertedConcentrationUm(value))
      };
    }
    function normalizedSeries(rows) {
      const glucoseRaw = rows.map((row) => parseMaybeNumber(row.Delta_Channel2_minus_Channel1));
      const lactateRaw = rows.map((row) => parseMaybeNumber(row.Delta_Channel2_minus_Channel3));
      const cgmRaw = rows.map((row) => parseMaybeNumber(row.CGM));
      const normalizeSparse = (rawValues) => {
        const known = rawValues.filter((value) => value != null);
        const knownNorm = minmax(known);
        const out = [];
        let ptr = 0;
        rawValues.forEach((value) => {
          if (value == null) out.push(null);
          else {
            out.push(knownNorm[ptr]);
            ptr += 1;
          }
        });
        return out;
      };
      return {
        glucose: normalizeSparse(smoothDense(glucoseRaw, state.smoothRatio)),
        lactate: normalizeSparse(smoothDense(lactateRaw, state.smoothRatio)),
        cgm: normalizeSparse(cgmRaw)
      };
    }
    function drawSplitGluLacPlot(canvas, ctx, title, glucoseLine, lactateLine) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const rows = currentRows();
      if (!rows.length) return;
      const base = state.bounds;
      const splitBounds = gluLacSplitBounds();
      const topBounds = splitBounds.top;
      const bottomBounds = splitBounds.bottom;
      const timeMs = rows.map((row) => parseMs(row.Timestamp));
      const startMs = timeMs[0];
      const endMs = timeMs[timeMs.length - 1];
      const xForTime = (ms) => base.left + ((ms - startMs) / Math.max(1, endMs - startMs)) * (base.right - base.left);
      const dietMarkers = currentDietMarkers()
        .map((marker) => ({ ...marker, ms: parseMs(marker.timestamp) }))
        .filter((marker) => !Number.isNaN(marker.ms) && marker.ms >= startMs && marker.ms <= endMs);
      const selected = selectedPeakAggBySignal('glu_lac');
      const glucoseRange = yRange([glucoseLine.values]);
      const lactateRange = yRange([lactateLine.values]);
      const yForIn = (bounds, range, value) => bounds.bottom - ((value - range.min) / (range.max - range.min)) * (bounds.bottom - bounds.top);
      function drawAxis(bounds, line, range, axisLabel) {
        ctx.fillStyle = '#fff';
        ctx.fillRect(bounds.left, bounds.top, bounds.right - bounds.left, bounds.bottom - bounds.top);
        ctx.strokeStyle = '#d7d1c4';
        ctx.strokeRect(bounds.left, bounds.top, bounds.right - bounds.left, bounds.bottom - bounds.top);
        for (let i = 0; i <= 4; i += 1) {
          const y = bounds.top + (i / 4) * (bounds.bottom - bounds.top);
          ctx.strokeStyle = '#f0ece2';
          ctx.beginPath();
          ctx.moveTo(bounds.left, y);
          ctx.lineTo(bounds.right, y);
          ctx.stroke();
          const value = range.max - (i / 4) * (range.max - range.min);
          ctx.fillStyle = '#777';
          ctx.font = '10px Helvetica';
          const text = value.toFixed(2);
          ctx.fillText(text, Math.max(4, bounds.left - ctx.measureText(text).width - 8), y + 3);
        }
        ctx.fillStyle = line.color;
        ctx.font = '12px Helvetica';
        ctx.fillText(axisLabel, bounds.left + 8, bounds.top + 18);
      }
      function drawTimeSpans(bounds, includeGuideLabels) {
        dietMarkers.forEach((marker, markerIdx) => {
          const x = xForTime(marker.ms);
          ctx.strokeStyle = 'rgba(196, 73, 83, 0.8)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x, bounds.top);
          ctx.lineTo(x, bounds.bottom);
          ctx.stroke();
          if (includeGuideLabels) {
            ctx.save();
            ctx.translate(x + 4, bounds.top + 16 + ((markerIdx % 3) * 12));
            ctx.rotate(-Math.PI / 2);
            ctx.fillStyle = '#7a2831';
            ctx.font = '10px Helvetica';
            ctx.fillText(marker.label, 0, 0);
            ctx.restore();
          }
        });
        drawFixedSegTripletGuides(ctx, bounds, timeMs, xForTime, { showLabel: includeGuideLabels });
        (state.regionPairs.glu_lac || []).forEach((pair) => {
          const left = xForTime(timeMs[pair.start_index]);
          const right = xForTime(timeMs[pair.end_index]);
          ctx.fillStyle = 'rgba(230,166,58,0.14)';
          ctx.fillRect(left, bounds.top, right - left, bounds.bottom - bounds.top);
        });
        const explicitCursorMs = Number(state.cursorTimeMs?.glu_lac);
        const cursorIdx = clamp(state.cursorIndex.glu_lac ?? 0, 0, timeMs.length - 1);
        const cursorX = xForTime(Number.isFinite(explicitCursorMs) ? explicitCursorMs : timeMs[cursorIdx]);
        ctx.strokeStyle = 'rgba(23,63,95,0.85)';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(cursorX, bounds.top);
        ctx.lineTo(cursorX, bounds.bottom);
        ctx.stroke();
      }
      function drawLineIn(bounds, line, range) {
        ctx.strokeStyle = line.color;
        ctx.lineWidth = line.options?.lineWidth || 1.8;
        let drawing = false;
        const sampled = line.values.some((value) => value == null || Number.isNaN(value)) ? downsampleSparse(line.values) : downsample(line.values);
        sampled.forEach(([idx, value], sampleIdx) => {
          if (value == null || Number.isNaN(value)) {
            if (drawing) {
              ctx.stroke();
              drawing = false;
            }
            return;
          }
          const x = xForTime(timeMs[idx]);
          const y = yForIn(bounds, range, value);
          if (!drawing) {
            ctx.beginPath();
            ctx.moveTo(x, y);
            drawing = true;
          } else {
            ctx.lineTo(x, y);
          }
          const nextPair = sampled[sampleIdx + 1];
          if (!nextPair || nextPair[1] == null || Number.isNaN(nextPair[1])) {
            ctx.stroke();
            drawing = false;
          }
        });
      }
      drawAxis(topBounds, glucoseLine, glucoseRange, 'SEB glucose proxy (normalized)');
      drawAxis(bottomBounds, lactateLine, lactateRange, 'SEB lactate proxy (normalized)');
      drawTimeSpans(topBounds, true);
      drawTimeSpans(bottomBounds, false);
      drawPeakAggHighlight(ctx, bottomBounds, timeMs, xForTime, selected.sel, 'rgba(58,134,255,0.10)', 'rgba(58,134,255,0.95)');
      const markerStart = state.markerIndex.glu_lac.start;
      const markerEnd = state.markerIndex.glu_lac.end;
      const drawMarker = (idx, color, label, bounds) => {
        if (idx == null) return;
        const cIdx = clamp(idx, 0, timeMs.length - 1);
        const x = xForTime(timeMs[cIdx]);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(x, bounds.top);
        ctx.lineTo(x, bounds.bottom);
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.font = '10px Helvetica';
        ctx.fillText(label, x + 4, bounds.bottom + 12);
      };
      [topBounds, bottomBounds].forEach((bounds) => {
        drawMarker(markerStart, 'rgba(255,140,0,0.95)', 'S', bounds);
        drawMarker(markerEnd, 'rgba(0,128,0,0.95)', 'E', bounds);
      });
      drawLineIn(topBounds, glucoseLine, glucoseRange);
      drawLineIn(bottomBounds, lactateLine, lactateRange);
      drawSavedLabels(ctx, bottomBounds, timeMs, xForTime, state.labelPoints.glu_lac || []);
      ctx.fillStyle = '#333';
      ctx.font = '13px Helvetica';
      ctx.fillText(title, base.left, base.top - 20);
      const legend = [[glucoseLine.label, glucoseLine.color], [lactateLine.label, lactateLine.color], ['Diet instance', '#c14953']];
      let legendX = base.left;
      legend.forEach(([label, color]) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(legendX, 44);
        ctx.lineTo(legendX + 18, 44);
        ctx.stroke();
        ctx.fillStyle = '#444';
        ctx.font = '12px Helvetica';
        ctx.fillText(label, legendX + 24, 48);
        legendX += 220;
      });
      ctx.fillStyle = '#555';
      ctx.font = '12px Helvetica';
      ctx.fillText(rows[0].Timestamp, base.left, canvas.height - 18);
      const endLabel = rows[rows.length - 1].Timestamp;
      ctx.fillText(endLabel, base.right - ctx.measureText(endLabel).width, canvas.height - 18);
    }
    function drawPlot(canvas, ctx, title, lines, panel) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const rows = currentRows();
      if (!rows.length) return;
      const isRawPanel = canvas === plotRaw;
      const bounds = state.bounds;
      ctx.fillStyle = '#fff';
      ctx.fillRect(bounds.left, bounds.top, bounds.right - bounds.left, bounds.bottom - bounds.top);
      ctx.strokeStyle = '#d7d1c4';
      ctx.strokeRect(bounds.left, bounds.top, bounds.right - bounds.left, bounds.bottom - bounds.top);
      const timeMs = rows.map((row) => parseMs(row.Timestamp));
      const visibleWindow = isRawPanel ? visibleRawTimeWindow(timeMs) : { startMs: timeMs[0], endMs: timeMs[timeMs.length - 1] };
      const startMs = visibleWindow.startMs;
      const endMs = visibleWindow.endMs;
      const dietMarkers = currentDietMarkers()
        .map((marker) => ({ ...marker, ms: parseMs(marker.timestamp) }))
        .filter((marker) => !Number.isNaN(marker.ms) && marker.ms >= startMs && marker.ms <= endMs);
      const visibleLines = lines.map((line) => {
        if (!isRawPanel) return line.values;
        return line.values.map((value, idx) => (timeMs[idx] >= startMs && timeMs[idx] <= endMs ? value : null));
      });
      const range = yRange(visibleLines);
      const xForTime = (ms) => bounds.left + ((ms - startMs) / Math.max(1, endMs - startMs)) * (bounds.right - bounds.left);
      const yFor = (value) => bounds.bottom - ((value - range.min) / (range.max - range.min)) * (bounds.bottom - bounds.top);
      for (let i = 0; i <= 5; i += 1) {
        const y = bounds.top + (i / 5) * (bounds.bottom - bounds.top);
        ctx.strokeStyle = '#f0ece2';
        ctx.beginPath();
        ctx.moveTo(bounds.left, y);
        ctx.lineTo(bounds.right, y);
        ctx.stroke();
      }
      dietMarkers.forEach((marker, markerIdx) => {
        const x = xForTime(marker.ms);
        ctx.strokeStyle = 'rgba(196, 73, 83, 0.9)';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(x, bounds.top);
        ctx.lineTo(x, bounds.bottom);
        ctx.stroke();
        ctx.save();
        ctx.translate(x + 4, bounds.top + 16 + ((markerIdx % 4) * 12));
        ctx.rotate(-Math.PI / 2);
        ctx.fillStyle = '#7a2831';
        ctx.font = '11px Helvetica';
        ctx.fillText(marker.label, 0, 0);
        ctx.restore();
      });
      if (panel === 'glu_lac') {
        const selected = selectedPeakAggBySignal(panel);
        drawFixedSegTripletGuides(ctx, bounds, timeMs, xForTime);
        drawPeakAggHighlight(ctx, bounds, timeMs, xForTime, selected.sel, 'rgba(58,134,255,0.10)', 'rgba(58,134,255,0.95)');
      }
      if (panel) {
        const pairs = state.regionPairs[panel] || [];
        pairs.forEach((pair) => {
          const left = xForTime(timeMs[pair.start_index]);
          const right = xForTime(timeMs[pair.end_index]);
          ctx.fillStyle = 'rgba(230,166,58,0.18)';
          ctx.fillRect(left, bounds.top, right - left, bounds.bottom - bounds.top);
        });
        const cursorIdx = clamp(state.cursorIndex[panel] ?? 0, 0, timeMs.length - 1);
        const explicitCursorMs = Number(state.cursorTimeMs?.[panel]);
        const cursorX = xForTime(Number.isFinite(explicitCursorMs) ? explicitCursorMs : timeMs[cursorIdx]);
        ctx.strokeStyle = 'rgba(23,63,95,0.85)';
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(cursorX, bounds.top);
        ctx.lineTo(cursorX, bounds.bottom);
        ctx.stroke();
        const markerStart = state.markerIndex[panel].start;
        const markerEnd = state.markerIndex[panel].end;
        const drawMarker = (idx, color, label) => {
          if (idx == null) return;
          const cIdx = clamp(idx, 0, timeMs.length - 1);
          const x = xForTime(timeMs[cIdx]);
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.moveTo(x, bounds.top);
          ctx.lineTo(x, bounds.bottom);
          ctx.stroke();
          ctx.fillStyle = color;
          ctx.font = '11px Helvetica';
          ctx.fillText(label, x + 4, bounds.bottom + 14);
          ctx.beginPath();
          ctx.arc(x, bounds.bottom + 24, 5, 0, Math.PI * 2);
          ctx.fill();
        };
        drawMarker(markerStart, 'rgba(255,140,0,0.95)', 'S');
        drawMarker(markerEnd, 'rgba(0,128,0,0.95)', 'E');
        const labels = state.labelPoints[panel] || [];
        drawSavedLabels(ctx, bounds, timeMs, xForTime, labels);
      }
      function drawLine(values, color, options = {}) {
        ctx.strokeStyle = color;
        ctx.lineWidth = options.lineWidth || 1.8;
        let drawing = false;
        const hasMissing = values.some((value) => value == null || Number.isNaN(value));
        const sampled = hasMissing ? downsampleSparse(values) : downsample(values);
        sampled.forEach(([idx, value], sampleIdx) => {
          if (value == null || Number.isNaN(value)) {
            if (drawing) {
              ctx.stroke();
              drawing = false;
            }
            return;
          }
          if (timeMs[idx] < startMs || timeMs[idx] > endMs) {
            if (drawing) {
              ctx.stroke();
              drawing = false;
            }
            return;
          }
          const x = xForTime(timeMs[idx]);
          const y = yFor(value);
          if (!drawing) {
            ctx.beginPath();
            ctx.moveTo(x, y);
            drawing = true;
          } else {
            ctx.lineTo(x, y);
          }
          const nextPair = sampled[sampleIdx + 1];
          const shouldBreak =
            !nextPair ||
            nextPair[1] == null ||
            Number.isNaN(nextPair[1]);
          if (shouldBreak) {
            ctx.stroke();
            drawing = false;
          }
        });
        if (options.showPoints) {
          ctx.fillStyle = color;
          sampled.forEach(([idx, value]) => {
            if (value == null || Number.isNaN(value)) return;
            if (timeMs[idx] < startMs || timeMs[idx] > endMs) return;
            const x = xForTime(timeMs[idx]);
            const y = yFor(value);
            ctx.beginPath();
            ctx.arc(x, y, options.pointRadius || 2.4, 0, Math.PI * 2);
            ctx.fill();
          });
        }
      }
      lines.forEach((line) => drawLine(line.values, line.color, line.options || {}));
      if (isRawPanel && state.rawHoverIndex != null && rows[state.rawHoverIndex]) {
        const idx = clamp(state.rawHoverIndex, 0, timeMs.length - 1);
        if (timeMs[idx] >= startMs && timeMs[idx] <= endMs) {
          const x = xForTime(timeMs[idx]);
          ctx.save();
          ctx.strokeStyle = 'rgba(23,63,95,0.75)';
          ctx.lineWidth = 1.2;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(x, bounds.top);
          ctx.lineTo(x, bounds.bottom);
          ctx.stroke();
          ctx.setLineDash([]);
          lines.forEach((line) => {
            const value = line.values[idx];
            if (value == null || Number.isNaN(value)) return;
            const y = yFor(value);
            ctx.fillStyle = line.color;
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fill();
          });
          ctx.restore();
        }
      }
      if (isRawPanel && state.rawDrag.active && state.rawDrag.startX != null && state.rawDrag.currentX != null) {
        const left = clamp(Math.min(state.rawDrag.startX, state.rawDrag.currentX), bounds.left, bounds.right);
        const right = clamp(Math.max(state.rawDrag.startX, state.rawDrag.currentX), bounds.left, bounds.right);
        if (right - left > 2) {
          ctx.save();
          ctx.fillStyle = 'rgba(23,63,95,0.12)';
          ctx.strokeStyle = 'rgba(23,63,95,0.85)';
          ctx.lineWidth = 1.2;
          ctx.setLineDash([5, 3]);
          ctx.fillRect(left, bounds.top, right - left, bounds.bottom - bounds.top);
          ctx.strokeRect(left, bounds.top, right - left, bounds.bottom - bounds.top);
          ctx.restore();
        }
      }
      const legend = lines.map((line) => [line.label, line.color]);
      legend.push(['Diet instance', '#c14953']);
      let legendX = bounds.left;
      legend.forEach(([label, color]) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(legendX, 44);
        ctx.lineTo(legendX + 18, 44);
        ctx.stroke();
        ctx.fillStyle = '#444';
        ctx.font = '12px Helvetica';
        ctx.fillText(label, legendX + 24, 48);
        legendX += 220;
      });
      ctx.fillStyle = '#333';
      ctx.font = '13px Helvetica';
      ctx.fillText(title, bounds.left, bounds.top - 20);
      ctx.fillStyle = '#555';
      ctx.font = '12px Helvetica';
      const startIdx = isRawPanel ? nearestRawIndexForX(bounds.left) : 0;
      const endIdx = isRawPanel ? nearestRawIndexForX(bounds.right) : rows.length - 1;
      ctx.fillText(rows[startIdx]?.Timestamp || rows[0].Timestamp, bounds.left, canvas.height - 18);
      const endLabel = rows[endIdx]?.Timestamp || rows[rows.length - 1].Timestamp;
      const width = ctx.measureText(endLabel).width;
      ctx.fillText(endLabel, bounds.right - width, canvas.height - 18);
    }
    function drawDualAxisPlot(canvas, ctx, title, leftLine, rightLine, panel, leftAxisLabel, rightAxisLabel) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const rows = currentRows();
      if (!rows.length) return;
      const bounds = state.bounds;
      ctx.fillStyle = '#fff';
      ctx.fillRect(bounds.left, bounds.top, bounds.right - bounds.left, bounds.bottom - bounds.top);
      ctx.strokeStyle = '#d7d1c4';
      ctx.strokeRect(bounds.left, bounds.top, bounds.right - bounds.left, bounds.bottom - bounds.top);
      const timeMs = rows.map((row) => parseMs(row.Timestamp));
      const startMs = timeMs[0];
      const endMs = timeMs[timeMs.length - 1];
      const dietMarkers = currentDietMarkers()
        .map((marker) => ({ ...marker, ms: parseMs(marker.timestamp) }))
        .filter((marker) => !Number.isNaN(marker.ms) && marker.ms >= startMs && marker.ms <= endMs);
      const leftRange = yRange([leftLine.values, (leftLine.points || []).map((point) => point.value)]);
      const rightRange = yRange([rightLine.values, (rightLine.points || []).map((point) => point.value)]);
      const xForTime = (ms) => bounds.left + ((ms - startMs) / Math.max(1, endMs - startMs)) * (bounds.right - bounds.left);
      const yForLeft = (value) => bounds.bottom - ((value - leftRange.min) / (leftRange.max - leftRange.min)) * (bounds.bottom - bounds.top);
      const yForRight = (value) => bounds.bottom - ((value - rightRange.min) / (rightRange.max - rightRange.min)) * (bounds.bottom - bounds.top);
      for (let i = 0; i <= 5; i += 1) {
        const y = bounds.top + (i / 5) * (bounds.bottom - bounds.top);
        ctx.strokeStyle = '#f0ece2';
        ctx.beginPath();
        ctx.moveTo(bounds.left, y);
        ctx.lineTo(bounds.right, y);
        ctx.stroke();
        const leftValue = leftRange.max - (i / 5) * (leftRange.max - leftRange.min);
        const rightValue = rightRange.max - (i / 5) * (rightRange.max - rightRange.min);
        ctx.font = '11px Helvetica';
        ctx.fillStyle = leftLine.color;
        const leftText = leftValue.toFixed(3);
        ctx.fillText(leftText, Math.max(4, bounds.left - ctx.measureText(leftText).width - 8), y + 4);
        ctx.fillStyle = rightLine.color;
        ctx.fillText(rightValue.toFixed(3), bounds.right + 8, y + 4);
        ctx.strokeStyle = '#bfb8aa';
        ctx.beginPath();
        ctx.moveTo(bounds.left - 4, y);
        ctx.lineTo(bounds.left, y);
        ctx.moveTo(bounds.right, y);
        ctx.lineTo(bounds.right + 4, y);
        ctx.stroke();
      }
      function drawNegativeRegions(values) {
        if (!state.showNegativeRegions || canvas !== plotConvertedCgm || !values || !values.length) return;
        let activeStart = null;
        const midpoint = (a, b) => a + ((b - a) / 2);
        for (let i = 0; i < values.length; i += 1) {
          const value = values[i];
          const isNegative = value != null && !Number.isNaN(value) && value < 0;
          if (isNegative && activeStart == null) {
            activeStart = i > 0 ? midpoint(timeMs[i - 1], timeMs[i]) : timeMs[i];
          }
          if ((!isNegative || i === values.length - 1) && activeStart != null) {
            const endMsRegion = isNegative && i === values.length - 1 ? timeMs[i] : (i > 0 ? midpoint(timeMs[i - 1], timeMs[i]) : timeMs[i]);
            const left = clamp(xForTime(activeStart), bounds.left, bounds.right);
            const right = clamp(xForTime(endMsRegion), bounds.left, bounds.right);
            if (right > left) {
              ctx.fillStyle = 'rgba(220, 38, 38, 0.16)';
              ctx.fillRect(left, bounds.top, right - left, bounds.bottom - bounds.top);
              ctx.strokeStyle = 'rgba(185, 28, 28, 0.45)';
              ctx.lineWidth = 1;
              ctx.setLineDash([4, 3]);
              ctx.strokeRect(left, bounds.top, right - left, bounds.bottom - bounds.top);
              ctx.setLineDash([]);
            }
            activeStart = null;
          }
        }
      }
      drawNegativeRegions(leftLine.values);
      dietMarkers.forEach((marker, markerIdx) => {
        const x = xForTime(marker.ms);
        ctx.strokeStyle = 'rgba(196, 73, 83, 0.9)';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(x, bounds.top);
        ctx.lineTo(x, bounds.bottom);
        ctx.stroke();
        ctx.save();
        ctx.translate(x + 4, bounds.top + 16 + ((markerIdx % 4) * 12));
        ctx.rotate(-Math.PI / 2);
        ctx.fillStyle = '#7a2831';
        ctx.font = '11px Helvetica';
        ctx.fillText(marker.label, 0, 0);
        ctx.restore();
      });
      if (panel === 'glu_cgm') {
        const selected = selectedPeakAggBySignal(panel);
        drawPeakAggHighlight(ctx, bounds, timeMs, xForTime, selected.seg, 'rgba(157,78,221,0.12)', 'rgba(157,78,221,0.95)');
        drawPeakAggHighlight(ctx, bounds, timeMs, xForTime, selected.cgm, 'rgba(108,117,125,0.12)', 'rgba(108,117,125,0.95)');
      }
      if (panel) {
        const pairs = state.regionPairs[panel] || [];
        pairs.forEach((pair) => {
          const left = xForTime(timeMs[pair.start_index]);
          const right = xForTime(timeMs[pair.end_index]);
          ctx.fillStyle = 'rgba(230,166,58,0.18)';
          ctx.fillRect(left, bounds.top, right - left, bounds.bottom - bounds.top);
        });
        const cursorIdx = clamp(state.cursorIndex[panel] ?? 0, 0, timeMs.length - 1);
        const explicitCursorMs = Number(state.cursorTimeMs?.[panel]);
        const cursorX = xForTime(Number.isFinite(explicitCursorMs) ? explicitCursorMs : timeMs[cursorIdx]);
        ctx.strokeStyle = 'rgba(23,63,95,0.85)';
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(cursorX, bounds.top);
        ctx.lineTo(cursorX, bounds.bottom);
        ctx.stroke();
        const markerStart = state.markerIndex[panel].start;
        const markerEnd = state.markerIndex[panel].end;
        const drawMarker = (idx, color, label) => {
          if (idx == null) return;
          const cIdx = clamp(idx, 0, timeMs.length - 1);
          const x = xForTime(timeMs[cIdx]);
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.moveTo(x, bounds.top);
          ctx.lineTo(x, bounds.bottom);
          ctx.stroke();
          ctx.fillStyle = color;
          ctx.font = '11px Helvetica';
          ctx.fillText(label, x + 4, bounds.bottom + 14);
          ctx.beginPath();
          ctx.arc(x, bounds.bottom + 24, 5, 0, Math.PI * 2);
          ctx.fill();
        };
        drawMarker(markerStart, 'rgba(255,140,0,0.95)', 'S');
        drawMarker(markerEnd, 'rgba(0,128,0,0.95)', 'E');
        const labels = state.labelPoints[panel] || [];
        drawSavedLabels(ctx, bounds, timeMs, xForTime, labels);
      }
      function drawLine(values, color, yFor, options = {}) {
        const hasMissing = values.some((value) => value == null || Number.isNaN(value));
        const sampled = hasMissing ? downsampleSparse(values) : downsample(values);
        if (!options.pointsOnly) {
          ctx.strokeStyle = color;
          ctx.lineWidth = options.lineWidth || 1.8;
          let drawing = false;
          sampled.forEach(([idx, value], sampleIdx) => {
            if (value == null || Number.isNaN(value)) {
              if (drawing) {
                ctx.stroke();
                drawing = false;
              }
              return;
            }
            const x = xForTime(timeMs[idx]);
            const y = yFor(value);
            if (!drawing) {
              ctx.beginPath();
              ctx.moveTo(x, y);
              drawing = true;
            } else {
              ctx.lineTo(x, y);
            }
            const nextPair = sampled[sampleIdx + 1];
            const shouldBreak = !nextPair || nextPair[1] == null || Number.isNaN(nextPair[1]);
            if (shouldBreak) {
              ctx.stroke();
              drawing = false;
            }
          });
        }
        if (options.showPoints) {
          ctx.fillStyle = color;
          sampled.forEach(([idx, value]) => {
            if (value == null || Number.isNaN(value)) return;
            const x = xForTime(timeMs[idx]);
            const y = yFor(value);
            ctx.beginPath();
            ctx.arc(x, y, options.pointRadius || 2.4, 0, Math.PI * 2);
            ctx.fill();
          });
        }
      }
      function drawTimestampedLine(points, color, yFor, options = {}) {
        const visiblePoints = (points || [])
          .filter((point) => point && point.value != null && !Number.isNaN(point.value) && point.ms >= startMs && point.ms <= endMs)
          .sort((a, b) => a.ms - b.ms);
        if (!visiblePoints.length) return;
        if (!options.pointsOnly) {
          ctx.strokeStyle = color;
          ctx.lineWidth = options.lineWidth || 1.8;
          ctx.beginPath();
          visiblePoints.forEach((point, idx) => {
            const x = xForTime(point.ms);
            const y = yFor(point.value);
            if (idx === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          });
          ctx.stroke();
        }
        if (options.showPoints !== false) {
          ctx.fillStyle = color;
          visiblePoints.forEach((point) => {
            const x = xForTime(point.ms);
            const y = yFor(point.value);
            ctx.beginPath();
            ctx.arc(x, y, options.pointRadius || 2.4, 0, Math.PI * 2);
            ctx.fill();
          });
        }
      }
      if (leftLine.points) drawTimestampedLine(leftLine.points, leftLine.color, yForLeft, leftLine.options || {});
      else drawLine(leftLine.values, leftLine.color, yForLeft, leftLine.options || {});
      if (rightLine.points) drawTimestampedLine(rightLine.points, rightLine.color, yForRight, rightLine.options || {});
      else drawLine(rightLine.values, rightLine.color, yForRight, rightLine.options || {});
      if (canvas === plotConvertedCgm && state.convertedHoverIndex != null && rows[state.convertedHoverIndex]) {
        const idx = clamp(state.convertedHoverIndex, 0, timeMs.length - 1);
        const x = xForTime(timeMs[idx]);
        ctx.save();
        ctx.strokeStyle = 'rgba(23,63,95,0.75)';
        ctx.lineWidth = 1.2;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(x, bounds.top);
        ctx.lineTo(x, bounds.bottom);
        ctx.stroke();
        ctx.setLineDash([]);
        const leftVal = leftLine.values[idx];
        let rightVal = rightLine.values[idx];
        let rightMs = timeMs[idx];
        if (rightLine.points && rightLine.points.length) {
          let nearest = null;
          let nearestDist = Infinity;
          rightLine.points.forEach((point) => {
            const dist = Math.abs(point.ms - timeMs[idx]);
            if (dist < nearestDist) {
              nearestDist = dist;
              nearest = point;
            }
          });
          if (nearest) {
            rightVal = nearest.value;
            rightMs = nearest.ms;
          }
        }
        if (leftVal != null && !Number.isNaN(leftVal)) {
          ctx.fillStyle = leftLine.color;
          ctx.beginPath();
          ctx.arc(x, yForLeft(leftVal), 4, 0, Math.PI * 2);
          ctx.fill();
        }
        if (rightVal != null && !Number.isNaN(rightVal)) {
          ctx.fillStyle = rightLine.color;
          ctx.beginPath();
          ctx.arc(xForTime(rightMs), yForRight(rightVal), 4, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      }
      const legend = [[leftLine.label, leftLine.color], [rightLine.label, rightLine.color], ['Diet instance', '#c14953']];
      let legendX = bounds.left;
      legend.forEach(([label, color]) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(legendX, 44);
        ctx.lineTo(legendX + 18, 44);
        ctx.stroke();
        ctx.fillStyle = '#444';
        ctx.font = '12px Helvetica';
        ctx.fillText(label, legendX + 24, 48);
        legendX += 220;
      });
      ctx.fillStyle = '#333';
      ctx.font = '13px Helvetica';
      ctx.fillText(title, bounds.left, bounds.top - 20);
      ctx.fillStyle = leftLine.color;
      ctx.font = '11px Helvetica';
      ctx.fillText(leftAxisLabel + ': ' + leftRange.min.toFixed(3) + ' to ' + leftRange.max.toFixed(3), bounds.left, bounds.bottom + 20);
      const rightLabel = rightAxisLabel + ': ' + rightRange.min.toFixed(3) + ' to ' + rightRange.max.toFixed(3);
      ctx.fillStyle = rightLine.color;
      const rightW = ctx.measureText(rightLabel).width;
      ctx.fillText(rightLabel, bounds.right - rightW, bounds.bottom + 20);
      if (panel === 'glu_cgm') {
        const dayRanges = currentOriginalCgmDayRanges();
        ctx.fillStyle = '#495057';
        ctx.font = '11px Helvetica';
        dayRanges.forEach((item, i) => {
          const txt = item.label + ' CGM: ' + item.min.toFixed(3) + ' to ' + item.max.toFixed(3);
          const w = ctx.measureText(txt).width;
          ctx.fillText(txt, bounds.right - w, bounds.top - 22 + (i * 13));
        });
      }
      ctx.fillStyle = '#555';
      ctx.font = '12px Helvetica';
      ctx.fillText(rows[0].Timestamp, bounds.left, canvas.height - 18);
      const endLabel = rows[rows.length - 1].Timestamp;
      const width = ctx.measureText(endLabel).width;
      ctx.fillText(endLabel, bounds.right - width, canvas.height - 18);
    }
    function renderPanelSafely(name, fn) {
      try {
        fn();
      } catch (error) {
        console.error('Failed to render ' + name, error);
        setStatus('Failed to render ' + name + ': ' + error.message);
      }
    }
    function rangeSelectionBounds() {
      const s = state.rangeLabelSelection.start;
      const e = state.rangeLabelSelection.end;
      if (s == null || e == null) return null;
      return { start: Math.min(s, e), end: Math.max(s, e) };
    }
    function drawRangeLabelPlot(canvas, ctx, key, title, values, color) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const rows = currentRows();
      const bounds = { left: 70, right: canvas.width - 50, top: 45, bottom: canvas.height - 52 };
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = '#e0ddd4';
      ctx.strokeRect(bounds.left, bounds.top, bounds.right - bounds.left, bounds.bottom - bounds.top);
      ctx.fillStyle = '#173f5f';
      ctx.font = '14px Helvetica';
      ctx.fillText(title, bounds.left, 24);
      if (!rows.length) return;
      const timeMs = rows.map((row) => parseMs(row.Timestamp));
      const startMs = timeMs[0];
      const endMs = timeMs[timeMs.length - 1] || (startMs + 1);
      const xForIndex = (idx) => {
        const frac = (timeMs[idx] - startMs) / Math.max(1, endMs - startMs);
        return bounds.left + frac * (bounds.right - bounds.left);
      };
      const yr = yRange([values]);
      const yForValue = (value) => {
        const frac = (value - yr.min) / Math.max(1e-12, yr.max - yr.min);
        return bounds.bottom - frac * (bounds.bottom - bounds.top);
      };
      const selection = rangeSelectionBounds();
      if (selection) {
        const x1 = xForIndex(selection.start);
        const x2 = xForIndex(selection.end);
        ctx.fillStyle = 'rgba(42, 157, 143, 0.16)';
        ctx.fillRect(Math.min(x1, x2), bounds.top, Math.abs(x2 - x1), bounds.bottom - bounds.top);
        ctx.strokeStyle = 'rgba(42, 157, 143, 0.9)';
        ctx.lineWidth = 1.5;
        [x1, x2].forEach((x) => {
          ctx.beginPath();
          ctx.moveTo(x, bounds.top);
          ctx.lineTo(x, bounds.bottom);
          ctx.stroke();
        });
      }
      if (state.rangeLabelDrag.active && state.rangeLabelDrag.source === key) {
        const x1 = clamp(state.rangeLabelDrag.startX, bounds.left, bounds.right);
        const x2 = clamp(state.rangeLabelDrag.currentX, bounds.left, bounds.right);
        ctx.fillStyle = 'rgba(23, 63, 95, 0.12)';
        ctx.fillRect(Math.min(x1, x2), bounds.top, Math.abs(x2 - x1), bounds.bottom - bounds.top);
      }
      currentDietMarkers()
        .map((marker) => ({ ...marker, ms: parseMs(marker.timestamp) }))
        .filter((marker) => !Number.isNaN(marker.ms) && marker.ms >= startMs && marker.ms <= endMs)
        .forEach((marker, markerIdx) => {
          const frac = (marker.ms - startMs) / Math.max(1, endMs - startMs);
          const x = bounds.left + frac * (bounds.right - bounds.left);
          ctx.strokeStyle = 'rgba(193, 73, 83, 0.75)';
          ctx.setLineDash([5, 5]);
          ctx.beginPath();
          ctx.moveTo(x, bounds.top);
          ctx.lineTo(x, bounds.bottom);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.save();
          ctx.translate(x + 4, bounds.top + 14 + ((markerIdx % 2) * 13));
          ctx.rotate(-Math.PI / 10);
          ctx.fillStyle = '#8a2f39';
          ctx.font = '11px Helvetica';
          ctx.fillText(marker.label || 'meal', 0, 0);
          ctx.restore();
        });
      const sampled = downsampleSparse(values, 1200);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      let started = false;
      sampled.forEach(([idx, value]) => {
        const x = xForIndex(idx);
        const y = yForValue(value);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      });
      if (started) ctx.stroke();
      if (state.rangeLabelHoverIndex != null && rows[state.rangeLabelHoverIndex]) {
        const idx = clamp(state.rangeLabelHoverIndex, 0, rows.length - 1);
        const x = xForIndex(idx);
        ctx.strokeStyle = 'rgba(0,0,0,0.35)';
        ctx.beginPath();
        ctx.moveTo(x, bounds.top);
        ctx.lineTo(x, bounds.bottom);
        ctx.stroke();
      }
      ctx.fillStyle = '#555';
      ctx.font = '11px Helvetica';
      ctx.fillText(formatMetric(yr.min), 12, bounds.bottom);
      ctx.fillText(formatMetric(yr.max), 12, bounds.top + 4);
      ctx.fillText(rows[0].Timestamp, bounds.left, canvas.height - 18);
      const endLabel = rows[rows.length - 1].Timestamp;
      ctx.fillText(endLabel, bounds.right - ctx.measureText(endLabel).width, canvas.height - 18);
    }
    function drawRangeLabelPlots() {
      const rows = currentRows();
      if (!rows.length) {
        Object.keys(rangeCanvases).forEach((key) => rangeContexts[key].clearRect(0, 0, rangeCanvases[key].width, rangeCanvases[key].height));
        return;
      }
      const series = rangeLabelSeries(rows);
      drawRangeLabelPlot(rangeCanvases.cgm, rangeContexts.cgm, 'cgm', 'CGM', series.cgm, '#6c757d');
      drawRangeLabelPlot(rangeCanvases.seg, rangeContexts.seg, 'seg', 'SEG_converted_concentration_uM', series.seg, '#2a9d8f');
      drawRangeLabelPlot(rangeCanvases.sel, rangeContexts.sel, 'sel', 'SEL_converted_concentration_uM', series.sel, '#3a86ff');
    }
    function drawCurrentPlot(panelOnly = null) {
      const rows = currentRows();
      if (!rows.length) {
        plotRawCtx.clearRect(0, 0, plotRaw.width, plotRaw.height);
        plotConvertedCgmCtx.clearRect(0, 0, plotConvertedCgm.width, plotConvertedCgm.height);
        plotGluCgmCtx.clearRect(0, 0, plotGluCgm.width, plotGluCgm.height);
        plotGluLacCtx.clearRect(0, 0, plotGluLac.width, plotGluLac.height);
        Object.keys(rangeCanvases).forEach((key) => rangeContexts[key].clearRect(0, 0, rangeCanvases[key].width, rangeCanvases[key].height));
        return;
      }
      const series = getRenderSeries(rows);
      if (!panelOnly || panelOnly === 'raw') {
        renderPanelSafely('Raw Channels', () => {
          drawPlot(
            plotRaw,
            plotRawCtx,
            'Raw channels view: Channel1, Channel2, Channel3',
            [
              { label: 'Channel1', color: '#ef476f', values: series.rawCh1, options: { lineWidth: 1.7 } },
              { label: 'Channel2', color: '#06d6a0', values: series.rawCh2, options: { lineWidth: 1.7 } },
              { label: 'Channel3', color: '#118ab2', values: series.rawCh3, options: { lineWidth: 1.7 } }
            ],
            null
          );
        });
      }
      if (!panelOnly || panelOnly === 'converted_cgm') {
        renderPanelSafely('Converted Concentration vs CGM', () => {
          drawDualAxisPlot(
            plotConvertedCgm,
            plotConvertedCgmCtx,
            'Dual-axis view: converted concentration vs CGM',
            { label: conversionSourceLabel(), color: '#2a9d8f', values: series.convertedConc, options: { lineWidth: 1.8 } },
            { label: 'CGM', color: '#6c757d', values: series.cgmRaw, points: series.cgmPoints, options: { lineWidth: 1.6, showPoints: true, pointRadius: 2.1 } },
            null,
            'Converted concentration',
            'CGM'
          );
        });
      }
      if (!panelOnly || panelOnly === 'glu_cgm') {
        renderPanelSafely('SEB Glu vs CGM', () => {
          drawDualAxisPlot(
            plotGluCgm,
            plotGluCgmCtx,
            'Dual-axis alignment view: SEB Glu vs CGM',
            { label: 'SEB glucose proxy (Ch2-Ch1)', color: '#9d4edd', values: series.gluRaw, options: { lineWidth: 1.8 } },
            { label: 'CGM', color: '#6c757d', values: series.cgmRaw, points: series.cgmPoints, options: { lineWidth: 1.6, showPoints: true, pointRadius: 2.1 } },
            'glu_cgm',
            'SEB Glu',
            'CGM'
          );
        });
      }
      if (!panelOnly || panelOnly === 'glu_lac') {
        renderPanelSafely('SEB Glu vs Lac', () => {
          syncGluLacCanvasHeight();
          const glucoseLine = { label: 'SEB glucose proxy (Ch2-Ch1)', color: '#9d4edd', values: series.glucoseNorm, options: { lineWidth: 1.8 } };
          const lactateLine = { label: 'SEB lactate proxy (Ch2-Ch3)', color: '#3a86ff', values: series.lactateNorm, options: { lineWidth: 1.8 } };
          if (state.gluLacSplitView) {
            drawSplitGluLacPlot(
              plotGluLac,
              plotGluLacCtx,
              'Split normalized alignment view: SEB Glu above SEB Lac',
              glucoseLine,
              lactateLine
            );
          } else {
            drawPlot(
              plotGluLac,
              plotGluLacCtx,
              'Min-max normalized alignment view: SEB Glu vs Lac',
              [glucoseLine, lactateLine],
              'glu_lac'
            );
          }
        });
      }
      if (!panelOnly || panelOnly === 'range_label') {
        renderPanelSafely('Range Labeling', drawRangeLabelPlots);
      }
      document.getElementById('selection-label').textContent =
        'Active panel: ' + (state.activePanel === 'glu_cgm' ? 'SEB Glu vs CGM' : 'SEB Glu vs Lac') + ' | Smoothing extent: ' + state.smoothRatio.toFixed(2);
    }
    function formatMetric(value) {
      return value == null ? 'NA' : Number(value).toFixed(3);
    }
    function corrClass(value) {
      if (value == null || Number.isNaN(Number(value))) return '';
      const absVal = Math.abs(Number(value));
      if (absVal >= 0.7) return 'corr-strong';
      if (absVal >= 0.4) return 'corr-mid';
      return 'corr-weak';
    }
    function sectionRow(label) {
      return '<tr class="section-row"><th colspan="2">' + label + '</th></tr>';
    }
    function dataRow(label, value, className = '') {
      const cls = className ? ' class="' + className + '"' : '';
      return '<tr><th>' + label + '</th><td' + cls + '>' + value + '</td></tr>';
    }
    function updateSegmentTable() {
      const tbody = document.getElementById('segment-table');
      const statusEl = document.getElementById('segment-summary-status');
      if (!state.segmentSummaryEnabled) {
        tbody.innerHTML = '';
        statusEl.textContent = 'Summary disabled. Turn on the switch to calculate.';
        return;
      }
      statusEl.textContent = 'Summary enabled.';
      const rows = currentRows();
      const gluCgmSegment = currentEffectiveSegment('glu_cgm');
      const gluLacSegment = currentEffectiveSegment('glu_lac');
      if (!rows.length || !gluCgmSegment || !gluLacSegment) {
        tbody.innerHTML = '';
        return;
      }
      const gluCgmRows = rows.slice(gluCgmSegment.start_index, gluCgmSegment.end_index + 1);
      const gluLacRows = rows.slice(gluLacSegment.start_index, gluLacSegment.end_index + 1);
      const gluCgmMetrics = computeMetricsClient(gluCgmRows);
      const gluLacMetrics = computeMetricsClient(gluLacRows);
      const html = []
      html.push(sectionRow('Selection Context'));
      html.push(dataRow('Included in Glu-CGM cohort', currentIncluded('glu_cgm') ? 'Yes' : 'No'));
      html.push(dataRow('Included in Glu-Lac cohort', currentIncluded('glu_lac') ? 'Yes' : 'No'));
      html.push(dataRow('Active panel', state.activePanel === 'glu_cgm' ? 'SEB Glu vs CGM' : 'SEB Glu vs Lac'));
      html.push(dataRow('Lag search window', state.lagMinutes + ' min'));
      html.push(dataRow('Glu-CGM region pairs', (state.regionPairs.glu_cgm || []).length));
      html.push(dataRow('Glu-Lac region pairs', (state.regionPairs.glu_lac || []).length));

      html.push(sectionRow('SEB Glu vs CGM'));
      html.push(dataRow('Duration', formatDuration(gluCgmRows.length)));
      html.push(dataRow('Time range', (gluCgmRows[0]?.Timestamp || '') + ' to ' + (gluCgmRows[gluCgmRows.length - 1]?.Timestamp || '')));
      html.push(dataRow('Matched CGM points', gluCgmMetrics.matched_cgm_points));
      html.push(dataRow('Correlation', formatMetric(gluCgmMetrics.corr_glucose_cgm), corrClass(gluCgmMetrics.corr_glucose_cgm)));
      html.push(dataRow('Best-lag correlation', formatMetric(gluCgmMetrics.best_lag_corr_glucose_cgm), corrClass(gluCgmMetrics.best_lag_corr_glucose_cgm)));
      html.push(dataRow('Best-lag shift', formatLagSteps(gluCgmMetrics.best_lag_corr_glucose_cgm_steps)));
      html.push(dataRow('RMSE', formatMetric(gluCgmMetrics.rmse_glucose_cgm)));
      html.push(dataRow('AUC (SEB Glu, normalized)', formatMetric(gluCgmMetrics.auc_glucose)));
      html.push(dataRow('AUC (CGM, normalized)', formatMetric(gluCgmMetrics.auc_cgm)));
      html.push(dataRow('AUC similarity', formatMetric(gluCgmMetrics.auc_similarity_glucose_cgm), corrClass(gluCgmMetrics.auc_similarity_glucose_cgm)));
      html.push(dataRow('Direction agreement', formatMetric(gluCgmMetrics.direction_glucose_cgm)));
      html.push(dataRow('Best-lag direction', formatMetric(gluCgmMetrics.best_lag_direction_glucose_cgm)));
      html.push(dataRow('Best-lag direction shift', formatLagSteps(gluCgmMetrics.best_lag_direction_glucose_cgm_steps)));

      html.push(sectionRow('SEB Glu vs Lac'));
      html.push(dataRow('Duration', formatDuration(gluLacRows.length)));
      html.push(dataRow('Time range', (gluLacRows[0]?.Timestamp || '') + ' to ' + (gluLacRows[gluLacRows.length - 1]?.Timestamp || '')));
      html.push(dataRow('Correlation', formatMetric(gluLacMetrics.corr_glucose_lactate), corrClass(gluLacMetrics.corr_glucose_lactate)));
      html.push(dataRow('Best-lag correlation', formatMetric(gluLacMetrics.best_lag_corr_glucose_lactate), corrClass(gluLacMetrics.best_lag_corr_glucose_lactate)));
      html.push(dataRow('AUC (SEB Glu, normalized)', formatMetric(gluLacMetrics.auc_glucose)));
      html.push(dataRow('AUC (SEB Lac, normalized)', formatMetric(gluLacMetrics.auc_lactate)));
      html.push(dataRow('AUC similarity', formatMetric(gluLacMetrics.auc_similarity_glucose_lactate), corrClass(gluLacMetrics.auc_similarity_glucose_lactate)));
      html.push(dataRow('Best-lag shift', formatLagSteps(gluLacMetrics.best_lag_corr_glucose_lactate_steps)));
      html.push(dataRow('Best-lag direction', formatMetric(gluLacMetrics.best_lag_direction_glucose_lactate)));
      html.push(dataRow('Best-lag direction shift', formatLagSteps(gluLacMetrics.best_lag_direction_glucose_lactate_steps)));
      tbody.innerHTML = html.join('');
    }
    function formatDuration(totalSeconds) {
      const total = Math.max(0, Math.round(totalSeconds));
      const hours = Math.floor(total / 3600);
      const minutes = Math.floor((total % 3600) / 60);
      const seconds = total % 60;
      return [hours, minutes, seconds].map((n) => String(n).padStart(2, '0')).join(':');
    }
    function bestLagMetricClient(xs, ys, maxLagSteps, metricFn, minPairs = 3) {
      let bestValue = null;
      let bestLag = 0;
      for (let lag = -maxLagSteps; lag <= maxLagSteps; lag += 1) {
        const lagXs = [];
        const lagYs = [];
        let startX = 0;
        let startY = 0;
        let length = 0;
        if (lag >= 0) {
          startX = 0;
          startY = lag;
          length = Math.min(xs.length, ys.length) - lag;
        } else {
          startX = -lag;
          startY = 0;
          length = Math.min(xs.length, ys.length) + lag;
        }
        if (length <= 0) continue;
        for (let i = 0; i < length; i += 1) {
          const x = xs[startX + i];
          const y = ys[startY + i];
          if (x == null || y == null || Number.isNaN(x) || Number.isNaN(y)) continue;
          lagXs.push(x);
          lagYs.push(y);
        }
        if (lagXs.length < minPairs) continue;
        const value = metricFn(lagXs, lagYs);
        if (value == null) continue;
        if (bestValue == null || value > bestValue) {
          bestValue = value;
          bestLag = lag;
        }
      }
      return { value: bestValue, lagSteps: bestLag };
    }
    function computeMetricsClient(rows) {
      const matched = rows.filter((row) => row.CGM != null && !Number.isNaN(Number(row.CGM)));
      const glucose = smoothDense(matched.map((row) => Number(row.Delta_Channel2_minus_Channel1)), state.smoothRatio);
      const lactate = smoothDense(matched.map((row) => Number(row.Delta_Channel2_minus_Channel3)), state.smoothRatio);
      const cgm = matched.map((row) => Number(row.CGM));
      const glucoseAll = smoothDense(rows.map((row) => Number(row.Delta_Channel2_minus_Channel1)), state.smoothRatio);
      const lactateAll = smoothDense(rows.map((row) => Number(row.Delta_Channel2_minus_Channel3)), state.smoothRatio);
      const cgmAll = rows.map((row) => row.CGM == null ? null : Number(row.CGM));
      const maxLagSteps = Math.max(0, Math.round(state.lagMinutes * 60));
      const lagCorrGluCgm = bestLagMetricClient(glucoseAll, cgmAll, maxLagSteps, corr);
      const lagDirGluCgm = bestLagMetricClient(glucoseAll, cgmAll, maxLagSteps, direction);
      const lagCorrGluLac = bestLagMetricClient(glucoseAll, lactateAll, maxLagSteps, corr);
      const lagDirGluLac = bestLagMetricClient(glucoseAll, lactateAll, maxLagSteps, direction);
      const glucoseNorm = minmax(glucose);
      const lactateNorm = minmax(lactate);
      const cgmNorm = minmax(cgm);
      const aucGlucose = aucTrapezoid(glucoseNorm);
      const aucLactate = aucTrapezoid(lactateNorm);
      const aucCgm = aucTrapezoid(cgmNorm);
      return {
        matched_cgm_points: matched.length,
        corr_glucose_cgm: corr(glucose, cgm),
        corr_lactate_cgm: corr(lactate, cgm),
        corr_glucose_lactate: corr(glucose, lactate),
        rmse_glucose_cgm: rmse(glucoseNorm, cgmNorm),
        rmse_lactate_cgm: rmse(lactateNorm, cgmNorm),
        direction_glucose_cgm: direction(glucose, cgm),
        direction_lactate_cgm: direction(lactate, cgm),
        auc_glucose: aucGlucose,
        auc_lactate: aucLactate,
        auc_cgm: aucCgm,
        auc_similarity_glucose_cgm: aucSimilarity(aucGlucose, aucCgm),
        auc_similarity_glucose_lactate: aucSimilarity(aucGlucose, aucLactate),
        best_lag_corr_glucose_cgm: lagCorrGluCgm.value,
        best_lag_corr_glucose_cgm_steps: lagCorrGluCgm.lagSteps,
        best_lag_direction_glucose_cgm: lagDirGluCgm.value,
        best_lag_direction_glucose_cgm_steps: lagDirGluCgm.lagSteps,
        best_lag_corr_glucose_lactate: lagCorrGluLac.value,
        best_lag_corr_glucose_lactate_steps: lagCorrGluLac.lagSteps,
        best_lag_direction_glucose_lactate: lagDirGluLac.value,
        best_lag_direction_glucose_lactate_steps: lagDirGluLac.lagSteps,
      };
    }
    function formatLagSteps(steps) {
      if (steps == null) return 'NA';
      const sign = steps > 0 ? '+' : '';
      return sign + Number(steps).toFixed(0) + ' s';
    }
    function corr(xs, ys) {
      const n = Math.min(xs.length, ys.length);
      if (n < 2) return null;
      const mx = xs.reduce((a, b) => a + b, 0) / n;
      const my = ys.reduce((a, b) => a + b, 0) / n;
      let vx = 0, vy = 0, cov = 0;
      for (let i = 0; i < n; i += 1) {
        vx += (xs[i] - mx) ** 2;
        vy += (ys[i] - my) ** 2;
        cov += (xs[i] - mx) * (ys[i] - my);
      }
      if (vx <= 1e-12 || vy <= 1e-12) return null;
      return cov / Math.sqrt(vx * vy);
    }
    function aucTrapezoid(values) {
      if (!values || values.length < 2) return null;
      let total = 0;
      let hasPair = false;
      for (let i = 1; i < values.length; i += 1) {
        const a = values[i - 1];
        const b = values[i];
        if (a == null || b == null || Number.isNaN(a) || Number.isNaN(b)) continue;
        total += (a + b) * 0.5;
        hasPair = true;
      }
      return hasPair ? total : null;
    }
    function aucSimilarity(a, b) {
      if (a == null || b == null) return null;
      const aa = Math.abs(a);
      const bb = Math.abs(b);
      const denom = Math.max(aa, bb);
      if (denom <= 1e-12) return null;
      return 1 - (Math.abs(aa - bb) / denom);
    }
    function cgmRangesByDay(rows, maxDays = 3) {
      const buckets = new Map();
      rows.forEach((row) => {
        const ts = String(row?.Timestamp || '');
        const day = ts.slice(0, 10);
        const n = row?.CGM == null ? null : Number(row.CGM);
        if (!day || n == null || Number.isNaN(n)) return;
        if (!buckets.has(day)) buckets.set(day, []);
        buckets.get(day).push(n);
      });
      return Array.from(buckets.keys()).sort().slice(0, maxDays).map((day, i) => {
        const vals = buckets.get(day) || [];
        return { label: 'Day ' + (i + 1), min: Math.min(...vals), max: Math.max(...vals) };
      });
    }
    function rmse(xs, ys) {
      const n = Math.min(xs.length, ys.length);
      if (!n) return null;
      let total = 0;
      for (let i = 0; i < n; i += 1) total += (xs[i] - ys[i]) ** 2;
      return Math.sqrt(total / n);
    }
    function direction(xs, ys) {
      const n = Math.min(xs.length, ys.length);
      if (n < 2) return null;
      let agree = 0;
      for (let i = 1; i < n; i += 1) {
        const dx = xs[i] - xs[i - 1];
        const dy = ys[i] - ys[i - 1];
        const sx = Math.abs(dx) < 1e-12 ? 0 : (dx > 0 ? 1 : -1);
        const sy = Math.abs(dy) < 1e-12 ? 0 : (dy > 0 ? 1 : -1);
        if (sx === sy) agree += 1;
      }
      return agree / (n - 1);
    }
    function timestampListMs() {
      const rows = currentRows();
      return rows.map((row) => parseMs(row.Timestamp));
    }
    function timeMsFromCanvasX(x) {
      const times = timestampListMs();
      if (!times.length) return null;
      const startMs = times[0];
      const endMs = times[times.length - 1];
      const frac = clamp((x - state.bounds.left) / Math.max(1, state.bounds.right - state.bounds.left), 0, 1);
      return startMs + frac * (endMs - startMs);
    }
    function nearestCgmPointForTime(targetMs) {
      const rows = currentRows();
      const series = getRenderSeries(rows);
      const points = series?.cgmPoints || [];
      if (!Number.isFinite(targetMs) || !points.length) return null;
      let best = null;
      let bestDist = Infinity;
      points.forEach((point) => {
        const d = Math.abs(point.ms - targetMs);
        if (d < bestDist) {
          bestDist = d;
          best = point;
        }
      });
      if (!best) return null;
      const idx = nearestIndexForTimestampMs(best.ms);
      return { ...best, index: idx };
    }
    function nearestIndexForTimestampMs(targetMs) {
      const times = timestampListMs();
      if (!times.length || !Number.isFinite(targetMs)) return 0;
      let bestIdx = 0;
      let bestDist = Infinity;
      for (let i = 0; i < times.length; i += 1) {
        const d = Math.abs(times[i] - targetMs);
        if (d < bestDist) {
          bestDist = d;
          bestIdx = i;
        }
      }
      return bestIdx;
    }
    function applyCgmLabelPosition(entry, targetMs) {
      const point = nearestCgmPointForTime(targetMs);
      if (!point) return false;
      entry.index = point.index;
      entry.timestamp = point.timestamp || '';
      entry.timestamp_ms = point.ms;
      entry.cgm_value = point.value;
      return true;
    }
    function pointToIndexFromCanvasX(x) {
      const rows = currentRows();
      const times = timestampListMs();
      if (!rows.length || !times.length) return 0;
      const startMs = times[0];
      const endMs = times[times.length - 1];
      const width = Math.max(1, state.bounds.right - state.bounds.left);
      const xForIdx = (idx) => {
        const cIdx = clamp(idx, 0, times.length - 1);
        return state.bounds.left + ((times[cIdx] - startMs) / Math.max(1, endMs - startMs)) * width;
      };
      // Snap to nearest actual row entry shown in the time series (entry index), not pixel interpolation.
      let bestIdx = 0;
      let bestDist = null;
      for (let i = 0; i < times.length; i += 1) {
        const xi = xForIdx(i);
        const d = Math.abs(xi - x);
        if (bestDist == null || d < bestDist) {
          bestDist = d;
          bestIdx = i;
        }
      }
      return bestIdx;
    }
    function interactiveBoundsForPanel(panel) {
      if (panel === 'glu_lac' && state.gluLacSplitView) {
        return {
          left: state.bounds.left,
          right: state.bounds.right,
          top: state.bounds.top,
          bottom: Math.max(state.bounds.bottom, plotGluLac.height - 80)
        };
      }
      return state.bounds;
    }
    function markerHitTest(canvas, panel, event) {
      const rows = currentRows();
      if (!rows.length) return null;
      const rect = canvas.getBoundingClientRect();
      const x = (event.clientX - rect.left) * (canvas.width / rect.width);
      const y = (event.clientY - rect.top) * (canvas.height / rect.height);
      const controlBounds = panel === 'glu_lac' && state.gluLacSplitView ? gluLacSplitBounds().bottom : state.bounds;
      if (y < controlBounds.bottom + 8 || y > controlBounds.bottom + 54) return null;
      const times = timestampListMs();
      const startIdx = state.markerIndex[panel].start;
      const endIdx = state.markerIndex[panel].end;
      const xForIdx = (idx) => {
        if (idx == null) return null;
        const cIdx = clamp(idx, 0, times.length - 1);
        const startMs = times[0];
        const endMs = times[times.length - 1];
        return state.bounds.left + ((times[cIdx] - startMs) / Math.max(1, endMs - startMs)) * (state.bounds.right - state.bounds.left);
      };
      const sx = xForIdx(startIdx);
      const ex = xForIdx(endIdx);
      if (sx != null && Math.abs(x - sx) <= 10) return 'start';
      if (ex != null && Math.abs(x - ex) <= 10) return 'end';
      return null;
    }
    function labelHitTest(canvas, panel, event) {
      const rows = currentRows();
      if (!rows.length) return null;
      const rect = canvas.getBoundingClientRect();
      const x = (event.clientX - rect.left) * (canvas.width / rect.width);
      const y = (event.clientY - rect.top) * (canvas.height / rect.height);
      const controlBounds = panel === 'glu_lac' && state.gluLacSplitView ? gluLacSplitBounds().bottom : state.bounds;
      if (y < controlBounds.bottom + 28 || y > controlBounds.bottom + 56) return null;
      const times = timestampListMs();
      const labels = state.labelPoints[panel] || [];
      const xForIdx = (idx) => {
        const cIdx = clamp(idx ?? 0, 0, times.length - 1);
        const startMs = times[0];
        const endMs = times[times.length - 1];
        return state.bounds.left + ((times[cIdx] - startMs) / Math.max(1, endMs - startMs)) * (state.bounds.right - state.bounds.left);
      };
      for (let i = 0; i < labels.length; i += 1) {
        const explicitMs = labelTimeMs(labels[i], times);
        const lx = Number.isFinite(explicitMs)
          ? state.bounds.left + ((explicitMs - times[0]) / Math.max(1, times[times.length - 1] - times[0])) * (state.bounds.right - state.bounds.left)
          : xForIdx(labels[i].index);
        if (Math.abs(x - lx) <= 10) return i;
      }
      return null;
    }
    function attachSelectionHandlers(canvas, panel) {
      canvas.addEventListener('mousedown', (event) => {
        const labelIdx = labelHitTest(canvas, panel, event);
        if (labelIdx != null) {
          setActivePanel(panel);
          focusLabelLog(panel, labelIdx);
          state.labelDrag = { panel, index: labelIdx };
          return;
        }
        const kind = markerHitTest(canvas, panel, event);
        if (!kind) return;
        setActivePanel(panel);
        state.markerDrag = { panel, kind };
      });
      canvas.addEventListener('click', (event) => {
        if (state.markerDrag.panel === panel) return;
        if (!currentRows().length) return;
        const rect = canvas.getBoundingClientRect();
        const x = (event.clientX - rect.left) * (canvas.width / rect.width);
        const y = (event.clientY - rect.top) * (canvas.height / rect.height);
        const activeBounds = interactiveBoundsForPanel(panel);
        if (x < activeBounds.left || x > activeBounds.right || y < activeBounds.top || y > activeBounds.bottom) return;
        setActivePanel(panel);
        state.cursorTimeMs[panel] = timeMsFromCanvasX(x);
        state.cursorIndex[panel] = pointToIndexFromCanvasX(x);
        drawCurrentPlot(panel);
        updateRegionStatus();
      });
      canvas.addEventListener('mousemove', (event) => {
        if (state.labelDrag.panel === panel) {
          const rect = canvas.getBoundingClientRect();
          const x = (event.clientX - rect.left) * (canvas.width / rect.width);
          const targetMs = timeMsFromCanvasX(x);
          state.cursorIndex[panel] = pointToIndexFromCanvasX(x);
          state.cursorTimeMs[panel] = targetMs;
          const dragged = state.labelPoints[panel][state.labelDrag.index];
          if (isCgmLabelName(dragged?.name)) {
            applyCgmLabelPosition(dragged, targetMs);
          } else {
            dragged.index = state.cursorIndex[panel];
            dragged.timestamp = '';
            dragged.timestamp_ms = null;
            dragged.cgm_value = null;
          }
          if (!state.dragRafPending) {
            state.dragRafPending = true;
            requestAnimationFrame(() => {
              state.dragRafPending = false;
              drawCurrentPlot(panel);
              updateRegionStatus();
              renderLabelLogs();
            });
          }
          return;
        }
        if (state.markerDrag.panel !== panel) return;
        const rect = canvas.getBoundingClientRect();
        const x = (event.clientX - rect.left) * (canvas.width / rect.width);
        state.cursorTimeMs[panel] = timeMsFromCanvasX(x);
        state.cursorIndex[panel] = pointToIndexFromCanvasX(x);
        state.markerIndex[panel][state.markerDrag.kind] = state.cursorIndex[panel];
        if (!state.dragRafPending) {
          state.dragRafPending = true;
          requestAnimationFrame(() => {
            state.dragRafPending = false;
            drawCurrentPlot(panel);
            updateRegionStatus();
          });
        }
      });
      canvas.addEventListener('mouseup', () => {
        if (state.labelDrag.panel === panel) {
          const current = currentMeta();
          const labelIndex = state.labelDrag.index;
          state.labelDrag = { panel: null, index: null };
          if (current && labelIndex != null) {
            const entry = state.labelPoints[panel][labelIndex];
            const sourceIndex = Number.isFinite(Number(entry?.source_index)) ? Number(entry.source_index) : labelIndex;
            fetchJson('/api/labels/move', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                path: current.path,
                panel,
                label_index: sourceIndex,
                index: entry.index,
                timestamp: entry.timestamp || '',
                timestamp_ms: entry.timestamp_ms,
                cgm_value: entry.cgm_value
              })
            }).then((data) => {
              state.savedSegments = data.saved_segments || state.savedSegments;
              refreshLabelStateForPanel(current.path, panel);
              renderLabelLogs();
              renderList();
            }).catch((error) => setStatus(error.message));
          }
        }
        if (state.markerDrag.panel === panel) state.markerDrag = { panel: null, kind: null };
      });
    }
    function attachRawPlotHandlers() {
      plotRaw.addEventListener('mousedown', (event) => {
        if (!currentRows().length) return;
        const p = rawPointFromEvent(event);
        if (p.x < state.bounds.left || p.x > state.bounds.right || p.y < state.bounds.top || p.y > state.bounds.bottom) return;
        state.rawDrag = { active: true, startX: p.x, currentX: p.x };
        state.rawHoverIndex = nearestRawIndexForX(p.x);
        updateRawHoverDetail(state.rawHoverIndex);
        drawCurrentPlot('raw');
      });
      plotRaw.addEventListener('mousemove', (event) => {
        if (!currentRows().length) return;
        const p = rawPointFromEvent(event);
        if (state.rawDrag.active) {
          state.rawDrag.currentX = clamp(p.x, state.bounds.left, state.bounds.right);
          state.rawHoverIndex = nearestRawIndexForX(state.rawDrag.currentX);
          updateRawHoverDetail(state.rawHoverIndex);
          drawCurrentPlot('raw');
          return;
        }
        if (p.x < state.bounds.left || p.x > state.bounds.right || p.y < state.bounds.top || p.y > state.bounds.bottom) {
          state.rawHoverIndex = null;
          updateRawHoverDetail(null);
          drawCurrentPlot('raw');
          return;
        }
        state.rawHoverIndex = nearestRawIndexForX(p.x);
        updateRawHoverDetail(state.rawHoverIndex);
        drawCurrentPlot('raw');
      });
      plotRaw.addEventListener('mouseup', () => {
        if (!state.rawDrag.active) return;
        const startX = state.rawDrag.startX;
        const endX = state.rawDrag.currentX;
        state.rawDrag = { active: false, startX: null, currentX: null };
        if (startX != null && endX != null && Math.abs(endX - startX) >= 12) {
          const a = rawTimeForX(startX);
          const b = rawTimeForX(endX);
          if (a != null && b != null && Math.abs(b - a) > 1000) {
            state.rawZoom = { startMs: Math.min(a, b), endMs: Math.max(a, b) };
          }
        }
        drawCurrentPlot('raw');
      });
      plotRaw.addEventListener('mouseleave', () => {
        if (state.rawDrag.active) return;
        state.rawHoverIndex = null;
        updateRawHoverDetail(null);
        drawCurrentPlot('raw');
      });
    }
    function attachConvertedCgmHandlers() {
      plotConvertedCgm.addEventListener('mousemove', (event) => {
        if (!currentRows().length) return;
        const p = canvasPoint(plotConvertedCgm, event);
        if (p.x < state.bounds.left || p.x > state.bounds.right || p.y < state.bounds.top || p.y > state.bounds.bottom) {
          state.convertedHoverIndex = null;
          updateConvertedHoverDetail(null);
          drawCurrentPlot('converted_cgm');
          return;
        }
        state.convertedHoverIndex = nearestIndexForCanvasX(plotConvertedCgm, p.x);
        updateConvertedHoverDetail(state.convertedHoverIndex);
        drawCurrentPlot('converted_cgm');
      });
      plotConvertedCgm.addEventListener('mouseleave', () => {
        state.convertedHoverIndex = null;
        updateConvertedHoverDetail(null);
        drawCurrentPlot('converted_cgm');
      });
    }
    function updateRangeLabelStatus(message = null) {
      const el = document.getElementById('range-label-status');
      if (!el) return;
      if (message) {
        el.textContent = message;
        return;
      }
      const rows = currentRows();
      const selection = rangeSelectionBounds();
      if (!rows.length || !selection) {
        el.textContent = 'Drag across any one of the three plots to select a shared time range. Display smoothing extent: ' + state.smoothRatio.toFixed(2) + '.';
        return;
      }
      const startTs = rows[selection.start]?.Timestamp || 'NA';
      const endTs = rows[selection.end]?.Timestamp || 'NA';
      const startMs = parseMs(startTs);
      const endMs = parseMs(endTs);
      const meals = currentDietMarkers().filter((marker) => {
        const ms = parseMs(marker.timestamp);
        return !Number.isNaN(ms) && ms >= startMs && ms <= endMs;
      });
      el.textContent = 'Selected idx ' + selection.start + '-' + selection.end + ' | ' + startTs + ' to ' + endTs + ' | smoothing: ' + state.smoothRatio.toFixed(2) + ' | meal timestamps: ' + (meals.map((m) => m.timestamp).join(', ') || 'none in range');
    }
    async function loadRangeLabelHistory() {
      const meta = currentMeta();
      if (!meta) {
        state.rangeLabelHistory = [];
        renderRangeLabelHistory();
        return;
      }
      try {
        const data = await fetchJson('/api/range-label-history?path=' + encodeURIComponent(meta.path));
        state.rangeLabelHistory = data.history || [];
      } catch (error) {
        state.rangeLabelHistory = [];
        setStatus('Failed to load range label history: ' + error.message);
      }
      renderRangeLabelHistory();
    }
    function applyRangeHistoryEntry(entry) {
      const rows = currentRows();
      if (!rows.length || !entry) return;
      const startMs = parseMs(entry.start_timestamp || '');
      const endMs = parseMs(entry.end_timestamp || '');
      if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) {
        setStatus('Cannot reload history item with invalid timestamps.');
        return;
      }
      state.rangeLabelSelection = {
        start: nearestIndexForTimestampMs(startMs),
        end: nearestIndexForTimestampMs(endMs)
      };
      state.rangeLabelDrag = { active: false, source: null, startX: null, currentX: null };
      updateRangeLabelStatus('Reloaded history range: ' + entry.start_timestamp + ' to ' + entry.end_timestamp + '.');
      drawCurrentPlot('range_label');
    }
    async function deleteRangeHistoryEntry(entry) {
      const meta = currentMeta();
      if (!meta || !entry) return;
      try {
        const data = await fetchJson('/api/range-labels/delete', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            id: entry.id || '',
            path: meta.path,
            start_timestamp: entry.start_timestamp || '',
            end_timestamp: entry.end_timestamp || ''
          })
        });
        state.rangeLabelHistory = data.history || [];
        const selection = rangeSelectionBounds();
        const rows = currentRows();
        if (selection && rows[selection.start]?.Timestamp === entry.start_timestamp && rows[selection.end]?.Timestamp === entry.end_timestamp) {
          state.rangeLabelSelection = { start: null, end: null };
        }
        renderRangeLabelHistory();
        updateRangeLabelStatus('Deleted range history entry and removed ' + data.removed_ranges + ' saved full range record(s).');
        drawCurrentPlot('range_label');
      } catch (error) {
        setStatus('Failed to delete range history entry: ' + error.message);
      }
    }
    function renderRangeLabelHistory() {
      const el = document.getElementById('range-label-history-list');
      if (!el) return;
      const history = state.rangeLabelHistory || [];
      if (!history.length) {
        el.innerHTML = '<div class="small">No saved range history for this file.</div>';
        return;
      }
      el.innerHTML = history.slice().reverse().map((entry, idx) => {
        const sourceIndex = history.length - 1 - idx;
        const mealCount = (entry.meal_timestamps || []).length;
        return (
          '<div class="log-row">' +
          '<div style="min-width:0;">' +
          '<div><strong>' + (entry.start_timestamp || 'NA') + '</strong></div>' +
          '<div class="small">to ' + (entry.end_timestamp || 'NA') + '</div>' +
          '<div class="small">smooth ' + (entry.smoothing_ratio ?? 'NA') + ' | meals ' + mealCount + (entry.note ? ' | ' + entry.note : '') + '</div>' +
          '</div>' +
          '<div style="display:flex; gap:4px; flex-wrap:wrap; justify-content:flex-end;">' +
          '<button class="tiny-btn" data-range-history-index="' + sourceIndex + '">Load</button>' +
          '<button class="tiny-btn" data-range-history-delete-index="' + sourceIndex + '">Delete</button>' +
          '</div>' +
          '</div>'
        );
      }).join('');
      el.querySelectorAll('button[data-range-history-index]').forEach((button) => {
        button.addEventListener('click', () => {
          const idx = Number(button.getAttribute('data-range-history-index'));
          applyRangeHistoryEntry(history[idx]);
        });
      });
      el.querySelectorAll('button[data-range-history-delete-index]').forEach((button) => {
        button.addEventListener('click', () => {
          const idx = Number(button.getAttribute('data-range-history-delete-index'));
          deleteRangeHistoryEntry(history[idx]);
        });
      });
    }
    function rangeIndexForEvent(canvas, event) {
      const p = canvasPoint(canvas, event);
      return nearestIndexForCanvasX(canvas, p.x);
    }
    function attachRangeLabelHandlers() {
      Object.entries(rangeCanvases).forEach(([key, canvas]) => {
        canvas.addEventListener('mousedown', (event) => {
          if (!currentRows().length) return;
          const p = canvasPoint(canvas, event);
          state.rangeLabelDrag = { active: true, source: key, startX: p.x, currentX: p.x };
          const idx = rangeIndexForEvent(canvas, event);
          state.rangeLabelSelection = { start: idx, end: idx };
          updateRangeLabelStatus();
          drawCurrentPlot('range_label');
        });
        canvas.addEventListener('mousemove', (event) => {
          if (!currentRows().length) return;
          const p = canvasPoint(canvas, event);
          state.rangeLabelHoverIndex = nearestIndexForCanvasX(canvas, p.x);
          if (state.rangeLabelDrag.active && state.rangeLabelDrag.source === key) {
            state.rangeLabelDrag.currentX = p.x;
            state.rangeLabelSelection.end = rangeIndexForEvent(canvas, event);
            updateRangeLabelStatus();
          }
          drawCurrentPlot('range_label');
        });
        canvas.addEventListener('mouseup', (event) => {
          if (state.rangeLabelDrag.active && state.rangeLabelDrag.source === key) {
            state.rangeLabelSelection.end = rangeIndexForEvent(canvas, event);
            state.rangeLabelDrag = { active: false, source: null, startX: null, currentX: null };
            updateRangeLabelStatus();
            drawCurrentPlot('range_label');
          }
        });
        canvas.addEventListener('mouseleave', () => {
          state.rangeLabelHoverIndex = null;
          drawCurrentPlot('range_label');
        });
      });
    }
    attachRawPlotHandlers();
    attachConvertedCgmHandlers();
    attachRangeLabelHandlers();
    attachSelectionHandlers(plotGluCgm, 'glu_cgm');
    attachSelectionHandlers(plotGluLac, 'glu_lac');
    window.addEventListener('mouseup', () => {
      if (state.rawDrag.active) {
        state.rawDrag = { active: false, startX: null, currentX: null };
        drawCurrentPlot('raw');
      }
      if (state.rangeLabelDrag.active) {
        state.rangeLabelDrag = { active: false, source: null, startX: null, currentX: null };
        drawCurrentPlot('range_label');
      }
      state.markerDrag = { panel: null, kind: null };
      state.labelDrag = { panel: null, index: null };
    });
    document.getElementById('clear-range-selection').addEventListener('click', () => {
      state.rangeLabelSelection = { start: null, end: null };
      state.rangeLabelHoverIndex = null;
      updateRangeLabelStatus();
      drawCurrentPlot('range_label');
      setStatus('Cleared range labeling selection.');
    });
    document.getElementById('save-range-selection').addEventListener('click', async () => {
      const meta = currentMeta();
      const selection = rangeSelectionBounds();
      if (!meta || !selection) {
        updateRangeLabelStatus('Select a range before saving.');
        return;
      }
      try {
        const note = document.getElementById('range-label-note')?.value || '';
        const data = await fetchJson('/api/range-labels/save', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            path: meta.path,
            start_index: selection.start,
            end_index: selection.end,
            label_source: 'three_series_range_plot',
            note,
            smoothing_ratio: state.smoothRatio
          })
        });
        updateRangeLabelStatus('Saved range ' + data.selection.start_index + '-' + data.selection.end_index + ' with ' + data.selection.values.length + ' rows, smoothing ' + data.selection.smoothing_ratio + ', and ' + data.selection.meal_timestamps.length + ' meal timestamp(s) to ' + data.output_path + '.');
        await loadRangeLabelHistory();
        setStatus('Saved range label JSON.');
      } catch (error) {
        updateRangeLabelStatus('Failed to save range: ' + error.message);
      }
    });
    document.getElementById('reset-raw-zoom').addEventListener('click', () => {
      state.rawZoom = { startMs: null, endMs: null };
      state.rawHoverIndex = null;
      state.rawDrag = { active: false, startX: null, currentX: null };
      updateRawHoverDetail(null);
      drawCurrentPlot('raw');
      setStatus('Reset raw channel zoom.');
    });
    document.getElementById('apply-lag').addEventListener('click', async () => {
      const input = document.getElementById('lag-minutes-input');
      state.lagMinutes = Math.max(0, Math.min(180, Number(input.value || 0)));
      input.value = String(state.lagMinutes);
      updateSegmentTable();
      setStatus('Applied lag search window of ' + state.lagMinutes + ' minutes.');
    });
    document.getElementById('enable-segment-summary').addEventListener('change', (event) => {
      state.segmentSummaryEnabled = !!event.target.checked;
      updateSegmentTable();
      setStatus(state.segmentSummaryEnabled ? 'Segment summary calculation enabled.' : 'Segment summary calculation disabled.');
    });
    document.getElementById('apply-smoothing').addEventListener('click', async () => {
      const input = document.getElementById('smooth-ratio-input');
      state.smoothRatio = Math.max(0, Math.min(0.30, Number(input.value || 0)));
      input.value = state.smoothRatio.toFixed(2);
      state.renderCache.clear();
      drawCurrentPlot();
      updateRangeLabelStatus();
      updateSegmentTable();
      setStatus('Applied smoothing extent of ' + state.smoothRatio.toFixed(2) + '.');
    });
    document.getElementById('export-conversion-toggle').addEventListener('change', (event) => {
      state.exportConversionOnApply = !!event.target.checked;
      document.getElementById('conversion-status').textContent = state.exportConversionOnApply ? 'Export enabled for the next apply.' : 'Preview only. Export toggle is off.';
    });
    document.getElementById('show-negative-regions-toggle').addEventListener('change', (event) => {
      state.showNegativeRegions = !!event.target.checked;
      drawCurrentPlot('converted_cgm');
      document.getElementById('conversion-status').textContent = state.showNegativeRegions ? 'Negative-region shading enabled.' : 'Negative-region shading disabled.';
    });
    document.getElementById('open-analysis-results').addEventListener('click', () => {
      window.open('/analysis', '_blank');
    });
    document.getElementById('conversion-source').addEventListener('change', (event) => {
      state.conversionSource = event.target.value === 'sel' ? 'sel' : 'seg';
      state.renderCache.clear();
      drawCurrentPlot('converted_cgm');
      document.getElementById('conversion-status').textContent = 'Updated conversion source. Preview refreshed.';
    });
    document.getElementById('apply-conversion').addEventListener('click', async () => {
      const meta = currentMeta();
      const input = document.getElementById('conversion-smoothing');
      state.conversionSmoothing = Math.max(0, Math.min(0.30, Number(input.value || 0)));
      input.value = state.conversionSmoothing.toFixed(2);
      state.conversionSource = document.getElementById('conversion-source').value === 'sel' ? 'sel' : 'seg';
      state.renderCache.clear();
      drawCurrentPlot('converted_cgm');
      if (!state.exportConversionOnApply || !meta) {
        document.getElementById('conversion-status').textContent = 'Preview refreshed. Export toggle is off.';
        return;
      }
      try {
        const data = await fetchJson('/api/converted/export', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: meta.path, source: state.conversionSource, smoothing_ratio: state.conversionSmoothing })
        });
        document.getElementById('conversion-status').textContent = 'Exported ' + data.row_count + ' rows to ' + data.output_path;
      } catch (error) {
        document.getElementById('conversion-status').textContent = 'Export failed: ' + error.message;
      }
    });
    function bindLabelFilterToggle(id, key, message) {
      const el = document.getElementById(id);
      if (!el) return;
      el.checked = !!state.labelFilter[key];
      el.addEventListener('change', (event) => {
        state.labelFilter[key] = !!event.target.checked;
        drawCurrentPlot();
        setStatus(message);
      });
    }
    bindLabelFilterToggle('show-labels-toggle', 'show', 'Updated label visibility.');
    bindLabelFilterToggle('show-label-cgm', 'cgm', 'Updated CGM label visibility.');
    bindLabelFilterToggle('show-label-seg', 'seg', 'Updated SEG label visibility.');
    bindLabelFilterToggle('show-label-sel', 'sel', 'Updated SEL label visibility.');
    bindLabelFilterToggle('show-label-peak', 'peak', 'Updated peak label visibility.');
    bindLabelFilterToggle('show-label-trough', 'trough', 'Updated trough label visibility.');
    document.getElementById('glu-lac-split-view-toggle').addEventListener('change', (event) => {
      state.gluLacSplitView = !!event.target.checked;
      drawCurrentPlot('glu_lac');
      setStatus(state.gluLacSplitView ? 'SEB Glu vs Lac split view enabled.' : 'SEB Glu vs Lac overlay view enabled.');
    });
    function bindLabelModeToggle(id, panel) {
      const el = document.getElementById(id);
      if (!el) return;
      el.addEventListener('change', (event) => {
        if (!event.target.checked) {
          event.target.checked = true;
          return;
        }
        setActivePanel(panel);
        drawCurrentPlot();
        setStatus('Active labeling lane switched to ' + panelLabel(panel) + '.');
      });
    }
    bindLabelModeToggle('label-mode-glu-cgm', 'glu_cgm');
    bindLabelModeToggle('label-mode-glu-lac', 'glu_lac');
    function bindRefreshPeakAggregates(panel) {
      const id = 'refresh-peak-aggregates-' + panel.replace('_', '-');
      const el = document.getElementById(id);
      if (!el) return;
      el.addEventListener('click', async () => {
        try {
          setActivePanel(panel);
          await loadPeakFeatureData();
          setStatus('Refreshed ' + panelLabel(panel) + ' peak aggregates from current annotations.');
        } catch (error) {
          setStatus('Failed to refresh peak aggregates: ' + error.message);
        }
      });
    }
    function bindSavePeakPair(panel) {
      const button = document.getElementById('save-peak-pair-' + panel.replace('_', '-'));
      if (!button) return;
      button.addEventListener('click', async () => {
      const meta = currentMeta();
      if (!meta) return;
      setActivePanel(panel);
      const ids = pairControlIds(panel);
      const segId = document.getElementById(ids.seg)?.value || '';
      const selId = panel === 'glu_lac' ? (document.getElementById(ids.sel)?.value || '') : '';
      const cgmId = panel === 'glu_cgm' ? (document.getElementById(ids.cgm)?.value || '') : '';
      if (panel === 'glu_lac') {
        if (segId === '' || !selId) {
          setStatus('Choose one fixed CGM-SEG instance and one SEL aggregate to create/update a triplet.');
          return;
        }
        try {
          const data = await fetchJson('/api/peak/pairs/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: meta.path, panel, pair_index: Number(segId), sel_id: selId })
          });
          state.peakPairs = data.paired_features || [];
          renderPeakPairControls();
          drawCurrentPlot();
          setStatus('Attached SEL to the selected CGM-SEG triplet.');
        } catch (error) {
          setStatus('Failed to attach SEL to triplet: ' + error.message);
        }
        return;
      }
      const picked = [segId, selId, cgmId].filter(Boolean).length;
      if (picked !== 2) {
        setStatus('Select exactly two aggregates to create a paired peak feature.');
        return;
      }
      try {
        const data = await fetchJson('/api/peak/pairs/save', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: meta.path, panel, seg_id: segId, sel_id: selId, cgm_id: cgmId })
        });
        state.peakPairs = data.paired_features || [];
        renderPeakPairList(panel);
        setStatus('Saved paired peak feature.');
      } catch (error) {
        setStatus('Failed to save paired peak feature: ' + error.message);
      }
      });
    }
    bindRefreshPeakAggregates('glu_cgm');
    bindRefreshPeakAggregates('glu_lac');
    bindSavePeakPair('glu_cgm');
    bindSavePeakPair('glu_lac');
    ['pair-seg-glu-cgm', 'pair-cgm-glu-cgm', 'pair-seg-glu-lac', 'pair-sel-glu-lac'].forEach((id) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.addEventListener('change', () => {
        const panel = id.endsWith('glu-lac') ? 'glu_lac' : 'glu_cgm';
        setActivePanel(panel);
        if (id === 'pair-seg-glu-lac') {
          const idx = el.value === '' ? null : Number(el.value);
          state.selectedPeakPairIndex.glu_lac = Number.isFinite(idx) ? idx : null;
          const item = idx == null ? null : (state.peakPairs || [])[idx];
          if (item) selectPeakPair(item, idx, 'glu_lac');
          renderPeakPairList('glu_lac');
        }
        drawCurrentPlot();
      });
    });
    function wirePanelControls(panel) {
      const suffix = panel === 'glu_cgm' ? 'glu-cgm' : 'glu-lac';
      document.getElementById('add-label-' + suffix).addEventListener('click', () => {
        setActivePanel(panel);
        const input = document.getElementById('new-label-name-' + suffix);
        const name = (input?.value || 'CGM peak').trim();
        const entry = { name, index: 0, timestamp: '', timestamp_ms: null, cgm_value: null };
        if (isCgmLabelName(name)) {
          const firstCgm = (getRenderSeries(currentRows())?.cgmPoints || []).sort((a, b) => a.ms - b.ms)[0];
          if (firstCgm) {
            entry.index = nearestIndexForTimestampMs(firstCgm.ms);
            entry.timestamp = firstCgm.timestamp || '';
            entry.timestamp_ms = firstCgm.ms;
            entry.cgm_value = firstCgm.value;
          }
        }
        const index = entry.index;
        state.cursorIndex[panel] = index;
        state.cursorTimeMs[panel] = entry.timestamp_ms;
        const meta = currentMeta();
        if (!meta) return;
        fetchJson('/api/labels/add', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            path: meta.path,
            panel,
            name,
            index,
            timestamp: entry.timestamp,
            timestamp_ms: entry.timestamp_ms,
            cgm_value: entry.cgm_value
          })
        }).then((data) => {
          state.savedSegments = data.saved_segments || state.savedSegments;
          refreshLabelStateForPanel(meta.path, panel);
          renderLabelLogs();
          renderList();
          setStatus('Added label "' + name + '" at index ' + index + ' for ' + panelLabel(panel) + '.');
        }).catch((error) => setStatus(error.message));
      });
      document.getElementById('set-start-' + suffix).addEventListener('click', () => {
        setActivePanel(panel);
        state.cursorIndex[panel] = 0;
        state.markerIndex[panel].start = 0;
        drawCurrentPlot(panel);
        updateRegionStatus();
        const rows = currentRows();
        const ts = rows[state.markerIndex[panel].start]?.Timestamp || 'NA';
        setStatus('Start marker set at ' + ts + ' (idx ' + state.markerIndex[panel].start + ') for ' + panelLabel(panel) + '.');
      });
      document.getElementById('set-end-' + suffix).addEventListener('click', () => {
        setActivePanel(panel);
        state.cursorIndex[panel] = 0;
        state.markerIndex[panel].end = 0;
        drawCurrentPlot(panel);
        updateRegionStatus();
        const rows = currentRows();
        const ts = rows[state.markerIndex[panel].end]?.Timestamp || 'NA';
        setStatus('End marker set at ' + ts + ' (idx ' + state.markerIndex[panel].end + ') for ' + panelLabel(panel) + '.');
      });
      document.getElementById('add-region-' + suffix).addEventListener('click', () => {
        setActivePanel(panel);
        const s = state.markerIndex[panel].start;
        const e = state.markerIndex[panel].end;
        if (s == null || e == null) {
          setStatus('Set both start and end markers first for ' + panelLabel(panel) + '.');
          return;
        }
        const meta = currentMeta();
        if (!meta) return;
        const pair = { start_index: Math.min(s, e), end_index: Math.max(s, e) };
        fetchJson('/api/regions/add', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: meta.path, panel, start_index: pair.start_index, end_index: pair.end_index })
        }).then((data) => {
          state.savedSegments = data.saved_segments || state.savedSegments;
          state.regionPairs[panel] = savedRegionPairs(meta.path, panel).map((entry) => ({ start_index: entry.start_index, end_index: entry.end_index }));
          drawCurrentPlot(panel);
          updateSegmentTable();
          updateRegionStatus();
          renderRegionLogs();
          renderList();
          setStatus('Added region [' + pair.start_index + ', ' + pair.end_index + '] for ' + panelLabel(panel) + ' and saved metadata.');
        }).catch((error) => setStatus(error.message));
      });
      document.getElementById('clear-regions-' + suffix).addEventListener('click', () => {
        setActivePanel(panel);
        const meta = currentMeta();
        if (!meta) return;
        fetchJson('/api/regions/clear', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: meta.path, panel })
        }).then((data) => {
          state.savedSegments = data.saved_segments || state.savedSegments;
          state.regionPairs[panel] = [];
          drawCurrentPlot(panel);
          updateSegmentTable();
          updateRegionStatus();
          renderRegionLogs();
          renderList();
          setStatus('Cleared regions for ' + panelLabel(panel) + ' and updated metadata.');
        }).catch((error) => setStatus(error.message));
      });
    }
    wirePanelControls('glu_cgm');
    wirePanelControls('glu_lac');
    document.getElementById('file-label-filter').addEventListener('change', (event) => {
      state.fileLabelFilter = event.target.value || 'all';
      renderList();
    });
    loadFiles().catch((error) => setStatus(error.message));
  </script>
</body>
</html>"""


class AlignmentHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _send_html(self, text, status=200):
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _send_js(self, text, status=200):
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/javascript; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(page_html())
            return
        if parsed.path == "/analysis":
            if analysis_results_page_html is None:
                self._send_html(
                    "<!doctype html><title>Analysis unavailable</title><p>Analysis dashboard could not be loaded. Check server imports for analysis_results_dashboard.py dependencies.</p>",
                    status=500,
                )
            else:
                self._send_html(analysis_results_page_html())
            return
        if parsed.path == "/analysis_dashboard.js":
            if analysis_dashboard_script is None:
                self._send_js("console.error('Analysis dashboard script could not be loaded.');", status=500)
            else:
                self._send_js(analysis_dashboard_script())
            return
        if parsed.path == "/api/files":
            datasets = [summarize_processed_file_fast(path) for path in list_processed_files()]
            self._send_json({"datasets": datasets, "saved_segments": load_segment_state()})
            return
        if parsed.path == "/api/analysis/current":
            self._send_json(build_current_analysis_payload())
            return
        if parsed.path == "/api/range-labels":
            self._send_json(load_range_label_state())
            return
        if parsed.path == "/api/range-label-history":
            rel_path = parse_qs(parsed.query).get("path", [""])[0]
            self._send_json(load_range_label_history(rel_path or None))
            return
        if parsed.path == "/api/projects" or parsed.path == "/api/projects/active_status":
            self._send_json({"error": "Project version-control endpoints are disabled. Use single-file autosave annotated_data/alignment_annotations.json."}, status=410)
            return
        if parsed.path == "/api/series":
            rel_path = parse_qs(parsed.query).get("path", [""])[0]
            try:
                rows = load_processed_series(rel_path)
                self._send_json(
                    {
                        "path": rel_path,
                        "rows": rows_for_browser(rows),
                        "cgm_series": independent_cgm_series_for_case(rel_path, processed_rows=rows),
                        "diet_markers": load_diet_markers(rel_path),
                        "diet_instances": diet_instances_for_case(rel_path),
                        "cgm_day_ranges_original": cgm_day_ranges_from_original(rel_path, max_days=3),
                    }
                )
            except Exception as error:
                self._send_json({"error": str(error)}, status=400)
            return
        if parsed.path == "/api/peak/features":
            rel_path = parse_qs(parsed.query).get("path", [""])[0]
            try:
                path = safe_rel_path(rel_path)
                seg_state = load_segment_state()
                file_state = seg_state.get(path, {})
                rows = rows_for_browser(load_processed_series(path))
                file_state_for_agg = file_state if isinstance(file_state, dict) else {}
                aggregated = compute_aggregated_peaks_for_case(path, file_state_for_agg, rows)
                aggregated_by_panel = {
                    "glu_cgm": compute_aggregated_peaks_for_case(path, file_state_for_agg, rows, label_keys=("labels_glu_cgm",)),
                    "glu_lac": compute_aggregated_peaks_for_case(path, file_state_for_agg, rows, label_keys=("labels_glu_lac",)),
                }
                peak_state = load_peak_pair_state()
                cases = peak_state.get("cases", {}) if isinstance(peak_state, dict) else {}
                case = cases.get(path, {}) if isinstance(cases, dict) else {}
                if not case and isinstance(cases, dict):
                    for legacy_path, legacy_case in cases.items():
                        try:
                            if remap_case_path_to_current(legacy_path, [path]) == path:
                                case = legacy_case
                                break
                        except Exception:
                            continue
                paired = case.get("paired_features", []) if isinstance(case, dict) else []
                self._send_json(
                    {
                        "path": path,
                        "aggregated_peaks": aggregated,
                        "aggregated_peaks_by_panel": aggregated_by_panel,
                        "paired_features": paired,
                    }
                )
            except Exception as error:
                self._send_json({"error": str(error)}, status=400)
            return
        if parsed.path == "/api/summary":
            lag_minutes = int(parse_qs(parsed.query).get("lag_minutes", ["0"])[0] or "0")
            lag_minutes = max(0, min(180, lag_minutes))
            self._send_json(cohort_summary(max_lag_steps=lag_minutes * 60))
            return
        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0") or "0")
        payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        try:
            if parsed.path == "/api/tsne":
                matrix = payload.get("matrix", [])
                perplexity = float(payload.get("perplexity", 10))
                if not isinstance(matrix, list) or len(matrix) < 3:
                    self._send_json({"ok": False, "error": "Need at least 3 rows for t-SNE."}, status=400)
                    return
                cleaned = []
                dims = None
                for row in matrix:
                    if not isinstance(row, list):
                        continue
                    vals = [float(x) for x in row]
                    if dims is None:
                        dims = len(vals)
                    if not dims or len(vals) != dims:
                        continue
                    cleaned.append(vals)
                if len(cleaned) < 3:
                    self._send_json({"ok": False, "error": "Insufficient valid rows for t-SNE."}, status=400)
                    return
                per = max(2.0, min(30.0, min(perplexity, (len(cleaned) - 1) / 3.0)))
                method = "scikit-learn t-SNE"
                if SklearnTSNE is not None:
                    try:
                        model = SklearnTSNE(
                            n_components=2,
                            perplexity=per,
                            init="pca",
                            learning_rate="auto",
                            random_state=42,
                            max_iter=1000,
                        )
                    except TypeError:
                        model = SklearnTSNE(
                            n_components=2,
                            perplexity=per,
                            init="pca",
                            learning_rate="auto",
                            random_state=42,
                            n_iter=1000,
                        )
                    emb = model.fit_transform(cleaned)
                elif fallback_tsne_embedding is not None:
                    emb = fallback_tsne_embedding(cleaned, perplexity=per, max_iter=500, random_state=42)
                    method = "NumPy t-SNE fallback" if NUMPY_AVAILABLE else "dependency-free PCA fallback"
                else:
                    raise ValueError("t-SNE dependencies are unavailable.")
                out = [[float(v[0]), float(v[1])] for v in emb]
                self._send_json({"ok": True, "embedding": out, "method": method})
                return
            if parsed.path == "/api/segment":
                path = safe_rel_path(payload.get("path", ""))
                panel = payload.get("panel", "glu_cgm")
                if panel not in ("glu_cgm", "glu_lac"):
                    raise ValueError("Invalid panel.")
                rows = load_processed_series(path)
                start = max(0, min(int(payload.get("start_index", 0)), len(rows) - 1))
                end = max(start, min(int(payload.get("end_index", len(rows) - 1)), len(rows) - 1))
                state = load_segment_state()
                file_state = state.get(path, {})
                if not isinstance(file_state, dict):
                    file_state = {}
                file_state.setdefault("included_glu_cgm", True)
                file_state.setdefault("included_glu_lac", True)
                file_state[panel] = {"start_index": start, "end_index": end}
                state[path] = file_state
                save_segment_state(state)
                self._send_json({"ok": True, "saved_segments": state})
                return
            if parsed.path.startswith("/api/projects"):
                raise ValueError("Project version-control endpoints are disabled. Use single-file autosave annotated_data/alignment_annotations.json.")
            if parsed.path == "/api/converted/export":
                path = safe_rel_path(payload.get("path", ""))
                source = payload.get("source", "seg")
                smoothing_ratio = float(payload.get("smoothing_ratio", 0.08))
                out_rel, row_count, label, ratio = export_converted_time_series(path, source=source, smoothing_ratio=smoothing_ratio)
                self._send_json(
                    {
                        "ok": True,
                        "output_path": out_rel,
                        "row_count": row_count,
                        "source": label,
                        "smoothing_ratio": ratio,
                    }
                )
                return
            if parsed.path == "/api/range-labels/save":
                selection, state = save_range_label_selection(
                    payload.get("path", ""),
                    payload.get("start_index", 0),
                    payload.get("end_index", 0),
                    label_source=payload.get("label_source", "range_plot"),
                    note=payload.get("note", ""),
                    smoothing_ratio=payload.get("smoothing_ratio", 0.08),
                )
                self._send_json(
                    {
                        "ok": True,
                        "selection": selection,
                        "range_count": len(state.get("ranges", [])),
                        "output_path": os.path.relpath(RANGE_LABELS_JSON, ROOT_DIR).replace(os.sep, "/"),
                    }
                )
                return
            if parsed.path == "/api/range-labels/delete":
                removed_history, removed_ranges = delete_range_label_entry(
                    target_id=str(payload.get("id", "") or ""),
                    path=payload.get("path", ""),
                    start_timestamp=str(payload.get("start_timestamp", "") or ""),
                    end_timestamp=str(payload.get("end_timestamp", "") or ""),
                )
                self._send_json(
                    {
                        "ok": True,
                        "removed_history": removed_history,
                        "removed_ranges": removed_ranges,
                        "history": load_range_label_history(payload.get("path", "")).get("history", []),
                    }
                )
                return
            if parsed.path == "/api/include":
                path = safe_rel_path(payload.get("path", ""))
                panel = payload.get("panel", "glu_cgm")
                if panel not in ("glu_cgm", "glu_lac"):
                    raise ValueError("Invalid panel.")
                included = bool(payload.get("included", True))
                state = load_segment_state()
                file_state = state.get(path, {})
                if not isinstance(file_state, dict):
                    file_state = {}
                key = "included_glu_cgm" if panel == "glu_cgm" else "included_glu_lac"
                file_state[key] = included
                file_state.setdefault("region_pairs_glu_cgm", [])
                file_state.setdefault("region_pairs_glu_lac", [])
                state[path] = file_state
                save_segment_state(state)
                # Re-load from disk so frontend always receives persisted truth.
                self._send_json({"ok": True, "saved_segments": load_segment_state()})
                return
            if parsed.path == "/api/seg-label-sync":
                path = safe_rel_path(payload.get("path", ""))
                sync = bool(payload.get("sync", True))
                state = load_segment_state()
                file_state = state.get(path, {})
                if not isinstance(file_state, dict):
                    file_state = {}
                file_state.setdefault("included_glu_cgm", True)
                file_state.setdefault("included_glu_lac", True)
                file_state.setdefault("region_pairs_glu_cgm", [])
                file_state.setdefault("region_pairs_glu_lac", [])
                file_state["sync_seg_labels"] = sync
                state[path] = file_state
                save_segment_state(state)
                self._send_json({"ok": True, "saved_segments": load_segment_state()})
                return
            if parsed.path == "/api/regions/add":
                path = safe_rel_path(payload.get("path", ""))
                panel = payload.get("panel", "glu_cgm")
                if panel not in ("glu_cgm", "glu_lac"):
                    raise ValueError("Invalid panel.")
                rows = load_processed_series(path)
                if not rows:
                    raise ValueError("No rows available.")
                start = max(0, min(int(payload.get("start_index", 0)), len(rows) - 1))
                end = max(0, min(int(payload.get("end_index", 0)), len(rows) - 1))
                pair = {"start_index": min(start, end), "end_index": max(start, end)}
                state = load_segment_state()
                file_state = state.get(path, {})
                if not isinstance(file_state, dict):
                    file_state = {}
                file_state.setdefault("included_glu_cgm", True)
                file_state.setdefault("included_glu_lac", True)
                key = "region_pairs_glu_cgm" if panel == "glu_cgm" else "region_pairs_glu_lac"
                pairs = file_state.get(key, [])
                if not isinstance(pairs, list):
                    pairs = []
                pairs.append(pair)
                file_state[key] = pairs
                file_state[panel] = {"start_index": pair["start_index"], "end_index": pair["end_index"]}
                state[path] = file_state
                save_segment_state(state)
                self._send_json({"ok": True, "saved_segments": state, "region_pairs": pairs})
                return
            if parsed.path == "/api/regions/remove":
                path = safe_rel_path(payload.get("path", ""))
                panel = payload.get("panel", "glu_cgm")
                if panel not in ("glu_cgm", "glu_lac"):
                    raise ValueError("Invalid panel.")
                region_index = int(payload.get("region_index", -1))
                state = load_segment_state()
                file_state = state.get(path, {})
                if not isinstance(file_state, dict):
                    file_state = {}
                key = "region_pairs_glu_cgm" if panel == "glu_cgm" else "region_pairs_glu_lac"
                pairs = file_state.get(key, [])
                if not isinstance(pairs, list):
                    pairs = []
                if region_index < 0 or region_index >= len(pairs):
                    raise ValueError("Invalid region index.")
                pairs.pop(region_index)
                file_state[key] = pairs
                if pairs:
                    last = pairs[-1]
                    file_state[panel] = {"start_index": int(last["start_index"]), "end_index": int(last["end_index"])}
                elif panel in file_state:
                    del file_state[panel]
                file_state.setdefault("included_glu_cgm", True)
                file_state.setdefault("included_glu_lac", True)
                state[path] = file_state
                save_segment_state(state)
                self._send_json({"ok": True, "saved_segments": state, "region_pairs": pairs})
                return
            if parsed.path == "/api/regions/clear":
                path = safe_rel_path(payload.get("path", ""))
                panel = payload.get("panel", "glu_cgm")
                if panel not in ("glu_cgm", "glu_lac"):
                    raise ValueError("Invalid panel.")
                state = load_segment_state()
                file_state = state.get(path, {})
                if not isinstance(file_state, dict):
                    file_state = {}
                key = "region_pairs_glu_cgm" if panel == "glu_cgm" else "region_pairs_glu_lac"
                file_state[key] = []
                if panel in file_state:
                    del file_state[panel]
                file_state.setdefault("included_glu_cgm", True)
                file_state.setdefault("included_glu_lac", True)
                state[path] = file_state
                save_segment_state(state)
                self._send_json({"ok": True, "saved_segments": state, "region_pairs": []})
                return
            if parsed.path == "/api/peak/pairs/save":
                path = safe_rel_path(payload.get("path", ""))
                panel = str(payload.get("panel", "") or "").strip()
                seg_id = str(payload.get("seg_id", "")).strip()
                sel_id = str(payload.get("sel_id", "")).strip()
                cgm_id = str(payload.get("cgm_id", "")).strip()
                if panel == "glu_cgm":
                    category = "CGM_vs_SEG"
                    sel_id = ""
                    if not (seg_id and cgm_id):
                        raise ValueError("Glu-CGM pairing requires exactly one SEG aggregate and one CGM aggregate.")
                elif panel == "glu_lac":
                    category = "CGM_vs_SEG"
                    pair_index = int(payload.get("pair_index", -1))
                    if not sel_id:
                        raise ValueError("Glu-Lac triplet update requires one SEL aggregate.")
                    cgm_id = ""
                else:
                    panel = ""
                present = {"seg": bool(seg_id), "sel": bool(sel_id), "cgm": bool(cgm_id)}
                if panel != "glu_lac" and sum(1 for v in present.values() if v) != 2:
                    raise ValueError("Exactly two aggregates must be selected.")
                if not panel:
                    if present["cgm"] and present["seg"]:
                        category = "CGM_vs_SEG"
                        panel = "glu_cgm"
                    elif present["cgm"] and present["sel"]:
                        category = "CGM_vs_SEL"
                    else:
                        raise ValueError("Unsupported pair.")
                seg_state = load_segment_state()
                file_state = seg_state.get(path, {})
                rows = rows_for_browser(load_processed_series(path))
                independent_cgm_points = independent_cgm_series_for_case(path, rows)
                file_state_for_agg = file_state if isinstance(file_state, dict) else {}
                if panel == "glu_lac":
                    peak_state = load_peak_pair_state()
                    cases = peak_state.setdefault("cases", {})
                    case = cases.get(path, {})
                    if not isinstance(case, dict):
                        case = {}
                    pairs = case.get("paired_features", [])
                    if not isinstance(pairs, list):
                        pairs = []
                    if pair_index < 0 or pair_index >= len(pairs):
                        raise ValueError("Choose a valid fixed CGM-SEG peak instance.")
                    existing_pair = pairs[pair_index]
                    if not isinstance(existing_pair, dict) or str(existing_pair.get("category", "")) != "CGM_vs_SEG":
                        raise ValueError("Selected instance is not a CGM-SEG peak instance.")
                    aggregates = compute_aggregated_peaks_for_case(path, file_state_for_agg, rows, label_keys=("labels_glu_lac",))
                    agg_lookup = {x.get("id"): x for x in aggregates if isinstance(x, dict)}
                    selected_existing = existing_pair.setdefault("selected_peak_metadata", {})
                    cgm_meta_existing = selected_existing.get("cgm") if isinstance(selected_existing, dict) else {}
                    shared_anchor_peak_ts = cgm_meta_existing.get("peak_timestamp", "") if isinstance(cgm_meta_existing, dict) else ""
                    shared_preceding_instance = find_preceding_diet_instance_for_peak(path, shared_anchor_peak_ts) if shared_anchor_peak_ts else None

                    def packed_sel_meta(agg_id):
                        agg = agg_lookup.get(agg_id)
                        if not isinstance(agg, dict) or agg.get("signal") != "sel":
                            return None
                        left_ts = agg.get("left_trough", {}).get("timestamp", "")
                        peak_ts = agg.get("peak", {}).get("timestamp", "")
                        right_ts = agg.get("right_trough", {}).get("timestamp", "")
                        duration_seconds = None
                        try:
                            ldt = parse_timestamp_flexible(left_ts)
                            rdt = parse_timestamp_flexible(right_ts)
                            if ldt and rdt:
                                duration_seconds = max(0, int((rdt - ldt).total_seconds()))
                        except Exception:
                            duration_seconds = None
                        preceding_instance = decorated_preceding_instance_for_peak(
                            shared_preceding_instance,
                            peak_ts,
                            anchor_signal="cgm",
                            anchor_peak_timestamp=shared_anchor_peak_ts,
                        )
                        meta = {
                            "aggregate_id": agg_id,
                            "signal": "sel",
                            "pair_context": "CGM_vs_SEG",
                            "label_panel": "glu_lac",
                            "trough1_timestamp": left_ts,
                            "peak_timestamp": peak_ts,
                            "trough2_timestamp": right_ts,
                            "duration_trough1_to_trough2_points": None,
                            "duration_trough1_to_trough2_seconds": duration_seconds,
                            "preceding_instance": preceding_instance,
                        }
                        return augment_peak_metadata_with_derivatives(path, rows, meta, cgm_points=None, aggregates=aggregates)

                    sel_meta = packed_sel_meta(sel_id)
                    if sel_meta is None:
                        raise ValueError("Selected SEL aggregate could not be found in Glu-Lac annotations.")
                    selected_existing["sel"] = sel_meta
                    existing_pair["selected_peak_metadata"] = selected_existing
                    existing_pair["sel_id"] = sel_id
                    existing_pair["triplet_complete"] = True
                    existing_pair["sel_attached_at_utc"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    existing_pair.setdefault("panel", "glu_cgm")
                    existing_pair.setdefault("category", "CGM_vs_SEG")
                    augment_pair_peak_time_differences(existing_pair)
                    pairs[pair_index] = existing_pair
                    case["paired_features"] = pairs
                    cases[path] = case
                    save_peak_pair_state(peak_state)
                    self._send_json({"ok": True, "paired_features": pairs})
                    return
                label_keys = aggregation_label_keys_for_pair_category(file_state_for_agg, category)
                aggregates = compute_aggregated_peaks_for_case(path, file_state_for_agg, rows, label_keys=label_keys)
                agg_lookup = {x.get("id"): x for x in aggregates if isinstance(x, dict)}

                def _agg_peak_ts(agg_id):
                    agg = agg_lookup.get(agg_id)
                    if not isinstance(agg, dict):
                        return ""
                    return agg.get("peak", {}).get("timestamp", "")

                shared_preceding_instance = None
                shared_anchor_signal = anchor_signal_for_pair_category(category)
                shared_anchor_peak_ts = ""
                if category == "CGM_vs_SEG" and cgm_id:
                    shared_anchor_peak_ts = _agg_peak_ts(cgm_id)
                    shared_preceding_instance = find_preceding_diet_instance_for_peak(path, shared_anchor_peak_ts)
                elif category == "CGM_vs_SEL" and cgm_id:
                    shared_anchor_peak_ts = _agg_peak_ts(cgm_id)
                    shared_preceding_instance = find_preceding_diet_instance_for_peak(path, shared_anchor_peak_ts)
                elif category == "SEG_vs_SEL" and sel_id:
                    shared_anchor_peak_ts = _agg_peak_ts(sel_id)
                    shared_preceding_instance = find_preceding_diet_instance_for_peak(path, shared_anchor_peak_ts)

                def packed_meta(agg_id, force_preceding_instance=None):
                    if not agg_id:
                        return None
                    agg = agg_lookup.get(agg_id)
                    if not isinstance(agg, dict):
                        return None
                    left_ts = agg.get("left_trough", {}).get("timestamp", "")
                    peak_ts = agg.get("peak", {}).get("timestamp", "")
                    right_ts = agg.get("right_trough", {}).get("timestamp", "")
                    duration_points = int(agg.get("window", {}).get("duration_points", 0) or 0)
                    duration_seconds = None
                    try:
                        ldt = parse_timestamp_flexible(left_ts)
                        rdt = parse_timestamp_flexible(right_ts)
                        if ldt and rdt:
                            duration_seconds = max(0, int((rdt - ldt).total_seconds()))
                    except Exception:
                        duration_seconds = None
                    base_preceding_instance = force_preceding_instance if force_preceding_instance is not None else find_preceding_diet_instance_for_peak(path, peak_ts)
                    preceding_instance = decorated_preceding_instance_for_peak(
                        base_preceding_instance,
                        peak_ts,
                        anchor_signal=shared_anchor_signal,
                        anchor_peak_timestamp=shared_anchor_peak_ts,
                    )
                    meta = {
                        "aggregate_id": agg_id,
                        "signal": agg.get("signal"),
                        "pair_context": category,
                        "label_panel": label_panel_for_pair_signal(category, agg.get("signal")),
                        "trough1_timestamp": left_ts,
                        "peak_timestamp": peak_ts,
                        "trough2_timestamp": right_ts,
                        "duration_trough1_to_trough2_points": None,
                        "duration_trough1_to_trough2_seconds": duration_seconds,
                        "preceding_instance": preceding_instance,
                    }
                    cgm_points = independent_cgm_points if str(meta.get("signal", "")).lower() == "cgm" else None
                    return augment_peak_metadata_with_derivatives(path, rows, meta, cgm_points=cgm_points, aggregates=aggregates)

                peak_state = load_peak_pair_state()
                cases = peak_state.setdefault("cases", {})
                case = cases.get(path, {})
                if not isinstance(case, dict):
                    case = {}
                pairs = case.get("paired_features", [])
                if not isinstance(pairs, list):
                    pairs = []
                idx = sum(1 for x in pairs if isinstance(x, dict) and str(x.get("category", "")) == category) + 1
                selected_peak_metadata = {}
                seg_meta = packed_meta(seg_id, shared_preceding_instance if category in ("CGM_vs_SEG", "SEG_vs_SEL") else None)
                sel_meta = packed_meta(sel_id, shared_preceding_instance if category == "SEG_vs_SEL" else None)
                cgm_meta = packed_meta(cgm_id, shared_preceding_instance if category == "CGM_vs_SEG" else None)
                if seg_meta is not None:
                    selected_peak_metadata["seg"] = seg_meta
                if sel_meta is not None:
                    selected_peak_metadata["sel"] = sel_meta
                if cgm_meta is not None:
                    selected_peak_metadata["cgm"] = cgm_meta
                instance_id = f"{path}::{category}_{idx}"
                new_pair = {
                    "name": f"{category}_{idx}",
                    "instance_id": instance_id,
                    "case_path": path,
                    "panel": panel or label_panel_for_pair_signal(category, "seg"),
                    "category": category,
                    "seg_id": seg_id or None,
                    "sel_id": sel_id or None,
                    "cgm_id": cgm_id or None,
                    "selected_peak_metadata": selected_peak_metadata,
                    "saved_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                }
                augment_pair_peak_time_differences(new_pair)
                pairs.append(new_pair)
                case["paired_features"] = pairs
                cases[path] = case
                save_peak_pair_state(peak_state)
                self._send_json({"ok": True, "paired_features": pairs})
                return
            if parsed.path == "/api/peak/pairs/clear-sel":
                path = safe_rel_path(payload.get("path", ""))
                pair_index = int(payload.get("pair_index", -1))
                peak_state = load_peak_pair_state()
                cases = peak_state.setdefault("cases", {})
                case = cases.get(path, {})
                if not isinstance(case, dict):
                    case = {}
                pairs = case.get("paired_features", [])
                if not isinstance(pairs, list):
                    pairs = []
                if pair_index < 0 or pair_index >= len(pairs):
                    raise ValueError("Invalid pair_index")
                pair = pairs[pair_index]
                if not isinstance(pair, dict) or str(pair.get("category", "")) != "CGM_vs_SEG":
                    raise ValueError("Selected item is not a CGM-SEG triplet anchor.")
                clear_sel_from_triplet_pair(pair)
                augment_pair_peak_time_differences(pair)
                pairs[pair_index] = pair
                case["paired_features"] = pairs
                cases[path] = case
                save_peak_pair_state(peak_state)
                self._send_json({"ok": True, "paired_features": pairs})
                return
            if parsed.path == "/api/peak/pairs/remove":
                path = safe_rel_path(payload.get("path", ""))
                pair_index = int(payload.get("pair_index", -1))
                peak_state = load_peak_pair_state()
                cases = peak_state.setdefault("cases", {})
                case = cases.get(path, {})
                if not isinstance(case, dict):
                    case = {}
                pairs = case.get("paired_features", [])
                if not isinstance(pairs, list):
                    pairs = []
                if pair_index < 0 or pair_index >= len(pairs):
                    raise ValueError("Invalid pair_index")
                pairs.pop(pair_index)
                case["paired_features"] = pairs
                cases[path] = case
                save_peak_pair_state(peak_state)
                self._send_json({"ok": True, "paired_features": pairs})
                return
            if parsed.path == "/api/labels/add":
                path = safe_rel_path(payload.get("path", ""))
                panel = payload.get("panel", "glu_cgm")
                if panel not in ("glu_cgm", "glu_lac"):
                    raise ValueError("Invalid panel.")
                rows = load_processed_series(path)
                if not rows:
                    raise ValueError("No rows available.")
                index = max(0, min(int(payload.get("index", 0)), len(rows) - 1))
                name = canonical_label_name(payload.get("name", ""))
                state = load_segment_state()
                file_state = state.get(path, {})
                if not isinstance(file_state, dict):
                    file_state = {}
                file_state.setdefault("included_glu_cgm", True)
                file_state.setdefault("included_glu_lac", True)
                file_state.setdefault("region_pairs_glu_cgm", [])
                file_state.setdefault("region_pairs_glu_lac", [])
                key = panel_label_key(panel)
                labels = file_state.get(key, [])
                if not isinstance(labels, list):
                    labels = []
                counter_base = visible_labels_for_panel(file_state, panel)
                counter = next_label_counter(counter_base, name)
                entry = {"name": name, "counter": counter, "index": index, "origin_panel": panel}
                if _signal_kind_from_label_name(name)[0] == "cgm":
                    timestamp = str(payload.get("timestamp", "") or "")
                    timestamp_ms = _to_finite_float(payload.get("timestamp_ms"))
                    cgm_value = _to_finite_float(payload.get("cgm_value"))
                    if timestamp:
                        entry["timestamp"] = timestamp
                    if timestamp_ms is not None:
                        entry["timestamp_ms"] = timestamp_ms
                    if cgm_value is not None:
                        entry["cgm_value"] = cgm_value
                labels.append(entry)
                file_state[key] = labels
                if is_seg_label(name) and seg_label_sync_enabled(file_state):
                    other_key = panel_label_key(other_panel(panel))
                    other_labels = file_state.get(other_key, [])
                    if not isinstance(other_labels, list):
                        other_labels = []
                    other_labels.append(dict(entry))
                    file_state[other_key] = other_labels
                state[path] = file_state
                save_segment_state(state)
                self._send_json({"ok": True, "saved_segments": state, "labels": labels})
                return
            if parsed.path == "/api/labels/remove":
                path = safe_rel_path(payload.get("path", ""))
                panel = payload.get("panel", "glu_cgm")
                if panel not in ("glu_cgm", "glu_lac"):
                    raise ValueError("Invalid panel.")
                label_index = int(payload.get("label_index", -1))
                state = load_segment_state()
                file_state = state.get(path, {})
                if not isinstance(file_state, dict):
                    file_state = {}
                key = panel_label_key(panel)
                labels = file_state.get(key, [])
                if not isinstance(labels, list):
                    labels = []
                if label_index < 0 or label_index >= len(labels):
                    raise ValueError("Invalid label index.")
                removed = labels.pop(label_index)
                file_state[key] = labels
                if isinstance(removed, dict) and is_seg_label(removed.get("name", "")) and seg_label_sync_enabled(file_state):
                    other_key = panel_label_key(other_panel(panel))
                    other_labels = file_state.get(other_key, [])
                    if not isinstance(other_labels, list):
                        other_labels = []
                    target_name = str(removed.get("name", ""))
                    target_counter = int(removed.get("counter", 0) or 0)
                    target_index = int(removed.get("index", -1) or -1)
                    for i, entry in enumerate(other_labels):
                        if not isinstance(entry, dict):
                            continue
                        if str(entry.get("name", "")) == target_name and int(entry.get("counter", 0) or 0) == target_counter and int(entry.get("index", -1) or -1) == target_index:
                            other_labels.pop(i)
                            break
                    file_state[other_key] = other_labels
                file_state.setdefault("included_glu_cgm", True)
                file_state.setdefault("included_glu_lac", True)
                state[path] = file_state
                save_segment_state(state)
                self._send_json({"ok": True, "saved_segments": state, "labels": labels})
                return
            if parsed.path == "/api/labels/rename":
                path = safe_rel_path(payload.get("path", ""))
                panel = payload.get("panel", "glu_cgm")
                if panel not in ("glu_cgm", "glu_lac"):
                    raise ValueError("Invalid panel.")
                label_index = int(payload.get("label_index", -1))
                name = canonical_label_name(payload.get("name", ""))
                state = load_segment_state()
                file_state = state.get(path, {})
                if not isinstance(file_state, dict):
                    file_state = {}
                key = panel_label_key(panel)
                labels = file_state.get(key, [])
                if not isinstance(labels, list):
                    labels = []
                if label_index < 0 or label_index >= len(labels):
                    raise ValueError("Invalid label index.")
                existing = labels[label_index] if isinstance(labels[label_index], dict) else {}
                old_name = str(existing.get("name", "")).strip()
                old_counter = int(existing.get("counter", 0) or 0)
                old_index = int(existing.get("index", -1) or -1)
                labels[label_index]["name"] = name
                if old_name != name or int(existing.get("counter", 0) or 0) < 1:
                    other_labels = [entry for i, entry in enumerate(labels) if i != label_index]
                    labels[label_index]["counter"] = next_label_counter(other_labels, name)
                file_state[key] = labels
                new_entry = labels[label_index] if isinstance(labels[label_index], dict) else {}
                if (is_seg_label(old_name) or is_seg_label(name)) and seg_label_sync_enabled(file_state):
                    other_key = panel_label_key(other_panel(panel))
                    other_labels = file_state.get(other_key, [])
                    if not isinstance(other_labels, list):
                        other_labels = []
                    match_idx = -1
                    for i, entry in enumerate(other_labels):
                        if not isinstance(entry, dict):
                            continue
                        if str(entry.get("name", "")) == old_name and int(entry.get("counter", 0) or 0) == old_counter and int(entry.get("index", -1) or -1) == old_index:
                            match_idx = i
                            break
                    if is_seg_label(name):
                        mirror = {
                            "name": new_entry.get("name", name),
                            "counter": int(new_entry.get("counter", 1) or 1),
                            "index": int(new_entry.get("index", old_index) or 0),
                        }
                        if match_idx >= 0:
                            other_labels[match_idx] = mirror
                        else:
                            other_labels.append(mirror)
                    elif match_idx >= 0:
                        other_labels.pop(match_idx)
                    file_state[other_key] = other_labels
                state[path] = file_state
                save_segment_state(state)
                self._send_json({"ok": True, "saved_segments": state, "labels": labels})
                return
            if parsed.path == "/api/labels/move":
                path = safe_rel_path(payload.get("path", ""))
                panel = payload.get("panel", "glu_cgm")
                if panel not in ("glu_cgm", "glu_lac"):
                    raise ValueError("Invalid panel.")
                label_index = int(payload.get("label_index", -1))
                rows = load_processed_series(path)
                if not rows:
                    raise ValueError("No rows available.")
                new_index = max(0, min(int(payload.get("index", 0)), len(rows) - 1))
                state = load_segment_state()
                file_state = state.get(path, {})
                if not isinstance(file_state, dict):
                    file_state = {}
                key = panel_label_key(panel)
                labels = file_state.get(key, [])
                if not isinstance(labels, list):
                    labels = []
                if label_index < 0 or label_index >= len(labels):
                    raise ValueError("Invalid label index.")
                labels[label_index]["index"] = new_index
                file_state[key] = labels
                moving = labels[label_index] if isinstance(labels[label_index], dict) else {}
                if _signal_kind_from_label_name(moving.get("name", ""))[0] == "cgm":
                    timestamp = str(payload.get("timestamp", "") or "")
                    timestamp_ms = _to_finite_float(payload.get("timestamp_ms"))
                    cgm_value = _to_finite_float(payload.get("cgm_value"))
                    if timestamp:
                        moving["timestamp"] = timestamp
                    if timestamp_ms is not None:
                        moving["timestamp_ms"] = timestamp_ms
                    if cgm_value is not None:
                        moving["cgm_value"] = cgm_value
                if is_seg_label(moving.get("name", "")) and seg_label_sync_enabled(file_state):
                    other_key = panel_label_key(other_panel(panel))
                    other_labels = file_state.get(other_key, [])
                    if not isinstance(other_labels, list):
                        other_labels = []
                    target_name = str(moving.get("name", ""))
                    target_counter = int(moving.get("counter", 0) or 0)
                    for entry in other_labels:
                        if not isinstance(entry, dict):
                            continue
                        if str(entry.get("name", "")) == target_name and int(entry.get("counter", 0) or 0) == target_counter:
                            entry["index"] = new_index
                            break
                    file_state[other_key] = other_labels
                state[path] = file_state
                save_segment_state(state)
                self._send_json({"ok": True, "saved_segments": state, "labels": labels})
                return
            self._send_json({"error": "Not found"}, status=404)
        except Exception as error:
            self._send_json({"error": str(error)}, status=400)

    def log_message(self, fmt, *args):
        return


def main():
    ensure_dir(STATE_DIR)
    ensure_dir(ALIGNMENT_DIR)
    # Keep the startup lock through metadata initialization to serialize repeated launches.
    with replacing_server(ThreadingHTTPServer, (HOST, PORT), AlignmentHandler,
                          __file__, os.path.join(ALIGNMENT_DIR, 'alignment_server.lock')) as server:
        server.daemon_threads = False
        ensure_unified_annotation_state()
        updated_metadata_features = backfill_peak_pair_metadata_features()
        signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    print("Alignment Dashboard")
    print(f"Open: http://{HOST}:{PORT}")
    print(f"Data root: {ROOT_DIR}")
    print(f"Datasets found: {len(list_processed_files())}")
    print(f"State: {ANNOTATION_STATE_JSON}")
    print(f"Source: {LOCAL_PROCESSED_DIR}")
    if updated_metadata_features:
        print(f"Backfilled paired peak metadata features: {updated_metadata_features}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
