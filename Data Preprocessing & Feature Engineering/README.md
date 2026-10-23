# SEB/CGM data processing and feature engineering

A local dashboard for inspecting sensor and CGM time series, labeling peaks and troughs, matching responses, and exploring engineered features. The code uses `SEG` for the glucose sensor and `SEL` for the lactate sensor.

## Processing pipeline

![Wearable data preprocessing, peak annotation, and feature engineering pipeline](assets/Data%20Processing%20Pipeline.png)

The figure shows the full study workflow. This package includes alignment, preprocessing functions, peak feature engineering, and the **Analysis Results** dashboard. Separate manual quality-screening and meal/medicine-labeling dashboards are not included. Dietary and patient metadata require separately prepared inputs. The figure's cohort counts refer to the study, not the synthetic example. See `dashboard/cgm_feature_engineering.py` for the implemented features.

## Run the dashboard

Use Python 3 on macOS or Linux (or WSL on Windows; the server uses Unix `fcntl` locking). From the package folder containing this README:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python dashboard/server.py
```

Open the local host link displayed in the terminal. Keep the terminal open; press **Ctrl+C** to stop. For subsequent runs, activate `.venv` and run the server again.

The default launch selects the bundled `synthetic_demo/` unless the package itself has an `annotated_data/series/` directory. To explicitly choose the demo or a different dataset:

```sh
SEB_DATA_ROOT="$PWD/synthetic_demo" python dashboard/server.py
# Or:
SEB_DATA_ROOT="/absolute/path/to/my_dataset" python dashboard/server.py
```

`SEB_DATA_ROOT` is the folder containing all three data folders below. Explicit paths take precedence. Default discovery is relative to the package; it works even if you launch the script by absolute path from another directory. Explicit relative environment paths are resolved from the terminal's working directory.

Check the printed **Data root**, **Datasets found**, source folder, and annotation file. The included demo should show **3 datasets**.

## Three-folder data layout

Each dataset uses exactly three top-level folders:

```text
synthetic_demo/                 # Or your own dataset root
├── raw_data/
│   ├── sensor/
│   │   └── P901/Day01/P901_Day01_20300101_SEB.csv
│   └── cgm/
│       └── P901_cleaned.csv
├── annotated_data/
│   ├── series/
│   │   └── P901/Day01/
│   │       ├── P901_Day01_20300101_SEB.csv
│   │       └── P901_Day01_20300101_SEB.csv.metadata.json
│   ├── annotations_input.json       # Demo labels before feature engineering
│   └── alignment_annotations.json   # Active dashboard labels and peak features
└── analysis/
    ├── manifest.json
    ├── feature_engineering_summary.json
    ├── converted_time_series/       # Created when conversion exports are requested
    └── smoothed_series/             # Created when smoothing output is written
```

- **raw_data:** source sensor and CGM measurements.
- **annotated_data:** dashboard-ready processed time series, their source metadata, and saved annotations. Active annotations can include derived peak features so the dashboard can consume one JSON file.
- **analysis:** summaries, exported results, and runtime files such as the server lock.

The dashboard discovers only CSVs under `annotated_data/series/`. Preserve the participant/day/filename suffix between `raw_data/sensor/` and `annotated_data/series/`. JSON references are relative to the dataset root, use forward slashes, and point to the current file locations. No `Organized Data`, `cleaned_cgm_files`, or `analysis/SEB_CGM_alignment` nesting is needed.

Scripts remain in `dashboard/`, the figure in `assets/`, and dependencies in the package's `requirements.txt`. The package-level `analysis/` files from the earlier reorganization are not the bundled demo's active state; default demo operation reads `synthetic_demo/annotated_data/alignment_annotations.json`.

## Use the dashboard

1. Select a recording from **File List** and inspect the raw channels and sensor/CGM plots.
2. Mark peaks and their preceding/following troughs using the labeling controls. The demo already has labels and matched pairs.
3. Click **Refresh Aggregates**, select corresponding SEG and CGM peaks, and click **Save Glu-CGM Pair**. Use **Attach SEL to Triplet** when associating a lactate response.
4. Click **Open Analysis Results**, load current annotations, choose feature columns, and use the analysis/export controls.

Changes save to `annotated_data/alignment_annotations.json` under the selected dataset root. Copy the dataset before experimenting if you want to preserve its initial annotations. Both dashboards use the same active file. Without an annotation file, startup creates an empty one; raw CSVs alone do not supply paired peak features.

## Add your own data

### 1. Prepare raw CSVs

Place sensor recordings in `my_dataset/raw_data/sensor/P001/Day01/`, for example `P001_Day01_SEB.csv`. Required headers and sample rows:

```csv
Timestamp,Channel1,Channel2,Channel3
2030-01-01 08:00:00,0.110,0.150,0.144
2030-01-01 08:01:00,0.111,0.152,0.145
```

Use numeric channel values and chronologically ordered `YYYY-MM-DD HH:MM:SS` timestamps; fractional seconds are supported. Use the same time basis throughout; the loader does not convert timezones. Channel values must match the units expected by your sensor calibration.

Put independent CGM readings in `my_dataset/raw_data/cgm/P001_cleaned.csv`:

```csv
timestamp,glucose_mmol_L
2030-01-01 08:00:00,5.2
2030-01-01 08:05:00,5.4
```

CGM must be in **mmol/L**; convert mg/dL first. Use consistent participant IDs such as `P001` or `H001`. The CGM loader normalizes IDs to at least three digits (`P1` resolves to `P001_cleaned.csv`). Different sampling intervals are supported. Preprocessing matches exact timestamps, or minutes for minute-precision sensor inputs, and leaves unmatched CGM rows empty; feature engineering also loads independent CGM readings.

### 2. Preprocess recordings

The dashboard does not preprocess all raw files automatically at startup. Inspect signal quality, select a baseline and segment, and generate processed CSVs. Run this example from the package folder after editing the dataset location, source path, and indices:

```sh
SEB_DATA_ROOT="/absolute/path/to/my_dataset" python - <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path('dashboard').resolve()))
import server

source = 'raw_data/sensor/P001/Day01/P001_Day01_SEB.csv'
rows = server.load_tool2_series(source)
if not rows:
    raise ValueError('No sensor rows loaded')
# Choose the baseline/asymptote as the start; end is inclusive.
start_index, end_index = 0, len(rows) - 1
processed = server.tool2_process_rows(rows, source, start_index, end_index)
output = 'annotated_data/series/' + source.removeprefix('raw_data/sensor/')
server.write_smoothed_rows_csv(output, processed)
# This writer writes the supplied rows; it does not add smoothing itself.
metadata = Path(server.ROOT_DIR) / (output + '.metadata.json')
metadata.write_text(json.dumps({
    'tool': 'tool2', 'source_path': source,
    'start_index': start_index, 'end_index': end_index
}, indent=2) + '\n')
print('Created:', Path(server.ROOT_DIR) / output)
PY
```

Repeat per recording. The helper subtracts Channel1 and Channel3 from Channel2, adjusts deltas relative to the starting sample, adds the existing `0.04` glucose-delta offset, attaches CGM, and generates min–max normalized columns. Review the baseline choice, offset, and hard-coded concentration coefficients before using another sensor or calibration. Smoothing is available through the dashboard.

Processed CSVs retain the raw channel headers and additionally export `Delta_Channel2_minus_Channel1`, `Delta_Channel2_minus_Channel3`, `CGM`, `Normalized_Delta_Channel2_minus_Channel1`, `Normalized_Delta_Channel2_minus_Channel3`, and `Normalized_CGM`. The demo provides complete examples. An optional CGM column in a raw sensor CSV does not replace the independent CGM file when using this helper.

### 3. Launch and annotate

```sh
SEB_DATA_ROOT="/absolute/path/to/my_dataset" python dashboard/server.py
```

Confirm the dataset count and label and pair peaks in the dashboard. Feature engineering uses existing labels; it does not infer missing peak/trough annotations automatically.

To reuse annotations, put compatible JSON at `annotated_data/alignment_annotations.json` before startup. Case keys and embedded `case_path` values must match processed CSV paths, for example:

```json
{
  "meta": {},
  "saved_segments": {
    "annotated_data/series/P001/Day01/P001_Day01_SEB.csv": {
      "labels_glu_cgm": [], "labels_glu_lac": []
    }
  },
  "peak_pair_features": {
    "cases": {
      "annotated_data/series/P001/Day01/P001_Day01_SEB.csv": {
        "paired_features": []
      }
    }
  }
}
```

This is an empty structural example. See `synthetic_demo/annotated_data/annotations_input.json` for full labels, aggregates, and `selected_peak_metadata`. Preserve recording row order and timestamps when reusing annotations because labels may reference row indices.

### 4. Export recalculated features

```sh
SEB_DATA_ROOT="/absolute/path/to/my_dataset" python dashboard/cgm_feature_engineering.py \
  /absolute/path/to/my_dataset/annotated_data/alignment_annotations.json \
  /absolute/path/to/my_dataset/analysis/annotations_feature_engineered.json
```

This creates a separate enriched JSON and prints a summary. Check `case_errors` and `peak_metadata_updated`. Load the output in Analysis Results; it does not overwrite active annotations. To make it the active state, explicitly replace the active file while the server is stopped.

## Synthetic data and regeneration

The bundled dataset contains three fictional participants (P901–P903), six hours of one-minute sensor readings and five-minute CGM readings each, with two artificial response peaks. Dates are January 1–3, 2030. No study records were used. Signals illustrate software behavior, not a validated physiological model; lab and meal data are omitted.

Generate a fresh example into a new directory:

```sh
python dashboard/generate_synthetic_data.py --output synthetic_demo_new --seed 20261023
SEB_DATA_ROOT="$PWD/synthetic_demo_new" python dashboard/server.py
```

The generator refuses to overwrite existing directories. Signals are reproducible for a given seed; provenance timestamps reflect execution time. Outputs follow the same three-folder layout. A newly generated example has 3 cases, 12 paired records, and 24 signal metadata records. Existing dashboard edits in the bundled dataset were preserved during migration. Both migrated and newly generated data passed checks for raw/CGM mappings, dashboard request handlers, feature recalculation, startup backfill, and conversion export. A full browser interaction test has not been performed.

## Troubleshooting

- **No files:** check `SEB_DATA_ROOT` and the printed dataset count. Processed CSVs must be under `annotated_data/series/`, not just `raw_data/`. An explicitly configured root takes precedence; even an empty package-level `annotated_data/series/` prevents automatic demo selection.
- **Missing CGM:** check participant IDs, headers, units, overlapping dates, and timestamp alignment.
- **No analysis features:** load current annotations and verify peaks have been labeled, paired, and saved.
- **Old paths after moving files:** update case keys, embedded case paths/instance IDs, and metadata `source_path`, then restart the server. The bundled demo has already been migrated; external datasets are not automatically rewritten.

## Source files

| File under `dashboard/` | Purpose |
| --- | --- |
| `server.py` | Main dashboard, preprocessing, loaders, and analysis routes |
| `data_paths.py` | Shared three-folder dataset path resolution |
| `cgm_feature_engineering.py` | Peak durations, point values, AUC, derivatives, and concentration features |
| `analysis_results_dashboard.py` | Analysis Results interface and embedding helpers |
| `server_instance.py` | Server locking and instance replacement |
| `generate_synthetic_data.py` | Reproducible data and annotation generation |

Core processing uses Python's standard library. `requirements.txt` adds NumPy and scikit-learn for optional analysis functionality; versions are unpinned. The original shared toolkit functions are merged into the main server.
