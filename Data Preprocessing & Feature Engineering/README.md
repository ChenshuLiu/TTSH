# SEB/CGM data processing and feature engineering

This package provides a local dashboard for inspecting sensor and CGM time series, labeling peaks and troughs, matching signal responses, and exploring engineered features in the **Analysis Results** dashboard. The code uses `SEG` for the glucose sensor and `SEL` for the lactate sensor.

## Data processing pipeline

![Data processing pipeline: wearable signals, preprocessing and delta extraction, peak-instance annotation, and feature engineering with dietary and patient metadata](Data%20Processing%20Pipeline.png)

The figure shows the full study workflow. This package contains the alignment and analysis dashboards, shared preprocessing functions, and temporal feature engineering. The original manual data-quality screening and separate meal/medicine-labeling dashboards are not included. Dietary and patient metadata require separately prepared inputs; the synthetic example demonstrates the signal-processing and peak-labeling branch. The figure's cohort counts describe the study, not the example dataset. Consult `cgm_feature_engineering.py` for the features implemented in this snapshot; the figure is not a list of automatically generated outputs.

## Quick start: run the included example

Use a Python 3 environment on macOS or Linux. Native Windows is not supported by the server's `fcntl` locking; use a Linux environment such as WSL. Run these commands from the folder containing this README and `server.py`:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python server.py
```

Open **http://127.0.0.1:8768** in your browser. Keep the terminal open while using the dashboard; press **Ctrl+C** to stop it. For subsequent runs, activate `.venv` and run `python server.py` again.

A plain launch uses `synthetic_demo/` if the default data root has no `analysis/SEB_CGM_alignment/processed_delta_files/` directory. To select the example explicitly:

```sh
SEB_DATA_ROOT="$PWD/synthetic_demo" python server.py
```

The startup output prints the selected **Data root**, **Datasets found**, source folder, and annotation file. The bundled example should show **3 datasets**. `SEB_DATA_ROOT` is the dataset root, not the processed CSV folder or the state folder.

### Use the dashboard

1. Select a recording from **File List** and inspect the raw channels and sensor/CGM plots.
2. Use the labeling controls to mark peaks and their preceding/following troughs. In the example, labels and matched pairs are already provided.
3. Click **Refresh Aggregates**, select corresponding SEG and CGM peaks, and click **Save Glu-CGM Pair**. Where relevant, use **Attach SEL to Triplet** to associate a lactate response.
4. Click **Open Analysis Results**. Load current annotations, select feature columns, and use the analysis and export controls. This page is served at **http://127.0.0.1:8768/analysis** by the same server.

Annotations are stored under the selected data root at `analysis/SEB_CGM_alignment/state/alignment_annotations.json`. Dashboard edits change that file, including when working with the synthetic example. Copy a dataset to a separate directory before experimenting if you want to preserve its original annotations.

## Use your own data

### 1. Create a dataset directory

Keep your dataset separate from the scripts. Use the following layout; `my_dataset/` is the directory to pass as `SEB_DATA_ROOT`:

```text
my_dataset/
├── Organized Data/
│   └── P001/Day01/P001_Day01_SEB.csv
├── cleaned_cgm_files/
│   └── P001_cleaned.csv
└── analysis/SEB_CGM_alignment/
    ├── processed_delta_files/
    │   └── Organized Data/P001/Day01/
    │       ├── P001_Day01_SEB.csv
    │       └── P001_Day01_SEB.csv.metadata.json
    └── state/
        └── alignment_annotations.json   # created by the dashboard if absent
```

The file list discovers CSVs under `analysis/SEB_CGM_alignment/processed_delta_files/`. Putting raw CSVs only in `Organized Data/` does not make them appear in that list. Preserve the `Organized Data/<participant>/<day>/...` suffix between raw and processed paths so the dashboard can find the original recording.

Use participant folders such as `P001` or `H001`. The CGM loader normalizes IDs to a P/H prefix and at least three digits: `P1` resolves to `P001_cleaned.csv`. Keep identifiers consistent across files and JSON. Use relative paths with forward slashes inside annotations and metadata.

### 2. Prepare sensor and CGM CSVs

Raw sensor CSVs require these columns, with numeric channel values:

```csv
Timestamp,Channel1,Channel2,Channel3
2030-01-01 08:00:00,0.110,0.150,0.144
2030-01-01 08:01:00,0.111,0.152,0.145
```

Use chronologically ordered timestamps such as `YYYY-MM-DD HH:MM:SS` (fractional seconds are also supported). Use the same time basis for sensor, CGM, and annotations; the loader does not perform timezone conversion. Keep channels in the units expected by your sensor calibration.

Independent CGM files use a different header:

```csv
timestamp,glucose_mmol_L
2030-01-01 08:00:00,5.2
2030-01-01 08:05:00,5.4
```

CGM values must be in **mmol/L**. Convert mg/dL inputs before using this format. Different sampling intervals are allowed. Preprocessing attaches CGM values at matching timestamps (or matching minutes for minute-precision sensor input); unmatched rows remain empty, while the independent CGM series is also available for feature calculations.

Processed CSVs retain `Timestamp`, `Channel1`, `Channel2`, and `Channel3`. The bundled preprocessing writer additionally exports:

```text
Delta_Channel2_minus_Channel1
Delta_Channel2_minus_Channel3
CGM
Normalized_Delta_Channel2_minus_Channel1
Normalized_Delta_Channel2_minus_Channel3
Normalized_CGM
```

Use the synthetic processed CSVs as format examples. Normalized values are generated by preprocessing; they are not additional raw measurements. An optional `CGM` column in a sensor CSV does not replace the independent CGM input when using the preprocessing helper below.

### 3. Convert raw recordings into dashboard inputs

The dashboard does not automatically preprocess every raw file at startup. After inspecting signal quality, use the shared preprocessing functions to select a segment and generate its processed CSV and metadata. The following example runs from the script folder and processes one recording; edit the dataset location, source path, and selected indices:

```sh
SEB_DATA_ROOT="/absolute/path/to/my_dataset" python - <<'PY'
import json
from pathlib import Path
import server

source = 'Organized Data/P001/Day01/P001_Day01_SEB.csv'
rows = server.load_tool2_series(source)
if not rows:
    raise ValueError('No sensor rows were loaded')

# Choose the first index at the intended baseline/asymptote.
# The end index is inclusive; these defaults select the whole recording.
start_index, end_index = 0, len(rows) - 1
processed = server.tool2_process_rows(rows, source, start_index, end_index)
relative_output = 'analysis/SEB_CGM_alignment/processed_delta_files/' + source
server.write_smoothed_rows_csv(relative_output, processed)
# Despite its name, this writer only writes the supplied rows; it adds no smoothing.
metadata_path = Path(server.ROOT_DIR) / (relative_output + '.metadata.json')
metadata_path.write_text(json.dumps({
    'tool': 'tool2',
    'source_path': source,
    'start_index': start_index,
    'end_index': end_index
}, indent=2) + '\n')
print('Created:', Path(server.ROOT_DIR) / relative_output)
PY
```

Repeat for each recording. The processing helper subtracts Channel1 and Channel3 from Channel2, adjusts both deltas relative to the selected starting sample, adds the existing `0.04` offset to the glucose delta, matches CGM readings, and generates min–max normalized columns. Smoothing can be applied using the dashboard controls. Review the baseline choice, offset, and hard-coded concentration conversion coefficients before applying the workflow to a different sensor or calibration.

### 4. Launch with your dataset

```sh
SEB_DATA_ROOT="/absolute/path/to/my_dataset" python server.py
```

Check that the startup data root and dataset count are correct, then open http://127.0.0.1:8768. If you have no saved annotations, the server creates an empty annotation file; label and pair peaks in the dashboard to populate it. Feature engineering uses existing labels and does not automatically infer missing peak/trough annotations.

To reuse annotations, place a compatible `alignment_annotations.json` in the dataset's `analysis/SEB_CGM_alignment/state/` folder **before** starting the server. Case keys must reference the corresponding processed CSV relative to the dataset root, for example:

```json
{
  "meta": {},
  "saved_segments": {
    "analysis/SEB_CGM_alignment/processed_delta_files/Organized Data/P001/Day01/P001_Day01_SEB.csv": {
      "labels_glu_cgm": [],
      "labels_glu_lac": []
    }
  },
  "peak_pair_features": {
    "cases": {
      "analysis/SEB_CGM_alignment/processed_delta_files/Organized Data/P001/Day01/P001_Day01_SEB.csv": {
        "paired_features": []
      }
    }
  }
}
```

This is an empty structural example, not a populated feature record. See `synthetic_demo/annotations_input.json` for complete labels, aggregates, paired records, and `selected_peak_metadata`. Preserve the recording's row order and timestamp alignment when reusing labels because annotations can reference row indices.

### 5. Recompute features without the dashboard

With the virtual environment active, run:

```sh
SEB_DATA_ROOT="/absolute/path/to/my_dataset" python cgm_feature_engineering.py \
  /absolute/path/to/my_dataset/analysis/SEB_CGM_alignment/state/alignment_annotations.json \
  /absolute/path/to/my_dataset/annotations_feature_engineered.json
```

This writes a separate enriched JSON and prints a processing summary. Inspect `case_errors` and `peak_metadata_updated`; an empty annotation file produces no peak features. The result can be loaded in Analysis Results. It does not replace the dashboard's active state unless you explicitly place it at the active annotation path while the server is stopped.

## Synthetic example and reproducibility

The included example contains fictional participants P901–P903, dated January 1–3, 2030: six hours of one-minute sensor readings and five-minute CGM readings per participant, with two artificial response peaks. No real participant records were used. These Gaussian responses illustrate software behavior and are not a validated physiological model. Lab workbooks and meal photos are not included; meal-dependent metadata remains unset.

| File or folder under `synthetic_demo/` | Purpose |
| --- | --- |
| `Organized Data/`, `cleaned_cgm_files/` | Raw sensor and independent CGM inputs |
| `analysis/SEB_CGM_alignment/processed_delta_files/` | Processed CSVs and metadata JSON |
| `annotations_input.json` | Labels and paired metadata before feature engineering |
| `analysis/SEB_CGM_alignment/state/alignment_annotations.json` | Enriched annotations used by the dashboard |
| `feature_engineering_summary.json` | Processing counts and case errors |
| `manifest.json` | Generation settings, pipeline, and limitations |

To generate another example, choose a new output directory (the generator refuses to overwrite an existing directory):

```sh
python generate_synthetic_data.py --output synthetic_demo_new --seed 20261023
SEB_DATA_ROOT="$PWD/synthetic_demo_new" python server.py
```

Signal generation is reproducible for a given seed; provenance timestamps reflect execution time. The included run processed 3 cases, 12 paired records, and 24 signal metadata records without errors. Page generation and the files, current-analysis, series, and peak-feature request handlers were checked against the example; a full browser interaction test has not been performed.

## Troubleshooting

- **No recordings appear:** check the printed data root and dataset count. Confirm the processed CSV directory matches the layout above. An explicitly set `SEB_DATA_ROOT` takes precedence over the demo. A local processed-data directory, even if empty, prevents automatic demo selection; select the demo explicitly if needed.
- **Files appear but CGM is absent:** check participant naming, the `timestamp,glucose_mmol_L` header, overlapping dates, units, and timestamp alignment.
- **Analysis Results has no features:** load current annotations and confirm labeled peaks have been paired and saved. Raw CSVs alone do not provide paired-feature records.
- **Changes to a path seem ineffective:** stop the server, launch again with the intended data root, and refresh the browser.

## Source files and dependencies

| File | Purpose |
| --- | --- |
| `server.py` | Main dashboard, shared loaders, preprocessing, and analysis routes |
| `cgm_feature_engineering.py` | Peak durations, point values, AUC, derivatives, and concentration features |
| `server_instance.py` | Server startup locking and instance replacement |
| `analysis_results_dashboard.py` | Analysis Results interface and embedding helpers |
| `generate_synthetic_data.py` | Reproducible artificial input and annotation generation |
| `requirements.txt` | NumPy and scikit-learn for optional analysis functionality |

The core processing code uses Python's standard library. Requirements are unpinned and are not a validated environment lock. The original standalone screening server has been merged into `server.py`; no separate `toolkit_server.py` is required.
