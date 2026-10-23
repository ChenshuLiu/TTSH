# Subject-level CGM reconstruction from SEG

Start with **[subject_level_cgm_modeling.ipynb](subject_level_cgm_modeling.ipynb)**. This notebook explains and runs the finalized subject-level workflow, including feature construction, calibration, held-out evaluation, prediction time-series plots, and Clarke error grids.

The package is self-contained and defaults to **entirely synthetic data**. It does not bundle private study recordings, saved patient predictions, or fitted study coefficients.

## Modeling architecture

![CGM prediction modeling architecture](CGM%20Prediction%20Modeling.png)

The diagram presents the broader original workflow. This notebook implements the finalized subject-level arm: train a separate model on each subject's earliest labeled event and evaluate later events from that same subject. Case-level alternative splits and outcome-filtered display subsets are not used here.

The model is:

$$\widehat{CGM}=b_0+b_1x+b_2x^2+b_3x^3+b_4\frac{dx}{dt}+b_5\phi.$$

`x` is smoothed and min–max normalized SEG; `dx/dt` is its native sensor-time slope per minute; `φ` is the signed, centered sigmoid of progress within the rising/falling limb. CGM is the target in mmol/L. SEG is the only sensor predictor; SEL is not required.

**Interpretation:** this is retrospective reconstruction using manually labeled CGM landmarks. Held-out events use full-event normalization, smoothing, and landmark warping. It is not a prospective, deployable real-time glucose prediction evaluation. Synthetic performance is only a software demonstration.

## Installation and notebook launch

From this folder, use Python 3.10 or newer:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab subject_level_cgm_modeling.ipynb
```

On Windows activate with `.venv\Scripts\activate`. Choose the environment's Python kernel and run the notebook from top to bottom. The numerical and figure code needs NumPy and Matplotlib; JupyterLab and ipykernel provide the notebook interface. Dependency versions are unpinned.

### Notebook sections

1. Scientific scope and architecture diagram.
2. Environment and local imports.
3. Configuration, features, and model equation.
4. Load and inspect paired-event JSON.
5. Inspect unwarped native-time inputs.
6. Landmark warping and observed-CGM bin matching.
7. Chronological calibration, overlap removal, and exclusions.
8. Fit one standardized ridge model per subject.
9. Separate calibration and held-out metrics.
10. Prediction time-series figures.
11. Clarke error-grid figures and zone summaries.
12. Export predictions, coefficients, metrics, and exclusions.
13. Use your own annotated data.
14. Reproducibility and implementation provenance.

## Finalized numerical settings

| Component | Setting |
| --- | --- |
| SEG smoothing | Original robust smoother, ratio 0.04 |
| SEG normalization | Min–max over the complete labeled event |
| Temporal alignment | Separate linear warps for trough1→peak and peak→trough2 |
| Slope | Native-time change per minute; zero for intervals <0.5 s or >60 s and each limb's first point |
| Matched samples | Feature averages in midpoint-bounded bins around observed CGM timestamps |
| Feature expansion | `[1, x, x², x³, dx/dt, φ]` before bin averaging |
| Shared peak | Deduplicate; falling-limb entry takes precedence |
| Calibration | Earliest event per subject, at least 12 retained CGM samples |
| Validation | Later events, at least 3 retained samples per event |
| Overlap | Remove validation timestamps at/before calibration end and duplicates across validation events |
| Standardization | Calibration-only means and population standard deviations; scale floor 1e-8 |
| Ridge | α = 10⁻⁶; unpenalized intercept; coefficients exported in original feature basis |
| Evaluation | MARD, RMSE, R², and Clarke zones; no MARD-based selection |

Do not skip an unusable earliest event and substitute a later event. Subjects without a valid calibration/validation combination are recorded in the exclusions table. No training samples or coefficients are pooled across subjects. Subject metrics pool that subject's retained samples; pooled-across-subject results are sample-weighted, not an unweighted mean of subject metrics.

## Input data

### Portable paired-event JSON

`data/synthetic_events.json` contains three fictional subjects with three events each. Each event has:

```text
subject_id: string
event_id: string (unique within subject)
seg: [{timestamp: ISO datetime, value: raw SEG delta current}, ...]
cgm: [{timestamp: ISO datetime, value: CGM in mmol/L}, ...]
landmarks:
   seg: {trough1: ISO datetime, peak: ISO datetime, trough2: ISO datetime}
   cgm: {trough1: ISO datetime, peak: ISO datetime, trough2: ISO datetime}
```

The top-level object contains `synthetic`, `seed`, and `events`. A bare event list is also accepted. Use one consistent timestamp basis. Observations must be finite, chronological, and unique; CGM must be positive and observations must cover the labeled event. No missing boundary extrapolation or imputation is performed. Samples can be irregular, but slopes spanning more than 60 seconds are set to zero as in the finalized model.

SEG and CGM may have different sampling rates and landmark times. The synthetic example uses 30-second SEG and five-minute CGM observations, with intentionally different event durations. Its labels and dates are invented. No clinical cohort statistics were used to fit the generator.

Regenerate into a new file:

```sh
python generate_synthetic_data.py --output local_data/new_synthetic_events.json --seed 20261023
```

### Companion alignment-dashboard data

`modeling.load_dashboard_events(dataset_root)` reads the existing three-folder layout:

```text
dataset_root/
├── raw_data/cgm/<participant>_cleaned.csv
├── annotated_data/
│   ├── alignment_annotations.json
│   └── series/<participant>/<day>/<recording>.csv
└── analysis/
```

The adapter uses saved SEG/CGM pairs, processed `Delta_Channel2_minus_Channel1` (or Channel2−Channel1 when absent), saved SEG aggregate indices where available, and labeled CGM timestamps. CGM files must use `timestamp,glucose_mmol_L` with numeric readings. Case paths must stay inside the selected dataset root. Other than reading those files, the adapter does not import or start a dashboard.

To use custom data in the notebook, change Section 4's loader, remove the synthetic-ID assertion, update provenance/synthetic status in Section 8, and choose a separate output directory. The command-line option below handles dashboard input directly. Keep private inputs outside the repository; do not commit executed notebook outputs containing them.

## Run without Jupyter

The runner follows the same preparation, fitting, evaluation, and figure functions as the notebook:

```sh
# Bundled synthetic data
python run_modeling.py --output outputs/synthetic_run

# Your own portable event JSON
python run_modeling.py --input /absolute/path/to/events.json --output outputs/custom_run

# Your own three-folder dashboard dataset
python run_modeling.py --dashboard-root /absolute/path/to/dataset_root --output outputs/dashboard_run
```

The command-line runner refuses to overwrite an existing output directory. The notebook writes to its configured `OUTPUT` directory and updates those files on subsequent runs.

## Figure and result outputs

| Output | Contents |
| --- | --- |
| `data/model_results.json` | Predictions, coefficients, metrics, settings, exclusions |
| `data/points.csv` | Matched target/prediction pairs, timestamps, splits, six feature inputs |
| `data/coefficients.csv` | One row per subject |
| `data/subject_metrics.csv` / `event_metrics.csv` | Calibration/validation metrics |
| `data/exclusions.csv` | Excluded subjects/events, when present |
| `data/clarke_summary.json` | Zone counts and percentages separately by split |
| `figures/subject_*_time_series.png` / `.svg` | All retained events per subject, with independent event facets |
| `figures/clarke_train.png` / `.svg` | Calibration Clarke grid |
| `figures/clarke_validation.png` / `.svg` | Held-out Clarke grid |
| `figures/clarke_*_points.csv` | Every plotted pair and its zone |

Black time-series traces show observed CGM, teal dashed traces calibration reconstruction, and rust dashed traces held-out reconstruction. Lines do not bridge separate events. Subject filenames use stable hashes of the identifiers; the plot title and data tables provide the identifier.

Clarke classification preserves the original figure code's boundary precedence and conversion factor **18.0 mg/dL per mmol/L**. Classification is performed in mg/dL and axes display mmol/L. The shaded grid covers 0–400 mg/dL; axes expand to include values outside it. Values are never clipped or selected by MARD. The comparator is CGM, not laboratory blood glucose, so this is not a clinical validation claim.

## Files and verification

| File | Role |
| --- | --- |
| `subject_level_cgm_modeling.ipynb` | Labeled analysis walkthrough |
| `modeling.py` | Loaders, smoothing, warping/binning, split, ridge fitting, metrics, exports |
| `plotting.py` | Reusable time-series and Clarke-grid figures |
| `run_modeling.py` | Non-interactive equivalent workflow |
| `generate_synthetic_data.py` | Standalone synthetic input generator |
| `tests/test_modeling.py` | Mathematical and data-split checks |
| `source_provenance.json` | Hashes of original source scripts; no patient-data hashes |

The numerical routines derive from `concentration_mapping_dashboard.py`, `subject_sensor_comparison.py`, `temporal_comparison.py`, and `lag_sensor_alignment.py`, orchestrated by the finalized `Subject-level SEG modeling/generate_figures.py`. Clarke geometry derives from `generate_p02_clarke_grid.py`. Original dashboard-only routes and subject-specific display filters are omitted.

```sh
python -m unittest discover -s tests -v
```

Checks cover observed-sample matching, feature expansion before averaging, training-only fitting, invalid earliest-event exclusion, minimum calibration size, overlapping validation timestamps, ridge equivalence, and Clarke boundaries. Feature vectors, timestamps, and targets were additionally compared against the original finalized figure implementation on three synthetic events. These checks establish implementation behavior, not reproduction of private cohort results.

The 11 notebook code cells were executed sequentially in Python on the bundled synthetic data (333 matched samples); PNG/SVG outputs were visually inspected. The Jupyter browser interface itself was not exercised. The companion dashboard adapter was also checked on its synthetic dataset.
