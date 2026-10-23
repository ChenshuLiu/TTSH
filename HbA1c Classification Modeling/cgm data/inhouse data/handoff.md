# Handoff: HbA1c classification using SEG-derived CGM and peak features

Snapshot: 2026-09-29. Audience: the receiving analyst and their AI assistant.

## 1. Read this first

This is a **retrospective, subject-calibrated research dataset**, not a validated
HbA1c classifier or a prospective glucose monitor. The requested next task is to
investigate HbA1c classification using the supplied predictions, peak features,
and optionally demographics. No HbA1c classification has yet been performed.

The predicted variable in this folder is **CGM glucose**, not CRP and not HbA1c.
`patient.hba1c` is a recorded patient-level lab measurement intended as the outcome.
The class definitions, thresholds, positive class, and intended clinical question
have NOT been specified. Confirm those with the researcher before creating labels;
do not choose cutoffs by maximizing performance on these data. Confirm HbA1c units
with the source owner before applying a unit-dependent cutoff.

Critical constraints:

- There are **11 subjects with known HbA1c**, not 1,038 independent labeled subjects.
- The current export has **44 of the original 55 CGM-SEG events**. It has NOT yet
  been expanded to all 55. See exclusions below.
- The CGM `train`/`validation` column is NOT an HbA1c train/test assignment.
- No information, feature selection, imputation, scaling, or threshold tuning may
  pass from an HbA1c evaluation subject into the HbA1c classifier's training fold.
- Predictions depend on same-subject CGM calibration and manual matched CGM
  landmarks. Do not describe them as glucose predictions independent of CGM access.
- Identifiers are pseudonyms, not anonymization. Keep the data private; do not
  upload clinical records to an external service without the data owner's approval.

## 2. Files and population

All paths below are relative to this handoff file. The CSVs and JSONs can be read
without the original dashboard, Python modeling environment, or source recordings.

| File | Contents / unit of observation | Shape |
| --- | --- | --- |
| `predicted_cgm_timeseries.csv` | Observed and predicted CGM at one timestamp | 1,038 rows, 17 columns |
| `event_features.csv` | Full saved metadata/features for one matched SEG-CGM event | 44 rows, 230 columns |
| `patient_metadata.csv` | Available clinical/lab/demographic data for one subject | 12 rows, 38 columns |
| `timeseries_with_features_and_metadata.csv` | Time-series sample with its event and patient data repeated | 1,038 rows, 278 columns |
| `by_subject/<subject>.json` | Nested patient data, model coefficients, events, features, time series | 12 files |
| `manifest.json` | Counts, source hashes, settings, coefficients and exclusions | Provenance |
| `data_dictionary.json` | Per-file column provenance and nonmissing counts | Schema inventory |
| `README.md` | Short dataset description | Documentation |
| `export_data.py` | Rebuilds the export using the original project sources | Optional provenance code |

Included subjects: H01, P01, P02, P04, P05, P07, P09, P10, P14, P16, P19, P20.
H01 has no lab record and no HbA1c in this export. Do not infer its HbA1c or class
from its identifier. After excluding H01 for supervised HbA1c analysis, there are
11 subjects, 40 events and 1,005 time-series points, still only 11 independent targets.

The complete prediction export has 12 calibration events (298 points) and 32
held-out events (740 points). Both splits are retained, explicitly labeled.

### Why 44, not 55 events?

This export inherits eligibility from the subject-specific CGM modeling run:

| Excluded subject | Number of annotated events excluded | Reason |
| --- | ---: | --- |
| P03 | 4 | First event has 8 observed CGM calibration points; minimum was 12 |
| P08 | 4 | First event has 8 observed CGM calibration points; minimum was 12 |
| P06 | 1 | Only one event, so no later independent event for validation |
| P11 | 1 | Only one event |
| P17 | 1 | Only one event |
| Total | 11 | 55 annotated events minus 11 = 44 exported events |

These are model-protocol exclusions, not absent annotations or confirmed poor
signal quality. Expanding to all 55 requires the original source annotations and
lab data. Missing predictions must be flagged, not fabricated. Changing the
calibration eligibility or fitting within-event models would be a new protocol
and must be reported separately. No alternative model was fitted to these cases.

## 3. How the CGM predictions were produced

### Sensor input and preprocessing

1. Use the processed SEG delta-current signal, identified in feature metadata as
   `Delta_Channel2_minus_Channel1`. This package does not contain complete raw
   recordings; it contains the saved model input features and CGM prediction points.
2. Retrieve each SEG event using its own manually annotated trough1, peak and
   trough2. Do NOT select SEG boundaries from the CGM timestamps.
3. Smooth SEG within the full event. Ratio is 0.04. The centered window is
   `max(5, round(N * 0.04))`, made odd and capped at 301 samples. Replace finite
   outliers with the local median when deviation exceeds `6 * 1.4826 * MAD` and
   MAD is positive, then apply a centered moving mean over finite values.
   This is a sample-count window, not necessarily a fixed duration.
4. Min-max normalize AFTER smoothing: `x = (SEG_smooth - min)/(max - min)` using
   the full event. Constant events use 0.5 for finite inputs. This emphasizes shape
   and removes event-specific absolute level/amplitude. CGM targets are NOT normalized.

SEG converted concentrations are also available among the saved event features:
`SEG_converted_concentration_uM = (raw_delta_value - 0.00453) / 0.00128`.
The model input is normalized smoothed delta current, not absolute uM. Min-max
normalization after this positive affine conversion would be equivalent, but the
saved raw/concentration landmark features are distinct from model inputs. Do not
confuse legacy `converted_concentration` with explicitly named `_uM` quantities.

### Alignment, sampling and regression

Match SEG trough1/peak/trough2 to the corresponding labeled CGM landmarks. Rising
and falling limbs are linearly warped separately:

```text
t_CGM = T_a + (t_SEG - t_a) / (t_b - t_a) * (T_b - T_a)
```

Lower-case endpoints refer to the SEG limb; upper-case endpoints to the CGM limb.
CGM and SEG event durations may differ. Timestamp warping is separate from value
regression and uses labels from the event being reconstructed, including held-out
events. It is not a learned prediction of CGM peak timing.

The SEG-only regression is:

```text
CGM_pred = b0 + b1*x + b2*x^2 + b3*x^3 + b4*dx/dt + b5*s(tau)
s(tau) = d * (2 / (1 + exp(-10*(tau - 0.5))) - 1)
d = +1 for trough1 -> peak, -1 for peak -> trough2
tau = relative progress within the current SEG limb, from 0 to 1
```

`dx/dt` uses native SEG time in minutes. Invalid intervals below 0.5 seconds or
above 60 seconds receive zero slope; the first sample of each limb also has zero
slope. This run uses the matched-fitting variant without the legacy slope clipping.
The derivative is diffusion-inspired, NOT a measured permeability or a direct
solution of Fick's law. The sigmoid is a retrospective event-phase feature.
There are no SEL terms in these predictions.

Feature vectors are averaged in bins bounded by midpoints between observed CGM
timestamps, clipped to the labeled limb. Target values and timestamps remain the
original CGM samples. Squared/cubed features are formed BEFORE averaging; they are
not the square/cube of the bin-mean input. Do not resample these exported points to
1 Hz and treat interpolated points as independent measurements.

Fit a separate model for each subject using their earliest chronological labeled
event. Standardize feature columns on that subject's training data only, then use
ridge regression with alpha `1e-6` and an unpenalized intercept. Coefficients in the
JSON/manifest are transformed back to the original feature basis.

Use later events from the SAME subject for validation; no cross-subject fitting.
Require at least 12 calibration CGM points and one later event with at least 3
retained points. Do not skip an insufficient first event to find a better one.
Deduplicate CGM timestamps, remove validation samples at/before calibration end,
and remove duplicate target timestamps across validation events.

Full-event smoothing, normalization, phase labels and CGM landmark alignment are
intentional: the researcher allows retrospective modeling. This is not prospective
rolling prediction. A new subject needs same-subject calibration under this protocol.

## 4. Data interpretation and joins

- Use `subject_id` to join patient metadata. Use `event_uid` to join event features.
  The latter hashes the subject, case path, and SEG/CGM aggregate IDs.
- `event_id` is the original dashboard ordinal and may change after annotation
  edits. Event names can repeat across days and even within a day; do not join by
  event name alone. The export verified subject, day, event name, split and start
  timestamp against source annotations, with annotation hashes checked first.
- `day` is the recording/day label. An event can cross midnight, so it need not
  equal the calendar date extracted from `cgm_timestamp`.
- `cgm_timestamp` is a readable local wall-clock timestamp (study timezone
  Asia/Singapore). `elapsed_seconds_from_cgm_event_start` is already seconds.
  Do not divide it by 1,000. It is relative to the labeled CGM event start, which
  need not be the first retained sample.
- Both `actual_cgm_mmol_L` and `predicted_cgm_mmol_L` are absolute mmol/L.
- `seg.*` and `cgm.*` are flattened saved event fields: landmark times and values,
  durations, derivatives, AUC, conversions and preceding meal/medication information.
  `pair.*` contains pair metadata and delays; use directional names to determine sign.
- Event features describe the complete labeled event, not only the retained
  prediction timestamps. They were exported as saved, not recomputed using the
  modeling smoothing setting. Do not assume identical preprocessing.
- Saved interval derivatives have time scaled to 0..1, as specified in their
  `time_normalization` field. They are NOT per-second slopes. In contrast,
  `model_input.mean_seg_rate_per_minute` is the averaged native-time model slope.
- Duration fields with `_seconds` are seconds. AUC uses timestamp-second
  trapezoidal integration; retain the recorded `method`, `value_source` and
  conversion formulas to distinguish units.
- `label_timestamp` is the chosen label time; `matched_timestamp` identifies the
  sampled signal value used for the feature. Their offset can be nonzero.
- Patient columns retain source values and categorical codes as recorded. Request
  a codebook for gender, ethnicity, medication and history codes; do not guess labels.
  Check units for lab fields other than CGM before converting or thresholding them.
- CSV missing cells are blank, not zeros. JSON fields can be absent/null. Do not
  overwrite valid zero values in checkbox/categorical fields. Initials were omitted.
- `prediction_metrics.*` contains CGM error metrics, not physiological predictors.
  MARD is a percentage; R-squared may be negative and must not be silently truncated.
- Drawing lines across within-event gaps was a display-only change. The exports
  contain the same observed sample locations, with no synthetic filled-in points.

## 5. Guidance for the receiving classification analyst/AI

1. Confirm the HbA1c classification definition, assay units, class counts and
   timing of the lab measurement relative to recordings. Timing is not established
   by the time-series timestamps. Do not label this a future-outcome forecast.
2. Choose and document the input scenario BEFORE evaluation. Possible comparisons:
   demographics-only; SEG event features; predicted-CGM summaries; and their
   combination. Observed-CGM features may be a separate comparator, not silently
   included in a purported sensor-only model. Predicted CGM here is itself
   calibration-assisted and landmark-aligned using observed CGM.
3. For a clean initial analysis, summarize only `split == validation` predicted-CGM
   samples into subject-level features; retain calibration-inclusive analysis as
   a separately labeled sensitivity analysis. This is a suggested downstream
   policy, not a filter already applied to the delivered files.
4. Choose a small prespecified set of interpretable features. For example, event
   mean/median predicted CGM and selected SEG duration/shape features, then subject
   summaries across events. State whether subjects' events are equally weighted
   or weighted by sample count/duration. These are selected peak windows, NOT
   full-day glucose coverage or an unbiased average glucose exposure.
5. Keep all events/points from a subject in one HbA1c fold. Subject aggregation is
   usually simpler than treating the repeated rows as a large sample. Demographics
   and HbA1c repeat across rows; do not average duplicate joined rows as if they
   were separate patients. Do not perform a random time-point/event train-test split.
6. Exclude `patient.hba1c` and any HbA1c-derived label from predictors. Also exclude
   IDs, paths, administrative fields, error metrics and arbitrary codes used only
   for joining. Review all lab variables for outcome proxies and intended use.
7. Fit imputation, scaling, feature selection, dimension reduction and any tuning
   only within training subjects. With 11 labeled subjects and hundreds of fields,
   unrestricted feature searches will be highly unstable. Prefer a simple,
   regularized, low-dimensional model and report exploratory status.
8. Use subject-level cross-validation feasible for the actual class counts. If
   leave-one-subject-out is used, AUROC is undefined for a one-person test fold;
   calculate it from the combined out-of-fold predictions, provided both classes
   are present. If any training fold lacks a class, revise the design or report
   that the proposed classifier cannot be evaluated. Never replace undefined
   metrics with zero. Small-sample tuning requires particular caution.
9. Report the class definition/counts, confusion matrix, AUROC, F1, precision,
   sensitivity, specificity and balanced accuracy as applicable. Specify the
   positive class, decision threshold and averaging convention. For multiclass
   analyses, state one-vs-rest and macro/weighted choices. Preserve one out-of-fold
   result per subject; any uncertainty resampling must use subjects, not points.
10. Report exclusions and missingness explicitly. H01 is not a negative-class
    example just because HbA1c is missing. Do not claim 55-event coverage for this
    package or use the CGM validation split as proof of HbA1c generalization.

## 6. Minimal loading and verification example

Python example, requiring pandas; it loads the package without running the original
model. It deliberately does not invent class thresholds or fit a classifier.

```python
from pathlib import Path
import pandas as pd

root = Path("HbA1c prediction data")  # adjust to the received folder
ts = pd.read_csv(root / "predicted_cgm_timeseries.csv", dtype={"subject_id": str, "event_uid": str})
events = pd.read_csv(root / "event_features.csv", dtype={"subject_id": str, "event_uid": str})
patients = pd.read_csv(root / "patient_metadata.csv", dtype={"subject_id": str})
assert len(ts) == 1038 and len(events) == 44 and len(patients) == 12
assert events.event_uid.is_unique and patients.subject_id.is_unique
assert not ts.duplicated(["subject_id", "cgm_timestamp"]).any()

joined = ts.merge(events, on=["event_uid", "subject_id"], validate="many_to_one",
                  how="left", suffixes=("", "_event"))
joined = joined.merge(patients, on="subject_id", validate="many_to_one", how="left")
assert len(joined) == len(ts)
patients["patient.hba1c"] = pd.to_numeric(patients["patient.hba1c"], errors="coerce")
labeled_subjects = patients.loc[patients["patient.hba1c"].notna(), "subject_id"]
assert len(labeled_subjects) == 11
held_out = joined.loc[(joined["split"] == "validation") &
                      joined["subject_id"].isin(labeled_subjects)].copy()
# Next: prespecify subject-level features and HbA1c class definition with the owner.
```

## 7. Provenance, regeneration and expected deliverables

The model predictions came from the adjacent `Subject-level SEG modeling` folder.
The authoritative peak metadata source was
`analysis/SEB_CGM_alignment/state/alignment_annotations.json`, under
`peak_pair_features`. The older standalone `peak_pair_features.json` has some
different timestamps and was NOT used to populate these event features.
Patient metadata came from `analysis/SEB_CGM_alignment/Lab_data.xlsx`.
`manifest.json` records source hashes and the exact fitted coefficients.

`export_data.py` needs that original project layout and source files to regenerate
the package; copying this folder alone is sufficient for downstream analysis but
not for reconstructing raw recordings or refitting the CGM models. If annotations
change, regenerate predictions first. Do not bypass source-hash checks to join
old ordinal IDs to new events.

The export has been checked for matching annotation hashes, subject/event identity,
unique subject-timestamp pairs, exact prediction/observed-value preservation after
CSV export, complete joins, matching HbA1c values, and per-subject JSON row counts.

Requested downstream deliverables should include: the agreed outcome definition;
subject-level feature table; fold assignments; per-subject out-of-fold class scores
and predictions; classification metrics and uncertainty limitations; exact model
and preprocessing choices; exclusions; and reproducible code. Keep outputs separate
from these source exports and do not silently overwrite the supplied data.
