# HbA1c prediction data

Dataset assembly for subsequent HbA1c analysis. No HbA1c model has been fitted.
HbA1c is the recorded patient lab value, not a predicted value. The time series
contains the existing subject-specific SEG-only predicted CGM values and their
observed CGM comparators. No models were retrained and no missing values imputed.

## Files

| File | Row meaning | Rows |
| --- | --- | ---: |
| `predicted_cgm_timeseries.csv` | One observed/predicted CGM timestamp | 1,038 |
| `event_features.csv` | One matched SEG/CGM event, all saved event features | 44 |
| `patient_metadata.csv` | One subject, available demographics and lab/clinical data | 12 |
| `timeseries_with_features_and_metadata.csv` | Time-series sample joined to event features and metadata | 1,038 |
| `by_subject/<subject>.json` | Patient metadata, coefficients, events, features and chronological samples | 12 files |

The fully joined CSV has 278 columns. Use the separate tables to avoid repeating
event/subject values in storage; use the joined table for direct data exploration.
`data_dictionary.json` records column provenance and nonmissing counts.
`manifest.json` records settings, source hashes, counts, exclusions and model
coefficients. `export_data.py` recreates the exports from their source files.

## Contents and joins

- `event_uid` joins time-series samples to event features. It is a hash of the
  subject, source case path and both aggregate landmark IDs, not a row number.
- `subject_id` joins patient metadata. `event_id` retains the dashboard identifier.
- Every event was checked against its prediction subject, day, name, split and
  CGM start timestamp. Annotation hashes must match those saved during modeling;
  export fails rather than silently joining predictions to changed annotations.
- Features come from `alignment_annotations.json` -> `peak_pair_features`, the
  authoritative source read by the dashboard. The older standalone
  `peak_pair_features.json` is not used to populate features.
- All saved SEG and CGM fields are retained: durations, interval derivatives, AUCs,
  trough/peak values, conversions, landmark timestamps, matched-point timestamps,
  preceding meal/medication information, and pairwise delays. SEL fields are not
  included. Features describe the full labeled event, even if some target points
  were removed from evaluation because of overlap.
- Patient data come from `Lab_data.xlsx` using the existing server loader. All
  available clinical/demographic fields are retained except participant initials.
  Gender/ethnicity/medication codes remain as recorded; no code labels are guessed.
- H01 has no lab record, so metadata and HbA1c remain missing. Eleven subjects have
  recorded HbA1c. Missing cells are blank in CSV and absent/null in JSON, never zero.

## Units and model context

- CGM values are mmol/L. Timestamps are source local wall-clock strings, not Unix
  milliseconds. `elapsed_seconds_from_cgm_event_start` is already in seconds.
- Duration fields ending in `_seconds` are seconds. AUC metadata retain their
  signal value source and timestamp-second integration method.
- Saved event derivatives use interval time normalized to 0..1. They are NOT
  derivatives per second. Their `time_normalization` and `value_source` fields
  explain the exact definitions.
- `model_input.mean_seg_rate_per_minute` is different: it is the bin-averaged rate
  of change of min-max normalized SEG in native sensor minutes. Squared/cubed
  inputs are means of the expanded features, not powers of the bin mean.
- `train` marks the earliest event used to calibrate each subject's separate
  model (298 points); `validation` marks later held-out events (740 points).
  No subjects were mixed for model fitting. Full-event normalization, smoothing
  ratio 0.04, phase and manual time warping make this retrospective reconstruction.

## For subsequent HbA1c modeling

HbA1c is a SUBJECT-level outcome. Repeating its value across event/time-series rows
does not create independent outcome observations. Split future HbA1c training and
evaluation by subject; do not randomly split these repeated rows. Exclude
`patient.hba1c` from predictors when it is the target. Keep observed CGM features
separate when evaluating a sensor-only prediction claim. The existing CGM
train/validation labels are not an HbA1c train/test split. Both are retained here
so downstream analyses can explicitly choose their inclusion policy.
