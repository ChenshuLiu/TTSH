# Synthetic in-house-format example

All records in this directory are fictional. Only the column names and export structure were retained from the former in-house interface. No real participant measurements, dates, identifiers, clinical values, fitted coefficients, hashes, or free-text records are used by the generator.

## Files

| File | Contents |
| --- | --- |
| `predicted_cgm_timeseries.csv` | 999 synthetic observed/predicted CGM samples; 17 columns |
| `event_features.csv` | 27 synthetic event records; 230 columns |
| `patient_metadata.csv` | 9 fictional subjects; 38 columns |
| `timeseries_with_features_and_metadata.csv` | 999 joined rows; original 278-column schema |
| `by_subject/SYNTHETIC_*.json` | Per-subject metadata, events, features, and samples |
| `inference_events.json` | 27 model-ready synthetic events, each with 64 mmol/L values |
| `schema.json` | Column names only; no source records or statistics |
| `data_dictionary.json` | Synthetic provenance and generated nonmissing counts |
| `manifest.json` | Seed, synthetic status, dimensions, and limitations |

`subject_id` joins patients to events and samples; `event_uid` joins events to time-series samples. Case paths are synthetic logical identifiers, not references to real recordings. The `actual_cgm_mmol_L` and `predicted_cgm_mmol_L` names preserve the interface: both are generated curves, and the latter is not the output of a trained SEG model. HbA1c values are invented subject-level labels. Unknown clinical fields, meal fields, and fitted coefficients are intentionally null (blank in CSV).

Events span 180 minutes with 37 five-minute samples, troughs at the boundaries and a peak at 90 minutes. CGM and SEG point values, durations, trapezoidal AUC, and normalized-time derivatives are calculated from generated curves. Predictions are analytic curves with a small deterministic deviation. Unsupported conversion/model-input fields remain null. Model-ready inputs are obtained by linear interpolation of the synthetic predicted CGM to 64 points.

The `train`/`validation` columns illustrate an event-level CGM calibration split only. They are not a valid subject-level HbA1c training/evaluation split. Do not use synthetic performance to claim clinical accuracy; class-dependent toy signals are deliberately easy to distinguish.

## Regenerate

From the modeling package root, choose a new directory:

```sh
python scripts/generate_synthetic_inhouse.py --output outputs/synthetic_inhouse_new --seed 20261023
```

The generator needs only Python's standard library and `schema.json`. It does not read any in-house source files and refuses to overwrite an existing directory.

## Run an example prediction

After installing the modeling requirements, from the package root:

```sh
python predict.py --model transformer --seed 42 \
  --input "cgm data/synthetic_inhouse/inference_events.json" \
  --output-dir outputs/synthetic_inhouse_demo
```

The model consumes only each event's CGM values; invented patient metadata and manual event features are not predictors for the supplied checkpoints.
