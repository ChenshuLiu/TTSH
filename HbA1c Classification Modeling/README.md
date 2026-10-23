# HbA1c classification models — CGM only

This package contains five model implementations, 25 trained checkpoints (five architectures × seeds 42–46), inference examples, and experiment configurations. Models predict probabilities for three HbA1c categories; they do not regress a continuous HbA1c value. The supplied checkpoints use CGM sequences only, without demographic metadata or manually engineered features.

The former in-house export folder has been removed from the working tree and replaced with **entirely synthetic, schema-compatible examples** in [`cgm data/synthetic_inhouse/`](cgm%20data/synthetic_inhouse/README.md). Public datasets and public-data checkpoints are retained. Previously committed private files may still exist in Git history; working-tree replacement does not purge that history.

## Modeling logic

![CGM sequence classification architectures: KNN, 1D CNN, TCN, BiGRU, and Transformer](HbA1c%20Modeling%20Architectures.png)

Each model receives a resampled CGM event. KNN predicts through distance-weighted neighbor voting; the neural models encode the sequence, pool its representation, and produce three class probabilities. The figure is schematic; the configuration table below specifies the shipped implementations, including all four TCN dilation blocks.

## Package contents

```text
HbA1c Classification Modeling/
├── README.md
├── HbA1c Modeling Architectures.png
├── requirements.txt
├── predict.py                         # Model loading and event/subject inference
├── verify_bundle.py                   # Verify all 25 checkpoints
├── model_manifest.json                # Checkpoints, configurations, provenance, hashes
├── verified_environment.json          # Previously recorded verification environment
├── verification_report.json           # Previous model verification report
├── SHA256SUMS                         # File integrity inventory
├── configs/
│   ├── experiment.json
│   └── segment_config.json
├── examples/
│   ├── example_events.json            # Small synthetic inference input
│   └── load_model.py
├── scripts/
│   └── generate_synthetic_inhouse.py
├── cgm data/
│   ├── synthetic_inhouse/             # Fictional CSV/JSON export examples
│   └── ...                            # Existing public datasets and preprocessing tools
├── cgm_bag/                           # Model, training, normalization, and audit code
└── checkpoints/
    ├── knn/seed_42...seed_46/          # model.npz + train_config.json
    ├── 1d_cnn/seed_42...seed_46/        # model.pt + train_config.json
    ├── tcn/seed_42...seed_46/
    ├── bigru/seed_42...seed_46/
    └── transformer/seed_42...seed_46/
```

## Install and predict

The recorded verification environment uses Python 3.12. From this package folder:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Inference only; this does not start training.
python predict.py --model transformer --seed 42 \
  --input examples/example_events.json --output-dir outputs/demo

# Check all 25 models with synthetic input.
python verify_bundle.py

# Python API example.
python examples/load_model.py
```

On Windows, activate with `.venv\Scripts\activate`. Entry points default to CPU and locate checkpoints relative to the scripts. Keep the package structure together when moving it. Floating-point results may vary slightly across hardware or dependency versions.

`--model` accepts `knn`, `1d_cnn`, `tcn`, `bigru`, or `transformer`; `--seed` accepts 42–46. For example:

```sh
python predict.py --model knn --seed 46 \
  --input "cgm data/synthetic_inhouse/inference_events.json" \
  --output-dir outputs/synthetic_knn
```

Python API:

```python
import json
from pathlib import Path
from predict import load_model, predict

events = json.loads(Path("examples/example_events.json").read_text())
loaded = load_model("transformer", seed=42)
tables = predict(loaded, events)
event_probabilities = tables["events"]
patient_probabilities = tables["patients"]
```

## Input format and preprocessing

Input is a JSON list of events. Every event has `subject_id`, `event_id`, and `values`: exactly 64 chronological CGM values in **mmol/L**. See `examples/example_events.json` for a complete runnable example. Event IDs must be unique within a subject. Inference does not require HbA1c labels, metadata, or manual features.

Supply **unstandardized absolute glucose values**. The original pipeline converts mg/dL by dividing by `18.0182`. The inference wrapper applies the scaler stored with the selected checkpoint; do not apply a second z-score or fit a replacement scaler on evaluation data.

All checkpoints use a complete **trough–peak–trough (TPT)** event resampled to 64 points. Select and quality-check the event in real time, then interpolate by timestamp:

```python
import numpy as np
# t: increasing valid observation times in seconds; g: glucose in mmol/L.
target = np.linspace(t[0], t[-1], 64)
values = np.interp(target, t, g).astype(np.float32)
```

The 64 points span each event's own duration; they are not a fixed sampling interval. Interpolate by elapsed time, not CSV row index, and do not extrapolate beyond the first/last valid observation. Public-data median smoothing was used to locate landmarks; model inputs were generated from unsmoothed CGM measurements.

| Extraction setting | Value |
| --- | --- |
| Event type / resampled length | Trough–peak–trough / 64 |
| Minimum valid raw observations | 5 |
| Duration | 20–360 minutes |
| Maximum gap between valid observations | 25 minutes |
| Maximum missing fraction | 0.2 |
| Glucose range | 1–40 mmol/L |
| Minimum amplitude | 0.4 mmol/L |
| Peak prominence | 0.5 mmol/L |
| Landmark smoothing window | 15 minutes |
| Minimum peak separation | 30 minutes |
| Trough search extent on each side | 180 minutes |

For sensor-derived CGM exports, use `predicted_cgm_mmol_L`. Observed `actual_cgm_mmol_L` may be evaluated separately as an oracle comparator. Complete events need at least two valid points on both rising and falling portions, with the peak inside the retained time range. Do not pad a trough-to-peak-only event to impersonate a complete TPT event.

`predict.py` validates format, length, uniqueness, and finite values. A 64-point array alone cannot establish whether the original timing and landmark quality criteria were met; check those upstream.

## Model architectures and outputs

| Model | Shipped configuration |
| --- | --- |
| KNN | Standardized 64-point vector; Euclidean distance; 5 neighbors; distance-weighted voting |
| 1D CNN | Three Conv1d layers, kernels 5/5/3, 32 channels, GroupNorm/GELU; global mean pooling; Linear(32,3) |
| TCN | 1×1 input projection; four causal residual blocks with dilations 1/2/4/8, two kernel-3 convolutions per block; temporal mean pooling; Linear(32,3) |
| BiGRU | One bidirectional GRU layer, hidden size 16 per direction; mean over all time steps; dropout; Linear(32,3) |
| Transformer | Linear(1,32) time-step embedding and sinusoidal position encoding; two encoder layers, four heads, FFN size 64; temporal mean pooling; Linear(32,3) |

BiGRU uses all-time-step mean pooling rather than only the final hidden state. Transformer tokens represent individual time steps, not patches. KNN is supervised classification, not clustering.

Neural checkpoints use the `PatientBagModel` container with `sample_unit="event"`, `K=1`, and `pooling="mean"`. Each event is one input; the underlying tensor shape is `[batch, 1, 64]`. Multiple events per subject are not required. Use `predict.py` to apply the saved scaler correctly.

Class order and the experiment's HbA1c label definitions are fixed:

| Index | Label | HbA1c (%) |
| --- | --- | --- |
| 0 | Healthy | < 5.7 |
| 1 | Prediabetes | ≥ 5.7 and < 6.5 |
| 2 | T2D | ≥ 6.5 |

Event and subject outputs contain `p_Healthy`, `p_Prediabetes`, `p_T2D`, predicted class, and confidence. Subject probabilities are the arithmetic mean of that subject's event probabilities, followed by argmax; aggregation is not majority voting.

Earlier external evaluation focused on Prediabetes and T2D, but the models still output three classes. A two-class display must account for errors predicted as Healthy; do not discard those cases or renormalize the other probabilities and present them as the original model outputs.

## Training design and checkpoint provenance

The recorded public training sources are Hall2018, CGMacros, and BIG IDEAS: 116 labeled subjects and 7,092 eligible events. Training and validation were split by subject, with approximately 25% public validation. Events from one subject do not cross that split.

Each seed defines one subject split shared by all five architectures. Across seeds, splits change, along with neural initialization and sample order. These are **five repeated random splits, not five-fold cross-validation**. KNN differences arise from the training split, not random initialization of neighbor inference.

| Training option | Setting |
| --- | --- |
| Seeds | 42, 43, 44, 45, 46 |
| Neural training limit | 200 epochs |
| Early stopping | Public validation loss; patience 10 |
| Optimizer | AdamW |
| Learning rate / weight decay | 0.0001 / 0.0001 |
| Batch size | 16 |
| Hidden dimension / dropout | 32 / 0.1 |
| Class weighting | Enabled; based on public training event counts |
| Scaling | Patient-balanced global z-score fitted on public training subjects only |
| Metadata / engineered features | Not used |
| Private-data fine-tuning | Not performed for the supplied checkpoints |

Neural weights correspond to the lowest public validation loss. Actual best epochs, run lengths, subject counts, and event counts are in `model_manifest.json`; 200 is a limit, not the number of epochs completed by every model.

Neural `.pt` files include weights, configuration, scaler, public subject splits, and training history. Loading uses `torch.load(..., weights_only=True)` with strict model/split checks. KNN `.npz` files include public training vectors, labels, subject/event identifiers, scaler, and provenance checks. KNN needs its training instances at inference and therefore is **not a data-free weights artifact**. Loading uses `allow_pickle=False` and reconstructs the neighbor estimator from stored instances; it does not select new training data.

## Synthetic replacement for the in-house export

[`cgm data/synthetic_inhouse/README.md`](cgm%20data/synthetic_inhouse/README.md) documents the replacement. It retains the four original CSV schemas, including the 278-column joined table, and provides per-subject JSON. New records use `SYNTHETIC_*` identifiers, invented 2035 timestamps, analytic curves, and fictional metadata. Unused clinical fields, meal fields, and fitted coefficients are null. No individual records or cohort statistics were used to fit the generator.

There are 9 subjects, 27 events, and 999 time-series samples. `inference_events.json` supplies 27 resampled events ready for `predict.py`. Synthetic examples demonstrate schema compatibility and execution; they cannot establish clinical accuracy or validate the original study results. Their illustrative CGM train/validation labels are not an HbA1c subject-level evaluation split.

Regenerate without accessing private source files:

```sh
python scripts/generate_synthetic_inhouse.py \
  --output outputs/synthetic_inhouse_new --seed 20261023
```

The standard-library generator reads only the column-name schema and refuses to overwrite an existing directory. The old private export path is ignored by Git to reduce accidental reintroduction. This does not remove previously tracked files from historical commits; history cleanup is a separate step before publication.

## Verification and reuse

`verify_bundle.py` checks the 25-model roster, seeds, configurations, checkpoint hashes, strict loaders, valid three-class probabilities, and subject aggregation using synthetic inputs. `verification_report.json` and `verified_environment.json` record the previous handoff verification; they are not a newly executed validation of this documentation/data replacement.

The replacement data were checked for matching CSV schemas, consistent subject/event joins, unique sample timestamps, and 64-point inference arrays. Checkpoint contents are unchanged. Full neural inference was not rerun during replacement because the current Python environment lacks the required modeling dependencies. After installing requirements, run `python verify_bundle.py` and the synthetic prediction command above.

Synthetic checks do not measure accuracy, F1, or AUROC. `requirements.txt` records versions used for the prior verification, not an environment lock captured during original training. Manifest `original_path` fields record provenance; the inference loader does not depend on those paths.

Training code remains available through `cgm_bag.training.train_model` and `cgm_bag.knn.train_knn`. For retraining, prepare raw events and use the corresponding saved `train_config.json` rather than older default bag settings. Running the prediction or verification commands does not train new models.
