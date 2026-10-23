# TTSH manuscript analysis code

This repository organizes the manuscript's analysis into three packages:

| Package | Contents |
| --- | --- |
| [Data preprocessing and feature engineering](Data%20Preprocessing%20%26%20Feature%20Engineering/README.md) | Alignment dashboard, signal processing, peak annotation, and synthetic demonstration data |
| [CGM prediction modeling](CGM%20Prediction%20Modeling/README.md) | Subject-level modeling notebook, synthetic inputs, time-series plots, and Clarke error grids |
| [HbA1c classification modeling](HbA1c%20Classification%20Modeling/README.md) | Five model architectures, pretrained checkpoints, inference code, and synthetic in-house-format examples |

Follow each package's README for its environment, input formats, and run commands. The two modeling tasks have different dependencies; use separate virtual environments.

## Repository contents

Keep source code, notebooks with cleared outputs, requirements, documentation, architecture figures, tests, provenance files, and clearly labeled synthetic examples under version control. HbA1c checkpoints are retained because the inference examples require them. KNN checkpoints contain public training vectors as well as model metadata.

Local environments, Python/Jupyter caches, Finder metadata, server lock files, logs, run outputs, local/private data directories, and transfer ZIP archives are excluded by `.gitignore`. Public raw datasets currently remain in the HbA1c package; they are not required for the shipped checkpoint-inference examples, but support the existing preprocessing workflows.

## Publication status

The former in-house export has been replaced in the current working tree with synthetic data. **Earlier Git commits still contain the original private export.** Ignore rules and deleting files in later commits do not remove those historical objects. The repository history must be cleaned before this can be treated as a shareable source repository. A local transfer ZIP also contains in-house data; it is ignored and must not be included in a manually assembled release archive.

The synthetic examples are software demonstrations and do not reproduce private-cohort performance. See the package documentation for the retrospective modeling assumptions and validation limits.
