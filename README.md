# Rush Hour: hourly bike-demand forecasting with a chained ML pipeline

A six-stage machine-learning pipeline that forecasts the hourly rental count of a bike-sharing system (`cnt`) and classifies the hours that are unusually busy. Each stage implements one algorithm and consumes the verified output of the stage before it. The core solvers are written in NumPy, and scikit-learn is used only for the seeded split and as a test oracle.

## Highlights

- Gradient descent, polynomial regression, bias-variance diagnosis, Ridge, Lasso and Elastic Net, and logistic regression, each in its own module with its own tests.
- Hash-chained artifacts. Every stage stores the SHA-256 of the artifact it consumed and of the configuration, so a changed upstream result rebuilds everything downstream.
- Leakage controls. Scalers, expansions, feature selection and class thresholds are fitted on training rows only. The hidden test file is read in one function, used once, and never used for tuning. A leak scanner and tests enforce this.
- Reproducibility. The random seed is derived from the team roster in `config.yaml`, not typed by hand, and the notebook runs top to bottom from a single configuration cell.
- A documented design history. The pipeline was run twice, and the second run changed one design choice, recorded with its evidence in the decision log.

## Results

Validation scores on the seeded 20% hold-out. R² and RMSE are on the original bike-count scale.

| Stage | Model | Validation |
|---|---|---|
| 1 | Linear model on 36 weights, gradient descent at half the stability bound | R² 0.736, RMSE 94.7 |
| 2 | Degree-3 polynomial in `temp`, plus working-day × hour interactions | R² 0.918, RMSE 52.8 |
| 3 | Diagnosis: under-fitting. Target design with 282 weights | R² 0.941 (held-out days 0.946, chronological split 0.938) |
| 4 | Three penalised methods tied within noise. Lasso on the surviving features | R² 0.941, RMSE 44.7 |
| 5 | "Busier than three quarters of comparable hours" label, cut-off 0.25 | Accuracy 0.784, F1 0.659, ROC-AUC 0.885 |

The source of truth is `artifacts/p1.json` to `p6.json`. The notebook and the report read their numbers from those files.

Validation scores are slightly optimistic, because the tuning used the same rows. The held-out-day and chronological estimates are reported beside them.

**Design iteration.** The first pass fitted `log1p(cnt)` and reached R² 0.939 on validation, 0.943 on held-out days and 0.914 chronologically (git tag `v1-submitted`). The second pass changed only the target transform, to a power transform with exponent 0.1 (ADR-017). It mainly improves the estimate for unseen months. The alternatives that were measured and rejected are listed in ADR-018 and in `exp/v2/`.

## Architecture

| Stage | Algorithm | Consumes | Produces |
|---|---|---|---|
| 1 | Gradient descent, written from scratch | the seeded train/validation split | weight vector, learning rate, iteration count |
| 2 | Polynomial regression, fitted with the stage 1 gradient descent | stage 1 weights as the starting point | degree and expanded feature list |
| 3 | Bias-variance diagnosis, plus one chronological split | stage 2 degree and features | diagnosis and target complexity |
| 4 | Ridge, Lasso and Elastic Net (own solvers) | stage 3 target and diagnosis | λ per method, surviving features, a verdict for every input column |
| 5 | Logistic regression (own gradient descent) | stage 4 surviving features | classifier, validation metrics, retrospective |
| 6 | Submission: refit the recommended model and predict the hidden test rows | the stage 4 artifact (the recommended model) | `sample_submission.csv` |

```
train.csv ─► seeded split ─► [1 GD] ─► [2 poly] ─► [3 bias-variance] ─► [4 penalised] ─► [5 logistic]
                                 │          │              │                  │                │
                                 └──────────┴──────────────┴──── artifacts/p1.json … p5.json ──┘
                                                                                                │
                                              test.csv ─► [6 submission, the only module that reads test rows]
```

Each artifact stores the hash of the artifact it consumed. Stage 2 starts from the stage 1 weights, so its first loss must equal the last loss of stage 1, and the code asserts this. `tools/chain_check.py` and assertions in the notebook verify the rest of the chain.

## Deliverables

| File | What it is |
|---|---|
| [rush_hour.ipynb](rush_hour.ipynb) | The executed notebook, built from `nb/`. It contains a configuration cell, then each stage with its expectation, code and outcome cells. It is self-contained: on Google Colab, upload only the notebook, `train.csv` and `test.csv` |
| [report.pdf](report.pdf) | The report (six pages at most). Every number in it is read from the artifacts |
| [sample_submission.csv](sample_submission.csv) | Predictions for `data/test.csv`, one row per test row in the same order. The provided empty template is `data/sample_submission.csv` |
| [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) | A guide to the formulas, change-and-re-run drills and likely questions about the pipeline |

## Quick start

Requirements: Python (built and tested on 3.14) with `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `pyyaml`, `threadpoolctl`, `nbformat`, `nbclient`, `ipykernel`, `markdown-it-py`, `pytest` and, optionally, `ruff`. The PDF build needs Chrome or Edge.

```bash
pip install numpy pandas scikit-learn matplotlib pyyaml threadpoolctl nbformat nbclient ipykernel markdown-it-py pytest ruff
```

`run.py` in the repository root is the task runner and the authoritative entry point. The `Makefile` is kept for reference and may lag behind it.

| Command | What it does | Time |
|---|---|---|
| `python run.py all` | Rebuild `artifacts/p1.json` to `p5.json`. A stage rebuilds only when something upstream changed | Stage 4 takes about 30 minutes; the rest take minutes |
| `python run.py p1` … `p5` | Rebuild one stage | Seconds, except stage 4 |
| `python -m src.bonus_control` | Control experiment for the asymmetric-cost bonus | Seconds |
| `python run.py submission` | Refit the recommended model and write `sample_submission.csv` | Seconds |
| `python run.py nb` | Assemble `nb/p*.py` into `rush_hour.ipynb` and execute it | |
| `python run.py report` | Render the report template and build the PDF | Seconds |
| `python run.py verify-fast` | Lint, tests, leak scan, chain check and traceability counts | Seconds |
| `python run.py verify` | All of the above, plus notebook execution, submission check and report number trace | |

The notebook imports the modules in `src/`. Inside the repository it uses the files there. Elsewhere, such as on Google Colab, its setup cell unpacks its own copy of `src/`, `config.yaml` and `artifacts/`, and asks for `train.csv` and `test.csv` if they are not beside it. Each stage cell loads the stored artifact when its configuration and upstream artifact are unchanged. Set `RECOMPUTE_ALL = True` in the configuration cell to recompute the chain. `python -X utf8 tools/standalone_check.py` re-runs the notebook in an empty folder that holds only the two data files.

### Seed derivation

The seed is computed from the team IDs in [config.yaml](config.yaml), never typed in:

```python
seed = int(sha256("_".join(sorted(team_ids))).hexdigest(), 16) % 100000
```

Changing `team_ids` changes the split and therefore every number. To rebuild, run `python run.py all`, `python -m src.bonus_control`, `python run.py submission`, `python run.py nb` and `python run.py report`.

## Engineering practices

- Every stage is tested against the reference it should match. The gradient is checked numerically, and the solvers are compared with oracle implementations.
- Scalers, imputation values, expansions and class thresholds are fitted on the training rows only, and tests check it.
- Each change is traced to a documented requirement in `docs/TRACE.csv`, and `run.py verify` checks the traceability counts.
- Design choices are recorded as short decision records (`docs/adr/`), each with the numbers behind it.

## Repository layout

```
config.yaml            team IDs and every re-run setting
run.py                 task runner (all, p1 to p5, submission, nb, report, verify-fast, verify)
data/                  train.csv, test.csv, sample_submission.csv (as provided)
src/                   the implementation
  common.py            seed, split, metrics, artifact I/O with hash chaining
  features.py          the feature design (fitted on train, transformed elsewhere)
  gd.py, gd_asym.py    gradient descent, and the asymmetric-cost bonus
  poly.py              lifting the stage 1 weights into the expanded design
  validation.py        the three validators and learning curves
  regularization.py    Ridge, coordinate-descent Lasso, Elastic Net, stability selection
  verdicts.py          the per-column verdict rule
  logistic.py, labels.py   logistic regression, metrics, the high-demand label
  phases/p1.py to p5.py    one module per stage: load upstream artifact, compute, write artifact
  predict.py           the only module that touches the hidden test rows
nb/                    notebook cells as plain percent-format files, one per stage
artifacts/             p1.json to p6.json, the numeric hand-overs between stages
report/                report template with {{artifact.key}} placeholders, figure and PDF builders
tests/                 unit tests: gradient checks, oracle comparisons, leakage and determinism tests
tools/                 checkers: leak_scan, chain_check, submission_check, number_trace, build_nb
docs/                  specification, architecture, data card, decision records, traceability, walkthrough
reports/               verification gates and audits
exp/                   exploratory and diagnostic scripts; nothing here enters the pipeline
```

## Documentation map

| If you want to know | Read |
|---|---|
| What is required and how each requirement is checked | [docs/SPEC.md](docs/SPEC.md), [docs/TRACE.csv](docs/TRACE.csv) |
| Why a choice was made, with the numbers behind it | [docs/adr/](docs/adr/) (18 decision records) |
| What the data looks like | [docs/DATA_CARD.md](docs/DATA_CARD.md) |
| Module interfaces and determinism rules | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| What was verified and what could be challenged | [reports/verify/](reports/verify/), [reports/analysis/](reports/analysis/) |
| Current status and open items | [STATE.md](STATE.md) |

## Known limits

- The trend is a straight line on the transformed scale. It fits well within 2011 and 2012 and runs somewhat too steep beyond that.
- The data contains no zero-demand hours, so the model has never seen one.
- Stage 4 takes about 30 minutes. Changing an early setting is therefore slow to propagate. [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) describes faster demonstrations.
- Validation scores are slightly optimistic, as noted above.
