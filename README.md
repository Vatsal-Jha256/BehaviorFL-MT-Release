# BehaviorFL-MT

Code for *Federated Multi-Task Learning under Unequal Vehicle Sensors and
Partial Supervision*, by Vatsal Jha and Ashok Bragadeshwaran.

The model predicts destination zones and a powertrain-state proxy from early
trip features. Clients have different sensors and labels. A fixed-size sampler
reserves slots for clients discovered to have state labels, while sensor-block
dropout trains the state head with OBD hidden.

This repo includes the processed dataset, 40 saved runs and their checkpoints.
Only VED supplies state labels; the target is a telemetry composite rather than
a fault diagnosis. MAS and FedHCA² are adaptations to the shared two-head model.

## Run it

Use Python 3.12 on Linux:

```bash
BOOTSTRAP=1 scripts/reproduce.sh check
scripts/reproduce.sh checksums
scripts/reproduce.sh test
scripts/reproduce.sh smoke
scripts/reproduce.sh figures
```

Bootstrap installs the pinned CPU dependencies into `.venv`. The wrapper then
uses that environment automatically. Set `PYTHON=/path/to/python` to use another
one. The processed dataset is included, so no raw-data download is needed.

`smoke` runs seven methods for three rounds and writes to
`results/multitask_smoke/`. `figures` rebuilds the result tables and plots from
the saved JSONs. `verify` checks all 40 runs against
[release_manifest.json](release_manifest.json); `checksums` checks the original
artifact bytes.

## Reproduce the experiments

```bash
scripts/reproduce.sh full        # seven methods, three seeds
scripts/reproduce.sh oracle      # eligibility known before round 1
scripts/reproduce.sh ablations   # four component ablations, three seeds
scripts/reproduce.sh sweeps      # residual-bound and scale sweeps, seed 42
# Run all batches, then rebuild and check the outputs:
scripts/reproduce.sh all
```

Completed runs are reused when their settings match. To train everything again
in a separate directory:

```bash
RESULTS_DIR=results/rerun scripts/reproduce.sh all
```

Runs use seeds 7, 42 and 123, 60 rounds, two local epochs, batch size 32, and
Adam (learning rate 0.001, weight decay 0.0001). The standard cohort has 112 of
5,617 clients per round. The state specialist uses the 188 state-eligible
clients and has a different exposure budget. Full runs use 16 Ray CPU slots;
allow several hours on CPU.

Custom runs can set `METHODS`, `MT_SEEDS`, `PARTICIPATION_FRACTION`,
`EVAL_EVERY_N_ROUNDS`, `DATASET_PATH`, `RAW_DATA_DIR`, and `RESULTS_DIR`. The bundled figure
scripts and manifest use the reported three-seed protocol. A different
protocol needs its own analysis. Fresh training can change checkpoint hashes
and numerical results.

| Method name | Configuration |
|---|---|
| `destination_specialist` | Destination only |
| `state_specialist` | State only |
| `joint_no_tierdrop` | FedAvg (Joint) |
| `joint_tierdrop` | Uniform sampling with sensor-block dropout |
| `behaviorfl_mt` | Full method |
| `mas`, `fedhca2` | Adapted baselines |
| `oracle_task_aware` | Eligibility known from round 1 |
| `behaviorfl_mt_no_dropout` | No sensor-block dropout |
| `behaviorfl_mt_zero_fill` | Zero-fill |
| `behaviorfl_mt_mean_fill` | Batch mean-fill |
| `behaviorfl_mt_no_residual` | No destination residual |

`tierdrop` in configuration names means sensor-block dropout. Batch mean-fill
depends on the inference batch, so its periodic and final evaluation scores can
differ. Tables use the final checkpoint evaluation.

## Docker

```bash
docker compose build
docker compose run --rm behaviorfl-mt check
docker compose run --rm behaviorfl-mt smoke
docker compose run --rm behaviorfl-mt all
```

Data, results and figures are mounted from the host. Compose defaults to UID/GID
1000; set `LOCAL_UID` and `LOCAL_GID` if your host IDs differ.

## Data and code

The dataset has 102,843 trips from VED, GeoLife and T-Drive, split by vehicle or
user into 5,617 training and 2,871 held-out clients. See
[data/README.md](data/README.md) for original download links, dataset citations,
and preprocessing instructions. Raw inputs live under `data/raw/ved`,
`data/raw/geolife`, and `data/raw/tdrive`.

```bash
scripts/reproduce.sh data-check
scripts/reproduce.sh data-build
```

The rebuild replaces the bundled feature tensor; the quickstart uses the
included tensor directly.

Models and Flower clients are in `src/`. The method registry and training loop
are in `experiments/run_behaviorfl_experiments.py`; table and plot generators
are in `figures/`. Dependencies are pinned in `requirements-cpu.txt` (CPU) and
`requirements.txt` (default PyTorch wheel).

Citation metadata is in [CITATION.cff](CITATION.cff).
