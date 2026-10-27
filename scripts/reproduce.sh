#!/usr/bin/env bash
# Run experiments and rebuild result tables and plots.
set -euo pipefail

MODE="${1:-check}"
PYTHON="${PYTHON:-}"
DATASET_PATH="${DATASET_PATH:-data/processed/behaviorfl_dataset.pt}"
RAW_DATA_DIR="${RAW_DATA_DIR:-data/raw}"
BOOTSTRAP="${BOOTSTRAP:-0}"
VENV_DIR="${VENV_DIR:-.venv}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements-cpu.txt}"
RESULTS_DIR="${RESULTS_DIR:-results}"
MT_SEEDS="${MT_SEEDS:-7 42 123}"
PARTICIPATION_FRACTION="${PARTICIPATION_FRACTION:-0.02}"
EVAL_EVERY_N_ROUNDS="${EVAL_EVERY_N_ROUNDS:-5}"
export PYTHON DATASET_PATH RAW_DATA_DIR RESULTS_DIR MT_SEEDS PARTICIPATION_FRACTION EVAL_EVERY_N_ROUNDS
ABLATIONS="behaviorfl_mt_no_dropout behaviorfl_mt_zero_fill behaviorfl_mt_mean_fill behaviorfl_mt_no_residual"
METHODS="${METHODS:-destination_specialist state_specialist joint_no_tierdrop joint_tierdrop behaviorfl_mt mas fedhca2}"

cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:$PWD/src"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
  REQUIREMENTS_FILE="requirements.txt"
fi

bootstrap_env() {
  if [ "$BOOTSTRAP" != "1" ]; then
    return
  fi
  if [ -z "$PYTHON" ]; then
    if [ -x "$VENV_DIR/bin/python" ]; then
      PYTHON="$VENV_DIR/bin/python"
    else
      BASE_PYTHON="${BASE_PYTHON:-python3.12}"
      command -v "$BASE_PYTHON" >/dev/null 2>&1 || BASE_PYTHON=python3
      echo "[setup] Creating $VENV_DIR with $($BASE_PYTHON -V 2>&1)"
      "$BASE_PYTHON" -m venv "$VENV_DIR"
      PYTHON="$VENV_DIR/bin/python"
    fi
  fi
  echo "[setup] Installing pinned requirements from $REQUIREMENTS_FILE with $PYTHON"
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install -r "$REQUIREMENTS_FILE"
}

bootstrap_env
if [ -z "$PYTHON" ]; then
  if [ -x "$VENV_DIR/bin/python" ]; then
    PYTHON="$VENV_DIR/bin/python"
  else
    PYTHON=python3
  fi
fi

export PYTHON

preflight() {
  echo "[check] Python: $($PYTHON -V 2>&1)"
  "$PYTHON" - <<'PY'
import flwr, opacus, torch
print(f"PyTorch={torch.__version__} Flower={flwr.__version__} Opacus={opacus.__version__}")
PY
  [ -f "$DATASET_PATH" ] || {
    echo "[check] Missing $DATASET_PATH." >&2
    echo "        Rebuild from raw VED/GeoLife/T-Drive with: scripts/reproduce.sh data-build" >&2
    exit 2
  }
  "$PYTHON" - "$DATASET_PATH" <<'PY'
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
required = {"destination_labels", "state_labels", "state_supervision_mask",
            "client_splits", "source_dataset"}
missing = sorted(required - set(d))
if missing:
    raise SystemExit(f"stale dataset schema; missing {missing}.")
if d.get("metadata", {}).get("schema") != "behaviorfl_multitask_v4":
    raise SystemExit("dataset is not behaviorfl_multitask_v4")
print(f"rows={len(d['trip_features'])} clients={len(d['fl_client_ids'])} "
      f"state_supervised={int(d['state_supervision_mask'].sum())}")
PY
  echo "[check] OK"
}

case "$MODE" in
  check)
    preflight
    ;;
  data-check)
    "$PYTHON" scripts/behavior/assemble_behaviorfl_dataset.py --raw-data-dir "$RAW_DATA_DIR" --check-raw-data
    ;;
  data-build)
    echo "[data-build] Rebuilding dataset from raw VED/GeoLife/T-Drive under $RAW_DATA_DIR ..."
    echo "[data-build] See data/README.md for download instructions and directory layout."
    if [ "$DATASET_PATH" != "data/processed/behaviorfl_dataset.pt" ]; then
      echo "[data-build] Assembly writes data/processed/behaviorfl_dataset.pt; use the default DATASET_PATH." >&2
      exit 2
    fi
    "$PYTHON" scripts/behavior/assemble_behaviorfl_dataset.py --raw-data-dir "$RAW_DATA_DIR" --rebuild-extractors
    preflight
    ;;
  smoke)
    preflight
    mkdir -p results/multitask_smoke
    "$PYTHON" experiments/run_behaviorfl_experiments.py \
      --mode smoke \
      --dataset "$DATASET_PATH" \
      --methods $METHODS \
      --seeds 42 \
      --participation-fraction 1.0 \
      --results-dir results/multitask_smoke
    echo "[smoke] OK - all methods ran end to end. See results/multitask_smoke/."
    ;;
  full)
    preflight
    mkdir -p "$RESULTS_DIR"
    echo "[full] Methods: $METHODS"
    echo "[full] Seeds: $MT_SEEDS  |  Participation fraction: $PARTICIPATION_FRACTION"
    "$PYTHON" experiments/run_behaviorfl_experiments.py \
      --mode full \
      --dataset "$DATASET_PATH" \
      --methods $METHODS \
      --seeds $MT_SEEDS \
      --participation-fraction "$PARTICIPATION_FRACTION" \
      --results-dir "$RESULTS_DIR" \
      --skip-round-evaluation \
      --skip-existing \
      --eval-every-n-rounds "$EVAL_EVERY_N_ROUNDS"
    ;;
  ablations)
    METHODS="$ABLATIONS" "$0" full
    ;;
  sweeps)
    METHODS="behaviorfl_mt_bound05 behaviorfl_mt_bound20 behaviorfl_mt_scale2 behaviorfl_mt_scale6" MT_SEEDS=42 "$0" full
    ;;
  figures)
    "$PYTHON" scripts/verify_workshop_results.py --results-dir "$RESULTS_DIR"
    export BEHAVIORFL_RESULTS_DIR="$RESULTS_DIR"
    for generator in gen_main_results_table.py gen_behaviorfl_mt_results.py gen_results_bars.py gen_supp_tables.py gen_convergence_plots.py; do
      "$PYTHON" "figures/$generator"
    done
    ;;
  verify)
    "$PYTHON" scripts/verify_workshop_results.py --results-dir "$RESULTS_DIR"
    ;;
  checksums)
    "$PYTHON" scripts/check_release_checksums.py
    ;;
  test)
    "$PYTHON" -m pytest -q
    ;;
  oracle)
    # Query state eligibility once before training.
    preflight
    mkdir -p "$RESULTS_DIR"
    "$PYTHON" experiments/run_behaviorfl_experiments.py \
      --mode full \
      --dataset "$DATASET_PATH" \
      --methods oracle_task_aware \
      --seeds $MT_SEEDS \
      --participation-fraction "$PARTICIPATION_FRACTION" \
      --results-dir "$RESULTS_DIR" \
      --skip-round-evaluation \
      --skip-existing \
      --eval-every-n-rounds "$EVAL_EVERY_N_ROUNDS"
    ;;
  all)
    "$0" full
    "$0" oracle
    "$0" ablations
    "$0" sweeps
    "$0" figures
    "$0" verify
    ;;
  *)
    echo "Usage: scripts/reproduce.sh [check|data-check|data-build|smoke|full|oracle|ablations|sweeps|figures|verify|checksums|test|all]" >&2
    echo "       BOOTSTRAP=1 scripts/reproduce.sh check   # create/use .venv first" >&2
    exit 2
    ;;
esac
