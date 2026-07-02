#!/usr/bin/env bash
# run_all.sh — full pipeline for one stage (3tev|10tev).
# Each step uses the matching config; outputs land under data/<stage>, models/<stage>,
# analysis/<stage>, dll/<stage>.  Re-running skips finished steps unless flagged.
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
STAGE=${1:?usage: bash run_all.sh {3tev|10tev}  [--retune-spanet] [--no-retune-spanet] [--retune-ml] [--skip-extract] [--force-extract]}
shift || true

# CRITICAL: lib/physics_constants.py reads PIPELINE_STAGE at import time and
# defaults to '3tev' silently if unset.  Without this export, a `bash run_all.sh
# 10tev` run would use 3 TeV LUMI (1000 fb⁻¹) and 3 TeV xsec values on 10 TeV
# data → wildly wrong DLL/w68 with no error message.  Fixed 2026-05-28.
case "$STAGE" in
  3tev|10tev) export PIPELINE_STAGE="$STAGE" ;;
  *) echo "STAGE must be '3tev' or '10tev', got '$STAGE'"; exit 1 ;;
esac

RETUNE_SPANET=1   # default ON for SPANet (6-feature, no prior best.json)
RETUNE_ML=0       # default OFF for ML1/ML2 (require models/<stage>/ml{1,2}_best.json
                  # from a prior 04a/05a run, or pass --retune-ml to generate one)
SKIP_EXTRACT=0
FORCE_EXTRACT=0
for arg in "$@"; do
  case "$arg" in
    --retune-spanet) RETUNE_SPANET=1 ;;
    --no-retune-spanet) RETUNE_SPANET=0 ;;
    --retune-ml) RETUNE_ML=1 ;;
    --skip-extract) SKIP_EXTRACT=1 ;;
    --force-extract) FORCE_EXTRACT=1 ;;
    *) echo "unknown flag: $arg"; exit 1 ;;
  esac
done

CFG="$HERE/config/${STAGE}.yaml"
[[ -f "$CFG" ]] || { echo "config not found: $CFG"; exit 1; }
mkdir -p "$HERE/logs"

ts() { date '+%H:%M:%S'; }
say() { echo "[$(ts)] >>> $*"; }
HHML_CONDA_LIB=${HHML_CONDA_LIB:-/path/to/your/conda/envs/<env>/lib}
PY() { LD_LIBRARY_PATH=$HHML_CONDA_LIB:$LD_LIBRARY_PATH python3 "$@"; }

# 1) extract
if (( ! SKIP_EXTRACT )); then
  say "01  extract features"
  EXTRA_FLAG=""
  (( FORCE_EXTRACT )) && EXTRA_FLAG="--force"
  PY "$HERE/src/01_extract_features.py" --config "$CFG" $EXTRA_FLAG 2>&1 | tee "$HERE/logs/${STAGE}_01_extract.log"
fi

# 2a/2) SPANet — tune (default ON for new pipeline) then train
if (( RETUNE_SPANET )); then
  say "02a  SPANet Optuna search"
  PY "$HERE/src/02a_tune_spanet.py" --config "$CFG" 2>&1 | tee "$HERE/logs/${STAGE}_02a_tune_spanet.log"
fi
say "02   SPANet train (uses spanet_best.json)"
PY "$HERE/src/02_train_spanet.py" --config "$CFG" 2>&1 | tee "$HERE/logs/${STAGE}_02_train_spanet.log"

# 3) pairing precompute
say "03   SPANet pairing precompute"
PY "$HERE/src/03_precompute_pairing.py" --config "$CFG" 2>&1 | tee "$HERE/logs/${STAGE}_03_pairing.log"

# 4a/4) ML1 — Optuna optional, then train
if (( RETUNE_ML )); then
  say "04a  ML1 Optuna search"
  PY "$HERE/src/04a_tune_ml1.py" --config "$CFG" 2>&1 | tee "$HERE/logs/${STAGE}_04a_tune_ml1.log"
fi
say "04   ML1 train (uses optuna.ml1.from_best in config, or local best.json)"
PY "$HERE/src/04_train_ml1.py" --config "$CFG" 2>&1 | tee "$HERE/logs/${STAGE}_04_ml1.log"

# 5a/5) ML2 — Optuna optional, then train
if (( RETUNE_ML )); then
  say "05a  ML2 Optuna search"
  PY "$HERE/src/05a_tune_ml2.py" --config "$CFG" 2>&1 | tee "$HERE/logs/${STAGE}_05a_tune_ml2.log"
fi
say "05   ML2 train (uses optuna.ml2.from_best in config, or local best.json)"
PY "$HERE/src/05_train_ml2.py" --config "$CFG" 2>&1 | tee "$HERE/logs/${STAGE}_05_ml2.log"

# 6) ML analysis (history + FI + correlation)
say "06   ML analysis (history, FI, correlation)"
PY "$HERE/src/06_ml_analysis.py" --config "$CFG" 2>&1 | tee "$HERE/logs/${STAGE}_06_analysis.log"

# 7) DLL morphing
say "07   DLL morphing + per-κ table"
PY "$HERE/src/07_dll_morphing.py" --config "$CFG" 2>&1 | tee "$HERE/logs/${STAGE}_07_dll.log"

# 8) DLL plots
say "08   DLL plots"
PY "$HERE/src/08_dll_plots.py" --config "$CFG" 2>&1 | tee "$HERE/logs/${STAGE}_08_plots.log"

say "ALL DONE for stage=${STAGE}"
