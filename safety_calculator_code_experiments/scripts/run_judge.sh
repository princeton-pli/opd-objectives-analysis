#!/bin/bash
# Classifies the saved outputs of the safety experiments as refusals or answers with an LLM judge.
#
#   export OPENAI_API_KEY=...
#   EXPERIMENT=safety_qwen3 bash scripts/run_judge.sh     # the initial evaluation and every OPD run of the experiment
#   bash scripts/run_judge.sh <run directory> ...         # specific directories
#
# Results are written to <run directory>/judge/.

source "$(dirname "$0")/common.sh"

WORKERS="${WORKERS:-4}"
: "${OPENAI_API_KEY:?Export OPENAI_API_KEY before running the judge}"

if (( $# > 0 )); then
    run_dirs=("$@")
else
    load_experiment
    shopt -s nullglob
    run_dirs=("$RUN_ROOT"/initial_eval_* "$RUN_ROOT"/*_seed_*)
    shopt -u nullglob
    if (( ${#run_dirs[@]} == 0 )); then
        echo "No runs found under $RUN_ROOT" >&2
        exit 2
    fi
fi

for run_dir in "${run_dirs[@]}"; do
    echo "[$(date '+%F %T')] judge | $run_dir"
    python -m opd.judges --run-dir "$run_dir" --model gpt-5.6-luna --reasoning-effort medium --workers "$WORKERS" \
        2>&1 | tee -a "$run_dir/judge.log"
done
