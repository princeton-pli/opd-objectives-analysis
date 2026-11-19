#!/bin/bash
#   bash scripts/run_initial_eval.sh                  # all students and teachers
#   MODELS="0.6B 32B" bash scripts/run_initial_eval.sh
#
# Each model writes to $OUTPUT_ROOT/initial_eval/<student|teacher>_<size>.

source "$(dirname "$0")/common.sh"

MODELS="${MODELS:-0.6B 1.7B 4B 8B 14B 32B}"

for size in $MODELS; do
    case "$size" in
        0.6B|1.7B)
            model=$(student_model "$size")
            output_dir="$OUTPUT_ROOT/initial_eval/student_$size"
            check_output_dir "$output_dir"
            evaluate "$model" math500_inst "$output_dir" 1
            evaluate "$model" mathtrain500_inst "$output_dir" 1
            ;;
        *)
            model=$(teacher_model "$size")
            output_dir="$OUTPUT_ROOT/initial_eval/teacher_$size"
            check_output_dir "$output_dir"
            evaluate "$model" math500_inst "$output_dir" "$(teacher_gpus "$size")"
            ;;
    esac
done
