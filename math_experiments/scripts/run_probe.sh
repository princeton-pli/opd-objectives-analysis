#!/bin/bash
# Tokens whose probability increased the most during training 
# The teacher's outputs are saved to $OUTPUT_ROOT/probe/teacher_<TEACHER>_outputs.json, 
# and the result of each run to $OUTPUT_ROOT/probe/student_<STUDENT>_teacher_<TEACHER>/<objective>_seed_<seed>.npz

source "$(dirname "$0")/common.sh"

: "${STUDENT:?Set STUDENT to 0.6B or 1.7B}"
: "${TEACHER:?Set TEACHER to 4B, 8B, 14B or 32B}"
OBJECTIVES="${OBJECTIVES:-fkl rkl}"
SEEDS="${SEEDS:-42 43 44}"
STUDENT_MODEL=$(student_model "$STUDENT")
TEACHER_MODEL=$(teacher_model "$TEACHER")

probe_dir="$OUTPUT_ROOT/probe"
teacher_outputs="$probe_dir/teacher_${TEACHER}_outputs.json"
mkdir -p "$probe_dir/student_${STUDENT}_teacher_${TEACHER}"

if [[ ! -f "$teacher_outputs" ]]; then
    echo "[$(date '+%F %T')] teacher outputs | $TEACHER_MODEL | $teacher_outputs"
    python generate_math.py --model "$TEACHER_MODEL" --tp "$(teacher_gpus "$TEACHER")" --out "$teacher_outputs"
fi

for seed in $SEEDS; do
    for objective in $OBJECTIVES; do
        case "$objective" in
            fkl|rkl) ;;
            *) echo "Unknown objective '$objective': use fkl or rkl" >&2; exit 2 ;;
        esac
        run_dir="$OUTPUT_ROOT/student_${STUDENT}_teacher_${TEACHER}/${objective}_seed_${seed}"
        output_file="$probe_dir/student_${STUDENT}_teacher_${TEACHER}/${objective}_seed_${seed}.npz"
        echo "[$(date '+%F %T')] probe | $run_dir | $output_file"
        python probe_top_increase.py --teacher_outputs "$teacher_outputs" --initial_student "$STUDENT_MODEL" \
            --trained_student "$run_dir" --output_file "$output_file"
    done
done
