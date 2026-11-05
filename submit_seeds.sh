#!/bin/bash
# Submits seeds 1 and 2 for every configuration in the paper (seed 0 already exists).
# Usage (from the repository root, on the cluster):
#     module load slurm/21.08.8
#     bash submit_seeds.sh
#
# Phase 1 is the full model comparison. Phase 2 is the rest of the Sudoku ablation
# and waits until every Phase 1 job has ended (SHREK-Large in Phase 1 is also the
# ablation's full-model row).

set -e

mkdir -p ~/HMR/logs
cd ~/HMR

SEEDS="1 2"

SHREK=source/SHREK-HRM/script/train
ABL=$SHREK/AblationStudy
HRM="source/HRM(Original)/HRM-main/script/train"
TRM=source/TinyRecursiveModels/script/train

PHASE1=(
    "$HRM/train_hrm_sudoku_1gpu.sh"
    "$ABL/train_shrek_tiny_sudoku_vanilla.sh"
    "$ABL/train_shrek_large_sudoku_vanilla_full.sh"
    "$TRM/train_trm_att_sudoku.sh"
    "$HRM/train_hrm_maze_1gpu.sh"
    "$SHREK/train_shrek_tiny_maze.sh"
    "$SHREK/train_shrek_large_maze.sh"
    "$TRM/train_trm_att_maze.sh"
)
PHASE2=(
    "$ABL/train_shrek_large_sudoku_vanilla_no_both.sh"
    "$ABL/train_shrek_large_sudoku_vanilla_no_error.sh"
    "$ABL/train_shrek_large_sudoku_vanilla_no_stagnation.sh"
)

ids=""
for seed in $SEEDS; do
    for script in "${PHASE1[@]}"; do
        id=$(sbatch --parsable --export=ALL,SEED=$seed "$script")
        ids="$ids:$id"
        echo "phase1 seed=$seed job=$id $script"
    done
done

for seed in $SEEDS; do
    for script in "${PHASE2[@]}"; do
        id=$(sbatch --parsable --dependency=afterany$ids --export=ALL,SEED=$seed "$script")
        echo "phase2 seed=$seed job=$id $script"
    done
done
