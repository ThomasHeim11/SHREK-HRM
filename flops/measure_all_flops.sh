#!/bin/bash -l
#SBATCH --job-name=flops
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --propagate=NONE
#SBATCH --output=/home/thheim/HMR/logs/flops_%j.log
#SBATCH --error=/home/thheim/HMR/logs/flops_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

FLOPS=~/HMR/flops/flops.py
RESULTS=~/HMR/flops/results/seed${SEED:?}
HRM_DIR=~/HMR/source/HRM\(Original\)/HRM-main
SHREK_DIR=~/HMR/source/SHREK-HRM
TRM_DIR=~/HMR/source/TinyRecursiveModels

mkdir -p $RESULTS

# ── Sudoku ──

echo "=== Original HRM (Sudoku) ==="
DISABLE_COMPILE=1 python3 $FLOPS measure \
    --checkpoint "$HRM_DIR/checkpoints/Sudoku-extreme-1k-aug-1000 ACT-torch/HRM_Sudoku_seed${SEED}" \
    --model-dir "$HRM_DIR" \
    --name "Original HRM" --task sudoku --num-samples 1000 --results-dir $RESULTS

echo "=== SHREK Large (Sudoku) ==="
DISABLE_COMPILE=1 python3 $FLOPS measure \
    --checkpoint "$SHREK_DIR/checkpoints/SHREK_Ablation_Sudoku/Ablation_Full_seed${SEED}" \
    --model-dir "$SHREK_DIR" \
    --name "SHREK Large" --task sudoku --num-samples 1000 --results-dir $RESULTS

echo "=== SHREK Small (Sudoku) ==="
DISABLE_COMPILE=1 python3 $FLOPS measure \
    --checkpoint "$SHREK_DIR/checkpoints/SHREK_Ablation_Sudoku/SHREK_Tiny_Vanilla_Sudoku_seed${SEED}" \
    --model-dir "$SHREK_DIR" \
    --name "SHREK Small" --task sudoku --num-samples 1000 --results-dir $RESULTS

echo "=== TRM Attention (Sudoku) ==="
DISABLE_COMPILE=1 python3 $FLOPS measure \
    --checkpoint "$TRM_DIR/checkpoints/HRM_Sudoku_Comparison/TRM_ATT_Sudoku_seed${SEED}" \
    --model-dir "$TRM_DIR" \
    --name "TRM Attention" --task sudoku --num-samples 1000 --results-dir $RESULTS

# ── Maze ──

echo "=== Original HRM (Maze) ==="
DISABLE_COMPILE=1 python3 $FLOPS measure \
    --checkpoint "$HRM_DIR/checkpoints/HRM_Maze_Comparison/HRM_Maze_seed${SEED}" \
    --model-dir "$HRM_DIR" \
    --name "Original HRM" --task maze --num-samples 1000 --results-dir $RESULTS

echo "=== SHREK Large (Maze) ==="
DISABLE_COMPILE=1 python3 $FLOPS measure \
    --checkpoint "$SHREK_DIR/checkpoints/HRM_Maze_Comparison/SHREK_Large_Maze_seed${SEED}" \
    --model-dir "$SHREK_DIR" \
    --name "SHREK Large" --task maze --num-samples 1000 --results-dir $RESULTS

echo "=== SHREK Small (Maze) ==="
DISABLE_COMPILE=1 python3 $FLOPS measure \
    --checkpoint "$SHREK_DIR/checkpoints/HRM_Maze_Comparison/SHREK_Tiny_Maze_seed${SEED}" \
    --model-dir "$SHREK_DIR" \
    --name "SHREK Small" --task maze --num-samples 1000 --results-dir $RESULTS

echo "=== TRM Attention (Maze) ==="
DISABLE_COMPILE=1 python3 $FLOPS measure \
    --checkpoint "$TRM_DIR/checkpoints/HRM_Maze_Comparison/TRM_ATT_Maze_seed${SEED}" \
    --model-dir "$TRM_DIR" \
    --name "TRM Attention" --task maze --num-samples 1000 --results-dir $RESULTS

echo ""
echo "Done! To generate charts locally:"
echo "  python3 flops/flops.py plot --results-dir flops/results"
