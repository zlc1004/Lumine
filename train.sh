#!/bin/bash
# Lumine Training Script - Run stages sequentially with configurable GPUs
# Usage: ./train.sh [stage] [--gpu <number>]
#   ./train.sh all --gpu 2    - Run all stages on 2 GPUs
#   ./train.sh 0 --gpu 8      - Run stage 0 (knowledge) on 8 GPUs
#   ./train.sh 1 --gpu 1      - Run only stage 1 on 1 GPU

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Path to VeOmni
VEOMNI_DIR="$SCRIPT_DIR/VeOmni"

# Activate virtual environment if exists
if [ -d "$VEOMNI_DIR/.venv" ]; then
    source "$VEOMNI_DIR/.venv/bin/activate"
fi

# Default values
STAGE="all"
GPU_COUNT=1

# Simple argument parsing
while [[ $# -gt 0 ]]; do
  case $1 in
    0|1|2|3|all)
      STAGE="$1"
      shift
      ;;
    --gpu)
      GPU_COUNT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Function to run a stage
run_stage() {
    local stage=$1
    local config=$2
    local log_file="$SCRIPT_DIR/stage${stage}_training.log"
    
    echo "=========================================="
    echo "Running Stage $stage on $GPU_COUNT GPU(s)"
    echo "Config: $config"
    echo "Log file: $log_file"
    echo "=========================================="
    
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TOKENIZERS_PARALLELISM=false
    torchrun --nproc_per_node="$GPU_COUNT" -m tasks.omni.train_qwen_vl "$config" 2>&1 | tee "$log_file"
    
    if [ $? -eq 0 ]; then
        echo "Stage $stage completed successfully!"
    else
        echo "Stage $stage failed!"
        exit 1
    fi
}

if [ "$STAGE" == "all" ]; then
    echo "Running all training stages (0, 1, 2, 3)..."
    run_stage 0 "$SCRIPT_DIR/configs/stage0_knowledge.yaml"
    run_stage 1 "$SCRIPT_DIR/configs/stage1_pretrain.yaml"
    run_stage 2 "$SCRIPT_DIR/configs/stage2_instruct.yaml"
    run_stage 3 "$SCRIPT_DIR/configs/stage3_reasoning.yaml"
    echo "=========================================="
    echo "All training stages completed!"
    echo "=========================================="
elif [ "$STAGE" == "0" ]; then
    run_stage 0 "$SCRIPT_DIR/configs/stage0_knowledge.yaml"
elif [ "$STAGE" == "1" ]; then
    run_stage 1 "$SCRIPT_DIR/configs/stage1_pretrain.yaml"
elif [ "$STAGE" == "2" ]; then
    run_stage 2 "$SCRIPT_DIR/configs/stage2_instruct.yaml"
elif [ "$STAGE" == "3" ]; then
    run_stage 3 "$SCRIPT_DIR/configs/stage3_reasoning.yaml"
fi
