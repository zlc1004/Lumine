#!/bin/bash
# Lumine Training - 2x H200 SXM (282GB VRAM)
# Usage: ./train_h200x2.sh [stage]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VEOMNI_DIR="$SCRIPT_DIR/VeOmni"

# Activate virtual environment if exists
if [ -d "$VEOMNI_DIR/.venv" ]; then
    source "$VEOMNI_DIR/.venv/bin/activate"
fi

# Function to run a stage
run_stage() {
    local stage=$1
    local config=$2
    echo "=========================================="
    echo "Running Stage $stage on 2x H200 SXM"
    echo "Config: $config"
    echo "=========================================="
    
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    torchrun --nproc_per_node=2 -m tasks.omni.train_qwen_vl "$config"
    
    if [ $? -eq 0 ]; then
        echo "Stage $stage completed successfully!"
    else
        echo "Stage $stage failed!"
        exit 1
    fi
}

# Parse arguments
STAGE=$1

if [ -z "$STAGE" ] || [ "$STAGE" == "all" ]; then
    echo "Running all 3 training stages on 2x H200 SXM..."
    
    run_stage 1 "$SCRIPT_DIR/configs/stage1_pretrain.yaml"
    run_stage 2 "$SCRIPT_DIR/configs/stage2_instruct.yaml"
    run_stage 3 "$SCRIPT_DIR/configs/stage3_reasoning.yaml"
    
    echo "=========================================="
    echo "All training stages completed!"
    echo "Final model: output/lumine_stage3_reasoning/hf_ckpt"
    echo "=========================================="
    
elif [ "$STAGE" == "1" ]; then
    run_stage 1 "$SCRIPT_DIR/configs/stage1_pretrain.yaml"
elif [ "$STAGE" == "2" ]; then
    run_stage 2 "$SCRIPT_DIR/configs/stage2_instruct.yaml"
elif [ "$STAGE" == "3" ]; then
    run_stage 3 "$SCRIPT_DIR/configs/stage3_reasoning.yaml"
else
    echo "Usage: $0 [stage]"
    echo "  (empty or 'all') - Run all stages"
    echo "  1 - Run only stage 1 (pre-training)"
    echo "  2 - Run only stage 2 (instruction-following)"
    echo "  3 - Run only stage 3 (reasoning)"
    exit 1
fi
