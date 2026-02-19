#!/bin/bash
# Test script for edited data
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VEOMNI_DIR="$SCRIPT_DIR/VeOmni"

# Activate virtual environment if exists
if [ -d "$VEOMNI_DIR/.venv" ]; then
    source "$VEOMNI_DIR/.venv/bin/activate"
fi

echo "Running test with edited data in input_log_20260218_113251_edited..."
python -m tasks.omni.train_qwen_vl --config "$SCRIPT_DIR/configs/test_edited.yaml"
