#!/bin/bash
# Test script for edited data using VeOmni's train.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VEOMNI_DIR="$SCRIPT_DIR/VeOmni"

# Activate virtual environment if exists
if [ -d "$VEOMNI_DIR/.venv" ]; then
    source "$VEOMNI_DIR/.venv/bin/activate"
fi

echo "Running test with edited data..."

# Run from the main Lumine directory so relative paths work
cd "$SCRIPT_DIR"
bash "$VEOMNI_DIR/train.sh" -m tasks.omni.train_qwen_vl "$SCRIPT_DIR/configs/test_edited.yaml"
