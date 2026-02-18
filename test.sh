#!/bin/bash
# Quick test script for local testing
# Usage: ./test.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VEOMNI_DIR="$SCRIPT_DIR/VeOmni"

# Activate virtual environment if exists
if [ -d "$VEOMNI_DIR/.venv" ]; then
    source "$VEOMNI_DIR/.venv/bin/activate"
fi

echo "Running test with Qwen2-VL-2B..."
echo "Config: configs/test_2b.yaml"
echo ""

python -m tasks.omni.train_qwen_vl --config "$SCRIPT_DIR/configs/test_2b.yaml"
