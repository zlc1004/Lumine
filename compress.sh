#!/bin/bash

# --- Configuration & Safety ---
set -e          # Exit immediately if a command fails
set -o pipefail # Ensure pipe errors are captured
IFS=$'\n\t'     # Better string handling

CHECKPOINT_PATH="${1:-}"
MODEL_ASSETS_PATH="${2:-}"
OUTPUT_DIR="/workspace/Lumine/output/hf_ckpt"

# Helper function for pretty logging
log() {
    echo -e "\033[1;34m[$(date '+%H:%M:%S')]\033[0m $1"
}

# Check if arguments are provided
if [[ -z "$CHECKPOINT_PATH" || -z "$MODEL_ASSETS_PATH" ]]; then
    echo "Usage: $0 <CHECKPOINT_PATH> <MODEL_ASSETS_PATH>"
    exit 1
fi

# --- 1. Environment Setup ---
log "Activating virtual environment..."
source /workspace/Lumine/VeOmni/.venv/bin/activate

# --- 2. Model Merging ---
log "Merging DCP checkpoints to HF format..."
log "Input: $CHECKPOINT_PATH"
log "Output: $OUTPUT_DIR"

cd /workspace/Lumine/VeOmni
python /workspace/Lumine/VeOmni/scripts/merge_dcp_to_hf.py \
    --load-dir "$CHECKPOINT_PATH" \
    --save-dir "$OUTPUT_DIR" \
    --model-assets-dir "$MODEL_ASSETS_PATH" \
    --shard-size 5000000000  # 5GB shards are better for your 500GB RAM

# --- 3. Assets Transfer ---
log "Injecting model assets (configs/tokenizers)..."
cd /workspace/Lumine

# List of required assets
ASSETS=(
    "config.json"
    "preprocessor_config.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "video_preprocessor_config.json"
    "vocab.json"
    "added_tokens.json"
)

for asset in "${ASSETS[@]}"; do
    if [[ -f "$MODEL_ASSETS_PATH/$asset" ]]; then
        cp "$MODEL_ASSETS_PATH/$asset" "$OUTPUT_DIR/"
        echo "  [✓] Copied $asset"
    else
        echo -e "  \033[1;31m[!] Warning: $asset not found in $MODEL_ASSETS_PATH\033[0m"
    fi
done

# --- 4. Compression ---
log "Starting compression (tar + pigz)..."
SIZE=$(du -sb "$OUTPUT_DIR" | awk '{print $1}')

tar -cf - -C /workspace/Lumine/output hf_ckpt \
 | pv -p -t -e -r -s "$SIZE" \
 | pigz > hf_ckpt.tar.gz

log "Process Complete! File saved as: hf_ckpt.tar.gz"