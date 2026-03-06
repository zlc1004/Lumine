#!/usr/bin/bash

# Usage check
if [ -z "$1" ]; then
    echo "Usage: ./serve.sh [vllm|sglang|tgi]"
    exit 1
fi

SERVER=$1
MODEL_NAME="Lumine-Agent-Base-VL-7B"
MODEL_DIR="./models/$MODEL_NAME"
REPO_ID="koboshchan/Lumine-Agent-Base-VL-7B"

# Activate the environment
source .venv/bin/activate

# Download logic
if [ -d "$MODEL_DIR" ]; then
    echo "$MODEL_NAME already downloaded"
else
    echo "Downloading $MODEL_NAME..."
    export HF_HUB_ENABLE_HF_TRANSFER=1
    ~/.local/bin/hf download $REPO_ID --local-dir $MODEL_DIR
fi

case $SERVER in
  vllm)
    echo "Launching vLLM server on port 8000..."
    # vLLM configuration with VLM optimizations:
    # - max-model-len: 16384 (suitable for Lumine Agent Base)
    # - disable-custom-all-reduce: for stability with VLMs
    # - enforce-eager: equivalent to CUDA_GRAPHS=0
    python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL_DIR \
        --served-model-name lumine-agent-vl-7b \
        --dtype bfloat16 \
        --port 8000 \
        --host 0.0.0.0 \
        --trust-remote-code \
        --max-model-len 16384 \
        --disable-custom-all-reduce \
        --enforce-eager
    ;;

  sglang)
    echo "Launching SGLang server on port 8000..."
    # Note: Using the Llama 3 Vision template as it's the closest fit for UI-TARS/Lumine
    python3 -m sglang.launch_server \
        --model-path $MODEL_DIR \
        --port 8000 \
        --host 0.0.0.0 \
        --dtype bfloat16 \
        --trust-remote-code \
        --chat-template llama_3_vision
    ;;

  tgi)
    echo "Launching TGI server via Docker on port 8000..."
    # Get the absolute path to the model directory
    MODEL_ABSOLUTE_PATH=$(realpath $MODEL_DIR)
    
    # TGI requires CUDA_GRAPHS=0 to prevent VRAM spikes with vision patches
    # PAYLOAD_LIMIT=8000000 prevents request failures due to large images
    docker run --gpus all --shm-size 1g \
        -e CUDA_GRAPHS=0 \
        -e PAYLOAD_LIMIT=8000000 \
        -p 8000:80 \
        -v $MODEL_ABSOLUTE_PATH:/data \
        ghcr.io/huggingface/text-generation-inference:3.2.1 \
        --model-id /data \
        --dtype bfloat16 \
        --max-input-length 65536 \
        --max-batch-prefill-tokens 65536 \
        --max-total-tokens 65537
    ;;

  *)
    echo "Error: Invalid server type '$SERVER'. Use vllm, sglang, or tgi."
    exit 1
    ;;
esac