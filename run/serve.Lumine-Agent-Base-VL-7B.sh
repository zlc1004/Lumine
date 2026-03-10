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
    echo "Launching SGLang server with proxy..."
    # Run SGLang on port 30000, proxy on 8000
    echo "Starting SGLang backend on port 30000..."
    SGLANG_DISABLE_CUDNN_CHECK=1 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python3 -m sglang.launch_server \
        --model-path $MODEL_DIR \
        --port 30000 \
        --host 127.0.0.1 \
        --dtype bfloat16 \
        --trust-remote-code \
        --chat-template llama_3_vision &
    
    SGLANG_PID=$!
    echo "SGLang backend started (PID: $SGLANG_PID)"
    
    # Wait for SGLang to start
    echo "Waiting for SGLang backend to be ready..."
    for i in {1..120}; do
        if curl -s http://127.0.0.1:30000/health > /dev/null 2>&1; then
            echo "SGLang backend is ready!"
            break
        fi
        sleep 2
    done
    
    # Start proxy (default 65k context for 7B model)
    echo "Starting proxy server on port 8000..."
    python3 ./sglang_proxy.py --port 8000 --backend-port 30000 --max-context 65536
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