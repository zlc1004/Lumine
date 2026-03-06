#!/usr/bin/bash

# Check if an argument was provided
if [ -z "$1" ]; then
    echo "Usage: ./serve.sh [vllm|sglang|tgi]"
    exit 1
fi

SERVER=$1
MODEL_DIR="./models/UI-TARS-1.5-7B"

# Activate the environment
source .venv/bin/activate

# Download logic
if [ -d "$MODEL_DIR" ]; then
    echo "UI-TARS-1.5-7B already downloaded"
else
    echo "Downloading UI-TARS-1.5-7B"
    # Ensure hf-transfer is installed for faster downloads on EC2
    export HF_HUB_ENABLE_HF_TRANSFER=1
    ~/.local/bin/hf download ByteDance-Seed/UI-TARS-1.5-7B --local-dir $MODEL_DIR
fi

case $SERVER in
  vllm)
    echo "Launching vLLM server with proxy on port 8000..."
    # vLLM configuration for UI-TARS-1.5-7B (based on Qwen2.5-VL-7B):
    # - max-model-len: 131072 (Qwen2.5 supports up to 128k tokens)
    # - max-num-seqs: 128 (reduced for larger context window)
    # - disable-custom-all-reduce: for stability with VLMs
    # - enforce-eager: equivalent to CUDA_GRAPHS=0 for TGI
    # Run vLLM on port 9000, proxy on 8000 to handle max_tokens adjustment
    echo "Starting vLLM backend on port 9000..."
    # Allow using context length beyond model's max_position_embeddings
    # UI-TARS-1.5-7B's Qwen2.5-VL base supports 128k RoPE scaling
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL_DIR \
        --served-model-name UI-TARS-1.5-7B \
        --dtype bfloat16 \
        --port 9000 \
        --host 127.0.0.1 \
        --trust-remote-code \
        --max-model-len 131072 \
        --max-num-seqs 128 \
        --disable-custom-all-reduce \
        --enforce-eager &
    
    VLLM_PID=$!
    echo "vLLM backend started (PID: $VLLM_PID)"
    
    # Wait for vLLM to start
    echo "Waiting for vLLM backend to be ready..."
    for i in {1..60}; do
        if curl -s http://127.0.0.1:9000/health > /dev/null 2>&1; then
            echo "vLLM backend is ready!"
            break
        fi
        sleep 2
    done
    
    # Start proxy
    echo "Starting proxy server on port 8000..."
    python3 ./vllm_proxy.py --port 8000 --backend-port 9000 --max-context 131072
    ;;

  sglang)
    echo "Launching SGLang server on port 8000..."
    # SGLang configuration for UI-TARS-1.5-7B (Qwen2.5-VL 128k context)
    python3 -m sglang.launch_server \
        --model-path $MODEL_DIR \
        --port 8000 \
        --host 0.0.0.0 \
        --dtype bfloat16 \
        --context-length 131072 \
        --chat-template llama_3_vision \
        --trust-remote-code
    ;;

  tgi)
    echo "Launching TGI server via Docker on port 8000..."
    # Get the absolute path to the model directory
    MODEL_ABSOLUTE_PATH=$(realpath $MODEL_DIR)
    
    # Note: TGI requires CUDA_GRAPHS=0 for VLMs to avoid illegal instruction errors
    # PAYLOAD_LIMIT=8000000 prevents request failures due to large images
    # See: https://github.com/huggingface/text-generation-inference/issues/2875
    # See: https://github.com/huggingface/text-generation-inference/issues/1802
    docker run --gpus all --shm-size 1g \
        -e CUDA_GRAPHS=0 \
        -e PAYLOAD_LIMIT=8000000 \
        -p 8000:80 \
        -v $MODEL_ABSOLUTE_PATH:/data \
        ghcr.io/huggingface/text-generation-inference:3.2.1 \
        --model-id /data \
        --dtype bfloat16 \
        --max-input-length 131072 \
        --max-batch-prefill-tokens 131072 \
        --max-total-tokens 131073
    ;;

  *)
    echo "Error: Invalid server type '$SERVER'. Use vllm, sglang, or tgi."
    exit 1
    ;;
esac