#!/usr/bin/bash

# Check if an argument was provided
if [ -z "$1" ]; then
    echo "Usage: ./serve.UI-TARS-72B-DPO.sh [vllm|sglang|tgi]"
    exit 1
fi

SERVER=$1
MODEL_NAME="UI-TARS-72B-DPO"
MODEL_DIR="./models/$MODEL_NAME"
REPO_ID="ByteDance-Seed/UI-TARS-72B-DPO"

# Activate the environment
source .venv/bin/activate

# Download logic
if [ -d "$MODEL_DIR" ]; then
    echo "$MODEL_NAME already downloaded"
else
    echo "Downloading $MODEL_NAME"
    # Ensure hf-transfer is installed for faster downloads
    export HF_HUB_ENABLE_HF_TRANSFER=1
    ~/.local/bin/hf download $REPO_ID --local-dir $MODEL_DIR
fi

case $SERVER in
  vllm)
    echo "Launching vLLM server with proxy on port 8000..."
    # vLLM configuration for UI-TARS-72B-DPO (based on Qwen2-VL-72B):
    # - max-model-len: 32768 (Qwen2-VL supports up to 32k tokens)
    # - max-num-seqs: 128 (good for 72B model with data parallelism)
    # - gpu-memory-utilization: 0.90 (maximize GPU usage for large model)
    # - tensor-parallel-size: 2 (2 GPUs per replica)
    # - pipeline-parallel-size: 4 (4 data parallel replicas, total 8 GPUs)
    # - disable-custom-all-reduce: for stability with VLMs
    # - enforce-eager: equivalent to CUDA_GRAPHS=0 for VLMs
    # Run vLLM on port 9000, proxy on 8000 to handle max_tokens adjustment
    
    # Detect number of GPUs
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Detected $GPU_COUNT GPUs"
    
    if [ "$GPU_COUNT" -lt 8 ]; then
        echo "WARNING: UI-TARS-72B-DPO requires 8 GPUs, but only $GPU_COUNT detected."
        echo "Proceeding anyway, but expect potential memory issues."
    fi
    
    echo "Using 4 data parallel replicas with 2-way tensor parallelism (8 GPUs total)"
    
    echo "Starting vLLM backend on port 9000..."
    python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL_DIR \
        --served-model-name $MODEL_NAME \
        --dtype bfloat16 \
        --port 9000 \
        --host 127.0.0.1 \
        --trust-remote-code \
        --max-model-len 32768 \
        --max-num-seqs 128 \
        --gpu-memory-utilization 0.90 \
        --tensor-parallel-size 2 \
        --pipeline-parallel-size 4 \
        --disable-custom-all-reduce \
        --enforce-eager &
    
    VLLM_PID=$!
    echo "vLLM backend started (PID: $VLLM_PID)"
    
    # Wait for vLLM to start
    echo "Waiting for vLLM backend to be ready..."
    for i in {1..120}; do
        if curl -s http://127.0.0.1:9000/health > /dev/null 2>&1; then
            echo "vLLM backend is ready!"
            break
        fi
        sleep 2
    done
    
    # Start proxy
    echo "Starting proxy server on port 8000..."
    python3 ./vllm_proxy.py --port 8000 --backend-port 9000 --max-context 32768
    ;;

  sglang)
    echo "Launching SGLang server on port 8000..."
    # SGLang configuration for UI-TARS-72B-DPO (Qwen2-VL 32k context)
    python3 -m sglang.launch_server \
        --model-path $MODEL_DIR \
        --port 8000 \
        --host 0.0.0.0 \
        --dtype bfloat16 \
        --context-length 32768 \
        --tensor-parallel-size 2 \
        --data-parallel-size 4 \
        --chat-template qwen2_vl \
        --trust-remote-code
    ;;

  tgi)
    echo "Launching TGI server via Docker on port 8000..."
    # Get the absolute path to the model directory
    MODEL_ABSOLUTE_PATH=$(realpath $MODEL_DIR)
    
    # Note: TGI requires CUDA_GRAPHS=0 for VLMs to avoid illegal instruction errors
    # PAYLOAD_LIMIT=8000000 prevents request failures due to large images
    docker run --gpus all --shm-size 1g \
        -e CUDA_GRAPHS=0 \
        -e PAYLOAD_LIMIT=8000000 \
        -p 8000:80 \
        -v $MODEL_ABSOLUTE_PATH:/data \
        ghcr.io/huggingface/text-generation-inference:3.2.1 \
        --model-id /data \
        --dtype bfloat16 \
        --max-input-length 32768 \
        --max-batch-prefill-tokens 32768 \
        --max-total-tokens 32769 \
        --num-shard 8
    ;;

  *)
    echo "Error: Invalid server type '$SERVER'. Use vllm, sglang, or tgi."
    exit 1
    ;;
esac
