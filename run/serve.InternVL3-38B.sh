#!/usr/bin/bash

# Usage check
if [ -z "$1" ]; then
    echo "Usage: ./serve.InternVL3-38B.sh [vllm|sglang|tgi]"
    exit 1
fi

SERVER=$1
MODEL_NAME="InternVL3-38B"
MODEL_DIR="./models/$MODEL_NAME"
REPO_ID="OpenGVLab/InternVL3-38B"

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
    echo "Launching vLLM server with proxy on port 8000..."
    # vLLM configuration for InternVL3-38B:
    # - max-model-len: 32768 (InternVL3 recommended context length)
    # - max-num-seqs: 128 (lower for 38B model)
    # - gpu-memory-utilization: 0.90 (maximize GPU usage for large model)
    # - tensor-parallel-size: 2 (use 2 GPUs if available, otherwise 1)
    # - disable-custom-all-reduce: for stability with VLMs
    # - enforce-eager: equivalent to CUDA_GRAPHS=0
    # Run vLLM on port 9000, proxy on 8000 to handle max_tokens adjustment
    
    # Detect number of GPUs
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$GPU_COUNT" -gt 1 ]; then
        TP_SIZE=2
        echo "Detected $GPU_COUNT GPUs, using tensor-parallel-size=$TP_SIZE"
    else
        TP_SIZE=1
        echo "Detected $GPU_COUNT GPU, using tensor-parallel-size=$TP_SIZE"
    fi
    
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
        --tensor-parallel-size $TP_SIZE \
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
    echo "Launching SGLang server on port 8000 (no proxy for multi-GPU)..."
    # SGLang configuration for InternVL3-38B with 2 GPUs
    python3 -m sglang.launch_server \
        --model-path $MODEL_DIR \
        --port 8000 \
        --host 0.0.0.0 \
        --dtype bfloat16 \
        --trust-remote-code \
        --context-length 32768 \
        --tensor-parallel-size 2
    ;;

  tgi)
    echo "Launching TGI server via Docker on port 8000..."
    # Get the absolute path to the model directory
    MODEL_ABSOLUTE_PATH=$(realpath $MODEL_DIR)
    
    # TGI requires CUDA_GRAPHS=0 for VLMs
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
        --num-shard 2
    ;;

  *)
    echo "Error: Invalid server type '$SERVER'. Use vllm, sglang, or tgi."
    exit 1
    ;;
esac
