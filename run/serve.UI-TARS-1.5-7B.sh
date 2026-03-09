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
    # Detect number of GPUs
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Detected $GPU_COUNT GPUs"
    
    # Configure parallelism based on GPU count
    if [ "$GPU_COUNT" -eq 1 ]; then
        # Single GPU: use full 128k context with proxy
        echo "Single GPU configuration: 128k context with proxy"
        PORT=9000
        PROXY_PORT=8000
        MAX_MODEL_LEN=131072
        TENSOR_PARALLEL_SIZE=1
        DATA_PARALLEL_SIZE=1
        USE_PROXY=true
    elif [ "$GPU_COUNT" -ge 4 ]; then
        # Multi-GPU: TP=2, DP=GPU_COUNT/2, default context, no proxy
        echo "Multi-GPU configuration: TP=2, DP=$((GPU_COUNT/2)), default context, direct port 8000"
        PORT=8000
        TENSOR_PARALLEL_SIZE=2
        DATA_PARALLEL_SIZE=$((GPU_COUNT/2))
        USE_PROXY=false
    else
        echo "WARNING: $GPU_COUNT GPUs detected. Recommended: 1, 4, or 8 GPUs."
        echo "Using single GPU config anyway."
        PORT=9000
        PROXY_PORT=8000
        MAX_MODEL_LEN=131072
        TENSOR_PARALLEL_SIZE=1
        DATA_PARALLEL_SIZE=1
        USE_PROXY=true
    fi
    
    GPU_MEMORY_UTIL=0.90
    
    echo "Starting vLLM server..."
    echo "Port: $PORT"
    echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
    echo "Data Parallel Size: $DATA_PARALLEL_SIZE"
    if [ "$USE_PROXY" = true ]; then
        echo "Max model length: $MAX_MODEL_LEN tokens"
        echo "Proxy enabled on port: $PROXY_PORT"
    else
        echo "Max model length: auto (default from model config)"
    fi
    echo "GPU memory utilization: ${GPU_MEMORY_UTIL}"
    echo ""
    
    # Build vLLM command
    VLLM_CMD="python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL_DIR \
        --served-model-name UI-TARS-1.5-7B \
        --dtype bfloat16 \
        --port $PORT \
        --trust-remote-code \
        --gpu-memory-utilization $GPU_MEMORY_UTIL \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --disable-custom-all-reduce \
        --enforce-eager"
    
    # Add host configuration
    if [ "$USE_PROXY" = true ]; then
        VLLM_CMD="$VLLM_CMD --host 127.0.0.1"
    else
        VLLM_CMD="$VLLM_CMD --host 0.0.0.0"
    fi
    
    # Add max-model-len for single GPU (128k context)
    if [ "$USE_PROXY" = true ]; then
        VLLM_CMD="VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 $VLLM_CMD --max-model-len $MAX_MODEL_LEN"
    fi
    
    # Add data parallelism for multi-GPU
    if [ "$DATA_PARALLEL_SIZE" -gt 1 ]; then
        VLLM_CMD="$VLLM_CMD --data-parallel-size $DATA_PARALLEL_SIZE"
    fi
    
    # Start vLLM
    if [ "$USE_PROXY" = true ]; then
        # Background process with proxy
        eval "$VLLM_CMD &"
        VLLM_PID=$!
        echo "vLLM backend started (PID: $VLLM_PID)"
        
        # Wait for vLLM to start
        echo "Waiting for vLLM backend to be ready..."
        for i in {1..60}; do
            if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
                echo "vLLM backend is ready!"
                break
            fi
            sleep 2
        done
        
        # Start proxy
        echo "Starting proxy server on port $PROXY_PORT..."
        python3 ./vllm_proxy.py --port $PROXY_PORT --backend-port $PORT --max-context $MAX_MODEL_LEN
    else
        # Direct execution without proxy
        eval "$VLLM_CMD"
    fi
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