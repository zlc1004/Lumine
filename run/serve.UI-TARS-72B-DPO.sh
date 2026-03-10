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
    PORT=8000
    GPU_MEMORY_UTIL=0.90
    TENSOR_PARALLEL_SIZE=4
    DATA_PARALLEL_SIZE=2
    
    echo "Starting vLLM with Tensor + Data Parallelism..."
    echo "Model: $REPO_ID"
    echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
    echo "Data Parallel Size: $DATA_PARALLEL_SIZE (Total: 8 GPUs)"
    echo "Port: $PORT"
    echo "GPU memory utilization: ${GPU_MEMORY_UTIL}"
    echo ""
    
    python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL_DIR \
        --served-model-name $MODEL_NAME \
        --port $PORT \
        --host 0.0.0.0 \
        --dtype bfloat16 \
        --trust-remote-code \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --data-parallel-size $DATA_PARALLEL_SIZE \
        --gpu-memory-utilization $GPU_MEMORY_UTIL \
        --disable-custom-all-reduce \
        --enforce-eager
    ;;

  sglang)
    echo "Launching SGLang server on port 8000 (no proxy for multi-GPU)..."
    # SGLang with 8 GPUs runs directly on port 8000
    SGLANG_DISABLE_CUDNN_CHECK=1 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python3 -m sglang.launch_server \
        --model-path $MODEL_DIR \
        --port 8000 \
        --host 0.0.0.0 \
        --dtype bfloat16 \
        --tensor-parallel-size 4 \
        --data-parallel-size 2 \
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
