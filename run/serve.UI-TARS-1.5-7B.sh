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
    /root/.local/bin/hf download ByteDance-Seed/UI-TARS-1.5-7B --local-dir $MODEL_DIR
fi

case $SERVER in
  vllm)
    echo "Launching vLLM server on port 8000..."
    # vLLM configuration matching TGI settings:
    # - max-model-len: 65536 (equivalent to max-input-length)
    # - max-num-seqs: 256 (equivalent to max-batch-prefill-tokens / avg_seq_len)
    # - disable-custom-all-reduce: for stability with VLMs
    # - enforce-eager: equivalent to CUDA_GRAPHS=0 for TGI
    python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL_DIR \
        --served-model-name UI-TARS-1.5-7B \
        --dtype bfloat16 \
        --port 8000 \
        --host 0.0.0.0 \
        --trust-remote-code \
        --max-model-len 65536 \
        --max-num-seqs 256 \
        --disable-custom-all-reduce \
        --enforce-eager
    ;;

  sglang)
    echo "Launching SGLang server on port 8000..."
    # SGLang is excellent for the fast TTFT needed for your GUI agent
    python3 -m sglang.launch_server \
        --model-path $MODEL_DIR \
        --port 8000 \
        --host 0.0.0.0 \
        --dtype bfloat16 \
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
        --max-input-length 65536 \
        --max-batch-prefill-tokens 65536 \
        --max-total-tokens 65537
    ;;

  *)
    echo "Error: Invalid server type '$SERVER'. Use vllm, sglang, or tgi."
    exit 1
    ;;
esac