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
    python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL_DIR \
        --served-model-name UI-TARS-1.5-7B \
        --dtype bfloat16 \
        --port 8000 \
        --trust-remote-code
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
    echo "Launching TGI launcher on port 8000..."
    # Note: TGI requires CUDA_GRAPHS=0 for many VLMs to avoid illegal instruction errors
    # Also uses 'port' instead of 'port' mapping in the launcher
    export CUDA_GRAPHS=0
    ./text-generation-inference/target/release/text-generation-launcher \
        --model-id $MODEL_DIR \
        --port 8000 \
        --dtype bfloat16 \
        --max-input-length 65536 \
        --max-total-tokens 65537
    ;;

  *)
    echo "Error: Invalid server type '$SERVER'. Use vllm, sglang, or tgi."
    exit 1
    ;;
esac