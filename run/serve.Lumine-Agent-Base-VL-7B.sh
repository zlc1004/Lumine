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
    /root/.local/bin/hf download $REPO_ID --local-dir $MODEL_DIR
fi

case $SERVER in
  vllm)
    echo "Launching vLLM server on port 8000..."
    python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL_DIR \
        --served-model-name lumine-agent-vl-7b \
        --dtype bfloat16 \
        --port 8000 \
        --trust-remote-code \
        --max-model-len 16384
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
    echo "Launching TGI launcher on port 8000..."
    # TGI requires CUDA_GRAPHS=0 to prevent VRAM spikes with vision patches
    export CUDA_GRAPHS=0
    text-generation-launcher \
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