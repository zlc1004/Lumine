#!/usr/bin/bash
# Activate the environment first
source .venv/bin/activate

if [ -d "./models/UI-TARS-1.5-7B" ]; then
    echo "UI-TARS-1.5-7B already downloaded"
else
    echo "Downloading UI-TARS-1.5-7B"
    /root/.local/bin/hf download ByteDance-Seed/UI-TARS-1.5-7B --local-dir ./models/UI-TARS-1.5-7B
fi

# Launch the vLLM server
python3 -m vllm.entrypoints.openai.api_server \
    --model ./models/UI-TARS-1.5-7B \
    --served-model-name UI-TARS-1.5-7B \
    --dtype bfloat16 \
    --port 8000
