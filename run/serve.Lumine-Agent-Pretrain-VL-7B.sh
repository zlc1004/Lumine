#!/usr/bin/bash
# Activate the environment first
source .venv/bin/activate

if [ -d "./models/Lumine-Agent-Pretrain-VL-7B" ]; then
    echo "Lumine-Agent-Pretrain-VL-7B already downloaded"
else
    echo "Downloading Lumine-Agent-Pretrain-VL-7B"
    /root/.local/bin/hf download koboshchan/Lumine-Agent-Pretrain-VL-7B --local-dir ./models/Lumine-Agent-Pretrain-VL-7B
fi

# Launch the vLLM server
python3 -m vllm.entrypoints.openai.api_server \
    --model ./models/Lumine-Agent-Pretrain-VL-7B \
    --served-model-name lumine-agent-vl-7b \
    --dtype bfloat16 \
    --port 8000
