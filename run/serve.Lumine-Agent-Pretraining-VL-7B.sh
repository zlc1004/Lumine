# Activate the environment first
source .venv/bin/activate

if [ -d "./models/Lumine-Agent-Pretraining-VL-7B" ]; then
    echo "Lumine-Agent-Pretraining-VL-7B already downloaded"
else
    echo "Downloading Lumine-Agent-Pretraining-VL-7B"
    /root/.local/bin/hf download koboshchan/Lumine-Agent-Pretraining-VL-7B --local-dir ./models/Lumine-Agent-Pretraining-VL-7B
fi

# Launch the vLLM server
python3 -m vllm.entrypoints.openai.api_server \
    --model ./models/Lumine-Agent-Pretraining-VL-7B \
    --served-model-name lumine-agent-vl-7b \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --port 8000