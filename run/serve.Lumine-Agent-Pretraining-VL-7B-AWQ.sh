# Activate the environment first
source .venv/bin/activate

if [ -d "./models/Lumine-Agent-Pretraining-VL-7B-AWQ" ]; then
    echo "Lumine-Agent-Pretraining-VL-7B-AWQ already downloaded"
else
    echo "Downloading Lumine-Agent-Pretraining-VL-7B-AWQ"
    /root/.local/bin/hf download koboshchan/Lumine-Agent-Pretraining-VL-7B-AWQ --local-dir ./models/Lumine-Agent-Pretraining-VL-7B-AWQ
fi

# Launch the vLLM server
python3 -m vllm.entrypoints.openai.api_server \
    --model ./models/Lumine-Agent-Pretraining-VL-7B-AWQ \
    --served-model-name lumine-agent-vl-7b \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --port 8000
