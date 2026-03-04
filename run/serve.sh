# Activate the environment first
source .venv/bin/activate

# Launch the vLLM server
python3 -m vllm.entrypoints.openai.api_server \
    --model ./hf_ckpt \
    --served-model-name lumine-agent-lv-7b \
    --dtype bfloat16 \
    --port 8000
