# Activate the environment first
source .venv/bin/activate

# Launch the vLLM server
python3 -m vllm.entrypoints.openai.api_server \
    --model ./hf_ckpt \
    --served-model-name custom-model \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --port 8000