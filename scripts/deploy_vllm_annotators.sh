#!/bin/bash
# Deploy Qwen3-VL-32B-Instruct with Data Parallelism (8 GPUs)
# Uses vLLM's internal load balancing with data-parallel-size

MODEL="Qwen/Qwen3-VL-32B-Instruct"
PORT=8000
MAX_MODEL_LEN=4096
GPU_MEMORY_UTIL=0.90
DATA_PARALLEL_SIZE=8

echo "Starting vLLM with Data Parallelism..."
echo "Model: $MODEL"
echo "Data Parallel Size: $DATA_PARALLEL_SIZE (8 GPUs)"
echo "Port: $PORT"
echo "Max model length: $MAX_MODEL_LEN tokens"
echo "GPU memory utilization: ${GPU_MEMORY_UTIL}"
echo ""

echo "Starting vLLM server with internal load balancing..."

~/.local/bin/uv run -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port $PORT \
    --host 0.0.0.0 \
    --data-parallel-size $DATA_PARALLEL_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs 256 \
    --trust-remote-code
