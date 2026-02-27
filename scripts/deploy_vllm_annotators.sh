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

# Create log directory
mkdir -p logs/vllm_annotators

LOG_FILE="logs/vllm_annotators/vllm_dp8.log"

echo "Starting vLLM server with internal load balancing..."

nohup vllm serve "$MODEL" \
    --port $PORT \
    --host 0.0.0.0 \
    --data-parallel-size $DATA_PARALLEL_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs 256 \
    --trust-remote-code \
    --disable-log-requests \
    > "$LOG_FILE" 2>&1 &

PID=$!

echo ""
echo "vLLM server started with PID: $PID"
echo "Log file: $LOG_FILE"
echo ""
echo "Wait ~2-3 minutes for model to load across 8 GPUs, then check health:"
echo "  curl http://localhost:$PORT/health"
echo ""
echo "To view logs:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To stop:"
echo "  kill $PID"
echo "  # or"
echo "  pkill -f 'vllm serve.*Qwen3-VL-32B-Instruct'"
