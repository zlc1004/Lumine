#!/bin/bash

# 1. Update and install system dependencies
echo "--- Installing System Dependencies ---"
apt update
apt install -y ffmpeg rsync pv lftp pigz build-essential

# 2. Install uv (Fast Python package manager) and Hugging Face CLI
echo "--- Installing uv and HF CLI ---"
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -LsSf https://hf.co/cli/install.sh | bash

# Source environment to make 'uv' available in this script
source $HOME/.local/bin/env

# 3. Extract the checkpoint using pigz (Multi-threaded gzip)
if [ -f "hf_ckpt.tar.gz" ]; then
    echo "--- Extracting Checkpoint ---"
    # Using pv to see progress and pigz for speed
    pv hf_ckpt.tar.gz | tar -I pigz -xvf -
else
    echo "!!! hf_ckpt.tar.gz not found in current directory !!!"
fi

# 4. Create a Virtual Environment and install Inference stack
echo "--- Setting up Python Environment ---"
uv venv
source .venv/bin/env

# Installing vLLM for the A40/H200 and other helpful libs
uv pip install vllm transformers accelerate flash-attn --no-build-isolation

echo "--- Setup Complete ---"
echo "To activate environment, run: source .venv/bin/activate"