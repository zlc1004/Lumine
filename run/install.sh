#!/bin/bash

# 1. Update and install system dependencies
echo "--- Installing System Dependencies ---"
apt update
apt install -y ffmpeg rsync pv lftp pigz build-essential iputils-ping

# 2. Install uv (Fast Python package manager) and Hugging Face CLI
echo "--- Installing uv and HF CLI ---"
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -LsSf https://hf.co/cli/install.sh | bash

# Source environment to make 'uv' available in this script
source $HOME/.local/bin/env


# 4. Create a Virtual Environment and install Inference stack
echo "--- Setting up Python Environment ---"
uv venv
source .venv/bin/activate

# Installing vLLM for the A40/H200 and other helpful libs
# uv pip install torch vllm transformers accelerate --no-build-isolation
# uv pip install vllm-flash-attn --no-build-isolation

uv pip install vllm transformers accelerate --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match --upgrade
uv pip install vllm-flash-attn --no-build-isolation
uv pip install --upgrade vllm torch transformers accelerate \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    --index-strategy unsafe-best-match
python3 -c "import torch; import vllm; from torch.library import infer_schema; print('Environment Restored: Torch ' + torch.__version__)"

echo "--- Setup Complete ---"
echo "To activate environment, run: source .venv/bin/activate"
