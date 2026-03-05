#!/bin/bash

# Usage check
if [ -z "$1" ]; then
    echo "Usage: ./install.sh [vllm|sglang|tgi]"
    exit 1
fi

MODE=$1
PYTHON_BIN="$PWD/.venv/bin/python"

# Check if sudo is available (RunPod uses single root user without sudo)
if command -v sudo &> /dev/null; then
    SUDO="sudo"
    echo "--- sudo detected, will use for system commands ---"
else
    SUDO=""
    echo "--- No sudo available (running as root), proceeding without sudo ---"
fi

# 1. Common System Dependencies
echo "--- Installing Common System Dependencies ---"
$SUDO apt update
$SUDO apt install -y ffmpeg rsync pv lftp pigz build-essential iputils-ping \
                    python3.10-venv libssl-dev gcc g++ make unzip

# 2. Install uv and HF CLI
echo "--- Installing uv and HF CLI ---"
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -LsSf https://hf.co/cli/install.sh | bash
source $HOME/.local/bin/env

# 3. Setup Python Environment
echo "--- Setting up Python Environment ---"
uv venv
source .venv/bin/activate

case $MODE in
  vllm)
    echo "--- Installing vLLM (Exact Sequence) ---"
    uv pip install vllm transformers accelerate \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        --index-strategy unsafe-best-match --upgrade
    uv pip install vllm-flash-attn --no-build-isolation
    uv pip install --upgrade vllm torch transformers accelerate \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        --index-strategy unsafe-best-match
    $PYTHON_BIN -c "import torch; import vllm; print(f'vLLM Ready: Torch {torch.__version__}')"
    ;;

  sglang)
    echo "--- Installing SGLang (Source) ---"
    git clone https://github.com/zlc1004/sglang
    cd sglang
    uv pip install --upgrade pip
    $PYTHON_BIN -m pip install -e "python"
    cd ..
    ;;

  tgi)
    echo "--- Installing TGI (Docker) ---"
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        echo "Docker not found. Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
        rm get-docker.sh
    fi
    
    # Pull the TGI Docker image (v3.2.1)
    echo "--- Pulling TGI Docker image (v3.2.1) ---"
    docker pull ghcr.io/huggingface/text-generation-inference:3.2.1
    
    echo "--- TGI Docker image ready ---"
    ;;

  *)
    echo "Error: Invalid argument '$MODE'. Use vllm, sglang, or tgi."
    exit 1
    ;;
esac

echo "--- $MODE Setup Complete ---"
echo "To activate environment, run: source .venv/bin/activate"