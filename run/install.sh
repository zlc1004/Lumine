#!/bin/bash

# Usage check
if [ -z "$1" ]; then
    echo "Usage: ./install.sh [vllm|sglang|tgi]"
    exit 1
fi

MODE=$1

# 1. Common System Dependencies
echo "--- Installing Common System Dependencies ---"
sudo apt update
sudo apt install -y ffmpeg rsync pv lftp pigz build-essential iputils-ping \
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
    # STEP 1: Core stack with cu124 index
    uv pip install vllm transformers accelerate \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        --index-strategy unsafe-best-match --upgrade
    
    # STEP 2: Flash Attention build (must be non-isolated to see STEP 1's torch)
    uv pip install vllm-flash-attn --no-build-isolation
    
    # STEP 3: Final package alignment
    uv pip install --upgrade vllm torch transformers accelerate \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        --index-strategy unsafe-best-match
    
    # Verification
    python3 -c "import torch; import vllm; print(f'Environment Restored: Torch {torch.__version__}')"
    ;;

  sglang)
    echo "--- Installing SGLang (Source) ---"
    git clone https://github.com/zlc1004/sglang
    cd sglang
    uv pip install --upgrade pip
    uv pip install -e "python"
    cd ..
    ;;

  tgi)
    echo "--- Installing TGI (Source) ---"
    # TGI Specific: Protoc installation
    PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
    curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
    sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
    sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
    rm -f $PROTOC_ZIP

    # TGI Specific: Rust installation (required for build)
    if ! command -v cargo &> /dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source $HOME/.cargo/env
    fi

    git clone https://github.com/huggingface/text-generation-inference
    cd text-generation-inference
    BUILD_EXTENSIONS=True make install
    cd ..
    ;;

  *)
    echo "Error: Invalid argument '$MODE'. Use vllm, sglang, or tgi."
    exit 1
    ;;
esac

echo "--- $MODE Setup Complete ---"
echo "To activate environment, run: source .venv/bin/activate"