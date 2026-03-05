#!/bin/bash

# Usage check
if [ -z "$1" ]; then
    echo "Usage: ./install.sh [vllm|sglang|tgi]"
    exit 1
fi

MODE=$1
PYTHON_BIN="$PWD/.venv/bin/python"

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
    echo "--- Installing TGI (Modern Source) ---"
    # Ensure system-level Rust/Cargo are removed to avoid Edition conflicts
    sudo apt remove -y rustc cargo
    
    # Protoc setup
    PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
    curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
    sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
    sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
    rm -f $PROTOC_ZIP

    # Ensure Modern Rust is installed and SOURCED
    if ! command -v cargo &> /dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    else
        rustup update stable
    fi
    source $HOME/.cargo/env

    # Pre-install Flash-Attn requirements to avoid Makefile escapes
    uv pip install torch==2.4.0 packaging wheel
    uv pip install flash-attn==2.6.1 --no-build-isolation

    git clone https://github.com/huggingface/text-generation-inference
    cd text-generation-inference
    
    # Force Makefile to use our specific venv Python for the server components
    # Use 'cargo build' logic if 'make install' fails to handle the Edition 2024
    PYTHON=$PYTHON_BIN BUILD_EXTENSIONS=True make install || (cargo build --release)
    cd ..
    ;;

  *)
    echo "Error: Invalid argument '$MODE'. Use vllm, sglang, or tgi."
    exit 1
    ;;
esac

echo "--- $MODE Setup Complete ---"
echo "To activate environment, run: source .venv/bin/activate"