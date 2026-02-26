git submodule update --init --recursive
apt update
apt install -y ffmpeg rsync pv lftp pigz iputils-ping
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -LsSf https://hf.co/cli/install.sh | bash
source $HOME/.local/bin/env
cd VeOmni
~/.local/bin/uv self update 0.9.8
~/.local/bin/uv sync --locked --extra gpu --extra video --extra dit
~/.local/bin/uv pip install accelerate autoawq llm-compressor datasets
~/.local/bin/uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130