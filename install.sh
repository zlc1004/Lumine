#!/usr/bin/bash

git submodule update --init --recursive
sudo apt update
sudo apt install -y ffmpeg rsync pv lftp pigz iputils-ping python3.10-venv
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -LsSf https://hf.co/cli/install.sh | bash
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/huggingface/xet-core/refs/heads/main/git_xet/install.sh | sh
git config lfs.concurrenttransfers 300 --global
git config submodule.fetchJobs 300 --global
source $HOME/.local/bin/env
cd VeOmni
~/.local/bin/uv self update 0.9.8
~/.local/bin/uv sync --locked --extra gpu --extra video --extra dit
source .venv/bin/activate
cd ..
~/.local/bin/uv pip install accelerate autoawq llmcompressor datasets
~/.local/bin/uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
