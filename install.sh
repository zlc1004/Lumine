git submodule update --init --recursive
apt update
apt install -y ffmpeg rsync pv lftp pigz iputils-ping
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -LsSf https://hf.co/cli/install.sh | bash
source $HOME/.local/bin/env
cd VeOmni
uv self update 0.9.8
uv sync --locked --extra gpu --extra video --extra dit