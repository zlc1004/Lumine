git clone https://github.com/zlc1004/VeOmni.git
apt update
apt install -y ffmpeg rsync
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
cd VeOmni
uv self update 0.9.8
uv sync --locked  --extra gpu --extra video