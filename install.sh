git clone https://github.com/zlc1004/VeOmni.git
apt update
apt install -y ffmpeg rsync pv
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -LsSf https://hf.co/cli/install.sh | bash
source $HOME/.local/bin/env
hf download Qwen/Qwen2-VL-7B --local-dir ./models/Qwen2-VL-7B-Base
cd VeOmni
uv self update 0.9.8
uv sync --locked  --extra gpu --extra video