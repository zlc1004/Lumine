git clone https://github.com/zlc1004/VeOmni.git
apt update
apt install -y ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
cd VeOmni
uv sync --locked  --extra gpu --extra video