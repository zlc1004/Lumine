#!/usr/bin/bash

if command -v sudo &> /dev/null; then
    SUDO="sudo"
    echo "--- sudo detected, will use for system commands ---"
else
    SUDO=""
    echo "--- No sudo available (running as root), proceeding without sudo ---"
fi

$SUDO add-apt-repository ppa:ubuntu-toolchain-r/test
$SUDO apt update
$SUDO apt install -y coreutils build-essential gcc-14 g++-14 libstdc++-14-dev libc++-dev libc++abi-dev
$SUDO ln -sf /usr/bin/g++-14 /usr/bin/g++
$SUDO apt install -y curl
curl -s https://packagecloud.io/install/repositories/ookla/speedtest-cli/script.deb.sh | $SUDO bash
$SUDO apt install -y speedtest
git clone https://github.com/aristocratos/btop.git
cd btop
make
$SUDO make install
