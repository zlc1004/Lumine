#!/usr/bin/bash

sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install -y coreutils build-essential gcc-14 g++-14 libstdc++-14-dev libc++-dev libc++abi-dev
sudo ln -sf /usr/bin/g++-14 /usr/bin/g++
sudo apt install -y curl
curl -s https://packagecloud.io/install/repositories/ookla/speedtest-cli/script.deb.sh | sudo bash
sudo apt install -y speedtest
git clone https://github.com/aristocratos/btop.git
cd btop
make
sudo make install
