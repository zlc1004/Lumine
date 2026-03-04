#!/usr/bin/bash

sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install -y coreutils build-essential gcc-13 g++-13
git clone https://github.com/aristocratos/btop.git
cd btop
make CXX=g++-13
sudo make install
