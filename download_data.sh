#!/bin/bash
source $HOME/.local/bin/env
# Download Qwen2-VL-7B-Instruct instead of Base for better stop behavior and instruction-following
~/.local/bin/hf download Qwen/Qwen2-VL-7B-Instruct --local-dir ./models/Qwen2-VL-7B-Instruct
git clone https://huggingface.co/datasets/koboshchan/genshinPlayData