#!/bin/bash
source $HOME/.local/bin/env
~/.local/bin/hf download Qwen/Qwen2-VL-7B --local-dir ./models/Qwen2-VL-7B-Base
git clone https://huggingface.co/datasets/koboshchan/genshinPlayData