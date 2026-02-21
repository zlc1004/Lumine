#!/bin/bash
source $HOME/.local/bin/env
/root/.local/bin/hf download Qwen/Qwen2-VL-7B --local-dir ./models/Qwen2-VL-7B-Base
/root/.local/bin/hf download --repo-type dataset koboshchan/genshinPlayData --local-dir ./genshinPlayData
