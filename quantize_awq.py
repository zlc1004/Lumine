#!/usr/bin/env python3
"""
Compress BF16 Qwen2-VL model to 4-bit AWQ using AutoAWQ.
Calibrated using genshinPlayData.

Usage:
    python quantize_awq.py ./models/Lumine-Agent-Pretraining-VL-7B
"""

import argparse
import os
import json
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def load_calibration_data(data_path, num_samples=128):
    print(f"Loading calibration data from {data_path}...")
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line)
            # For Lumine/Genshin data, we'll use the 'action' field or fall back to the whole string
            text = data.get("action", "") or str(data)
            samples.append(text)
    return samples


def main():
    parser = argparse.ArgumentParser(description="Quantize Qwen2-VL to 4-bit AWQ")
    parser.add_argument("model_dir", type=str, help="Path to the BF16 model directory")
    parser.add_argument(
        "--data",
        type=str,
        default="./genshinPlayData/metadata.jsonl",
        help="Path to calibration data",
    )
    args = parser.parse_args()

    model_path = args.model_dir.rstrip("/")
    output_path = f"{model_path}-AWQ"

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    # Load model and tokenizer
    print(f"Loading model: {model_path}")
    model = AutoAWQForCausalLM.from_pretrained(
        model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize
    print("Starting AWQ quantization (this may take a while)...")
    calibration_data = load_calibration_data(args.data)
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calibration_data)

    # Save
    print(f"Saving AWQ model to: {output_path}")
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)

    # Copy auxiliary files (README, preprocessors, etc.)
    import shutil

    for item in os.listdir(model_path):
        s = os.path.join(model_path, item)
        d = os.path.join(output_path, item)
        if not os.path.exists(d) and os.path.isfile(s):
            shutil.copy2(s, d)

    print("Done!")


if __name__ == "__main__":
    main()
