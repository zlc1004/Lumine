#!/usr/bin/env python3
"""
Compress BF16 Qwen2-VL model to 4-bit AWQ using llm-compressor.
Calibrated using genshinPlayData.

Usage:
    python quantize_awq.py ./models/Lumine-Agent-Pretraining-VL-7B
"""

import argparse
import os
import json
import torch
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset


def load_calibration_data(data_path, num_samples=128):
    print(f"Loading calibration data from {data_path}...")
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line)
            # For Lumine/Genshin data, we'll use the 'action' field or fall back to the whole string
            text = data.get("action", "") or data.get("text", "") or str(data)
            samples.append({"text": text})
    return Dataset.from_list(samples)


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Qwen2-VL to 4-bit AWQ using llm-compressor"
    )
    parser.add_argument("model_dir", type=str, help="Path to the BF16 model directory")
    parser.add_argument(
        "--data",
        type=str,
        default="./genshinPlayData/metadata.jsonl",
        help="Path to calibration data",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="Number of calibration samples (default: 128)",
    )
    args = parser.parse_args()

    model_path = args.model_dir.rstrip("/")
    output_path = f"{model_path}-AWQ"

    # Load model and tokenizer
    print(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load calibration data
    print("Loading calibration data...")
    calibration_dataset = load_calibration_data(args.data, args.num_samples)

    # Configure GPTQ quantization (AWQ-style)
    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",  # 4-bit weights, 16-bit activations
        ignore=["lm_head"],  # Don't quantize the language model head
    )

    # Quantize
    print("Starting quantization (this may take a while)...")
    oneshot(
        model=model,
        dataset=calibration_dataset,
        recipe=recipe,
        max_seq_length=2048,
        num_calibration_samples=args.num_samples,
    )

    # Save
    print(f"Saving quantized model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, save_compressed=True)
    tokenizer.save_pretrained(output_path)

    # Copy auxiliary files (README, preprocessors, etc.)
    import shutil

    for item in os.listdir(model_path):
        s = os.path.join(model_path, item)
        d = os.path.join(output_path, item)
        if not os.path.exists(d) and os.path.isfile(s):
            try:
                shutil.copy2(s, d)
            except Exception as e:
                print(f"Warning: Could not copy {item}: {e}")

    print("Done!")


if __name__ == "__main__":
    main()
