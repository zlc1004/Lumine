#!/usr/bin/env python3
"""
Compress BF16 model to INT4 and INT8 using bitsandbytes.

Usage:
    python quantize.py /path/to/model
"""

import argparse
import os
import shutil
import torch
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig


def load_model(model_path, load_in_4bit=False, load_in_8bit=False):
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    return model


def save_quantized_model(model, source_dir, output_path):
    # Save model weights and config
    model.save_pretrained(output_path)

    # Copy all auxiliary files (tokenizer, preprocessors, README, etc.)
    print(f"Copying auxiliary files from {source_dir} to {output_path}...")

    # Patterns for model weights we want to skip (they are already saved in quantized form)
    weight_patterns = [
        "model-00",
        "model.safetensors",
        "pytorch_model.bin",
        "model.msgpack",
        "training_args.bin",
    ]

    for item in os.listdir(source_dir):
        s = os.path.join(source_dir, item)
        d = os.path.join(output_path, item)

        # Skip directories like .git or .cache
        if os.path.isdir(s):
            if item not in [".git", ".cache", "__pycache__"]:
                shutil.copytree(s, d, dirs_exist_ok=True)
            continue

        # For files, check if it's a weight file we should skip
        is_weight = any(p in item for p in weight_patterns)

        # We also skip config.json and generation_config.json as save_pretrained already handled them
        is_core_config = item in [
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
        ]

        if not is_weight and not is_core_config:
            if not os.path.exists(d):
                shutil.copy2(s, d)
            else:
                # If it's a small file like README or jinja, overwrite it to be safe
                if not item.endswith(".json"):
                    shutil.copy2(s, d)

    print(f"Saved complete quantized model to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compress BF16 model to INT4/INT8")
    parser.add_argument("model_dir", type=str, help="Path to the HF model directory")
    args = parser.parse_args()

    model_dir = args.model_dir.rstrip("/")
    model_name = os.path.basename(model_dir)

    int8_dir = f"{model_dir}-INT8"
    int4_dir = f"{model_dir}-INT4"

    print(f"Loading model from {model_dir} (INT8)...")
    model = load_model(model_dir, load_in_8bit=True)

    print(f"Saving INT8 model to {int8_dir}...")
    save_quantized_model(model, model_dir, int8_dir)

    del model
    torch.cuda.empty_cache()

    print(f"Loading model from {model_dir} (INT4)...")
    model = load_model(model_dir, load_in_4bit=True)

    print(f"Saving INT4 model to {int4_dir}...")
    save_quantized_model(model, model_dir, int4_dir)

    print("Done!")


if __name__ == "__main__":
    main()
