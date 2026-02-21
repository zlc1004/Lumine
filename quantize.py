#!/usr/bin/env python3
"""
Compress BF16 model to INT4 and INT8 using bitsandbytes.

Usage:
    python quantize.py /path/to/model
"""

import argparse
import os
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


def save_quantized_model(model, output_path):
    model.save_pretrained(output_path)
    print(f"Saved quantized model to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compress BF16 model to INT4/INT8")
    parser.add_argument("model_dir", type=str, help="Path to the HF model directory")
    args = parser.parse_args()

    model_dir = args.model_dir.rstrip("/")
    model_name = os.path.basename(model_dir)

    int8_dir = f"{model_dir}-INT8"
    int4_dir = f"{model_dir}-INT4"

    print(f"Loading model from {model_dir}...")
    model = load_model(model_dir, load_in_8bit=True)

    print(f"Saving INT8 model to {int8_dir}...")
    save_quantized_model(model, int8_dir)

    del model
    torch.cuda.empty_cache()

    print(f"Loading model from {model_dir}...")
    model = load_model(model_dir, load_in_4bit=True)

    print(f"Saving INT4 model to {int4_dir}...")
    save_quantized_model(model, int4_dir)

    print("Done!")


if __name__ == "__main__":
    main()
