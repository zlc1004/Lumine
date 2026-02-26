#!/usr/bin/env python3
"""
Compress BF16 Qwen2-VL model to 4-bit using llm-compressor.
Calibrated using genshinPlayData.

Usage:
    python quantize_awq.py ./models/Lumine-Agent-Pretraining-VL-7B
"""

import argparse
import os
import json
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoTokenizer, AutoModel, AutoConfig


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Qwen2-VL to 4-bit using llm-compressor"
    )
    parser.add_argument("model_dir", type=str, help="Path to the BF16 model directory")
    parser.add_argument(
        "--scheme",
        type=str,
        default="W4A16",
        help="Quantization scheme (default: W4A16)",
    )
    args = parser.parse_args()

    model_path = args.model_dir.rstrip("/")
    output_path = f"{model_path}-{args.scheme}"

    # Load model and tokenizer
    print(f"Loading model: {model_path}")

    # Load config to check model type
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Use AutoModel to automatically detect the correct model class
    model = AutoModel.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Configure quantization recipe
    # W4A16 = 4-bit weights, 16-bit activations using RTN (Round-to-Nearest)
    recipe = QuantizationModifier(
        targets="Linear",
        scheme=args.scheme,
        ignore=["lm_head"],  # Don't quantize the language model head
    )

    # Apply quantization using oneshot
    print(f"Starting {args.scheme} quantization (this may take a while)...")
    oneshot(model=model, recipe=recipe)

    # Test generation (skip for VL models that need image inputs)
    print("========== SAMPLE GENERATION ==============")
    try:
        input_ids = tokenizer("Hello, I am", return_tensors="pt").input_ids.to(
            model.device
        )
        output = model.generate(input_ids, max_new_tokens=20)
        print(tokenizer.decode(output[0]))
    except Exception as e:
        print(f"Generation test skipped (VL model may need image input): {e}")
    print("==========================================")

    # Save to disk in compressed-tensors format
    print(f"Saving quantized model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
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
