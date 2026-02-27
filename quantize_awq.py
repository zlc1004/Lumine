#!/usr/bin/env python3
"""
Compress BF16 Qwen2-VL model to 4-bit using AWQ quantization via llm-compressor.
Calibrated using GenshinData.

Usage:
    python quantize_awq.py ./models/Lumine-Agent-Pretraining-VL-7B
    python quantize_awq.py ./models/Lumine-Agent-Pretraining-VL-7B --dataset ./MyData
"""

import argparse
import os
import json
from pathlib import Path
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from transformers import AutoTokenizer, AutoModel, AutoConfig
from datasets import Dataset


def load_calibration_data(data_path, num_samples=512):
    """Load calibration data from JSONL file or directory"""
    print(f"Loading calibration data from {data_path}...")
    samples = []

    data_path = Path(data_path)

    # Find JSONL files
    if data_path.is_file() and data_path.suffix == ".jsonl":
        jsonl_files = [data_path]
    elif data_path.is_dir():
        jsonl_files = list(data_path.glob("*.jsonl"))
        if not jsonl_files:
            jsonl_files = list(data_path.glob("**/*.jsonl"))
    else:
        raise ValueError(f"Invalid data path: {data_path}")

    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {data_path}")

    print(f"Found {len(jsonl_files)} JSONL file(s)")

    # Load samples from JSONL files
    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if len(samples) >= num_samples:
                    break
                try:
                    data = json.loads(line)
                    # Try different text field names
                    text = (
                        data.get("text", "")
                        or data.get("action", "")
                        or data.get("content", "")
                        or str(data)
                    )
                    if text.strip():
                        samples.append({"text": text})
                except json.JSONDecodeError:
                    continue

        if len(samples) >= num_samples:
            break

    print(f"Loaded {len(samples)} calibration samples")
    return Dataset.from_list(samples)


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Qwen2-VL to 4-bit using AWQ via llm-compressor"
    )
    parser.add_argument("model_dir", type=str, help="Path to the BF16 model directory")
    parser.add_argument(
        "--dataset",
        type=str,
        default="./GenshinData",
        help="Path to calibration dataset directory or JSONL file (default: ./GenshinData)",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default="W4A16",
        help="Quantization scheme (default: W4A16)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=512,
        help="Number of calibration samples (default: 512)",
    )
    args = parser.parse_args()

    model_path = args.model_dir.rstrip("/")
    output_path = f"{model_path}-AWQ"

    # Load calibration data
    calibration_dataset = load_calibration_data(args.dataset, args.num_samples)

    # Load model and tokenizer
    print(f"Loading model: {model_path}")

    # Load config to check model type
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Use AutoModel to automatically detect the correct model class
    model = AutoModel.from_pretrained(
        model_path,
        device_map="auto",
        dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Configure AWQ quantization recipe
    # AWQ = Activation-aware Weight Quantization
    recipe = AWQModifier(
        targets="Linear",
        scheme=args.scheme,
        ignore=["lm_head"],  # Don't quantize the language model head
    )

    # Apply AWQ quantization using oneshot with calibration data
    print(f"Starting AWQ {args.scheme} quantization (this may take a while)...")
    oneshot(
        model=model,
        dataset=calibration_dataset,
        recipe=recipe,
        max_seq_length=2048,
        num_calibration_samples=args.num_samples,
    )

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
