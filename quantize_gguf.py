#!/usr/bin/env python3
"""
Convert Qwen2-VL model to GGUF format and quantize to Q8_0 and Q4_K_M.
This script automates the use of llama.cpp conversion tools.

Prerequisites:
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp && make
    pip install -r llama.cpp/requirements.txt

Usage:
    python quantize_gguf.py ./models/Lumine-Agent-Pretraining-VL-7B --llama_dir ./llama.cpp
"""

import argparse
import os
import subprocess
import shutil


def run_command(command):
    print(f"Executing: {' '.join(command)}")
    result = subprocess.run(command, check=True)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert and Quantize Qwen2-VL to GGUF"
    )
    parser.add_argument("model_dir", type=str, help="Path to the BF16 model directory")
    parser.add_argument(
        "--llama_dir",
        type=str,
        default="./llama.cpp",
        help="Path to llama.cpp repository",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.rstrip("/")
    model_name = os.path.basename(model_dir)
    llama_dir = args.llama_dir

    # Paths to llama.cpp scripts
    convert_script = os.path.join(llama_dir, "convert_hf_to_gguf.py")
    quantize_bin = os.path.join(llama_dir, "llama-quantize")

    if not os.path.exists(convert_script):
        print(f"Error: Could not find {convert_script}. Ensure --llama_dir is correct.")
        return

    # 1. Convert to F16 GGUF first (Requirement for further quantization)
    f16_gguf = f"{model_dir}/{model_name}.f16.gguf"
    print(f"--- Step 1: Converting HF to F16 GGUF ---")
    run_command(["python", convert_script, model_dir, "--outfile", f16_gguf])

    # 2. Quantize to Q8_0 and Q4_K_M
    quants = ["Q8_0", "Q4_K_M"]

    for q_type in quants:
        output_file = f"{model_dir}/{model_name}.{q_type}.gguf"
        print(f"\n--- Step 2: Quantizing to {q_type} ---")

        # Check if llama-quantize exists
        if not os.path.exists(quantize_bin) and os.path.exists(
            os.path.join(llama_dir, "quantize")
        ):
            quantize_bin = os.path.join(llama_dir, "quantize")

        run_command([quantize_bin, f16_gguf, output_file, q_type])

    # Cleanup F16 GGUF if you want to save space
    # os.remove(f16_gguf)

    print(f"\nSuccess! GGUF files are located in {model_dir}")


if __name__ == "__main__":
    main()
