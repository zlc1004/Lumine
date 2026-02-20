#!/usr/bin/env python3
"""
DataCombine - Combine multiple datasets into one

Usage:
    python DataCombine.py --input dataset/ --output combined/
    python DataCombine.py --input dataset/ --output combined/ --prefix
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm


def combine_dataset(input_dir: Path, output_dir: Path, add_prefix: bool = False):
    """Combine all sub-dataset JSONL files into one"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all metadata.jsonl files
    metadata_files = list(input_dir.glob("*/metadata.jsonl"))

    if not metadata_files:
        print(f"No metadata.jsonl files found in {input_dir}")
        return

    print(f"Found {len(metadata_files)} datasets")

    # Combine all samples
    all_samples = []
    stage1_samples = []
    stage2_samples = []
    stage3_samples = []

    for meta_file in tqdm(metadata_files, desc="Combining datasets"):
        dataset_name = meta_file.parent.name

        # Read metadata
        with open(meta_file, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                if add_prefix:
                    sample["image"] = f"{dataset_name}_{sample['image']}"
                    sample["source"] = dataset_name
                all_samples.append(sample)

        # Read stage1 pretrain
        stage1_file = meta_file.parent / "stage1_pretrain.jsonl"
        if stage1_file.exists():
            with open(stage1_file, "r", encoding="utf-8") as f:
                for line in f:
                    sample = json.loads(line)
                    # Convert to new format: {"images": ["frames/..."], "text": "..."}
                    if "image" in sample:
                        image_filename = sample.pop("image")
                        sample["images"] = [f"frames/{image_filename}"]
                    if add_prefix:
                        sample["images"] = [
                            f"{dataset_name}/{img}" for img in sample.get("images", [])
                        ]
                    stage1_samples.append(sample)

        # Read stage2 instruct
        stage2_file = meta_file.parent / "stage2_instruct.jsonl"
        if stage2_file.exists():
            with open(stage2_file, "r", encoding="utf-8") as f:
                for line in f:
                    sample = json.loads(line)
                    if "image" in sample:
                        image_filename = sample.pop("image")
                        sample["images"] = [f"frames/{image_filename}"]
                    if add_prefix:
                        sample["images"] = [
                            f"{dataset_name}/{img}" for img in sample.get("images", [])
                        ]
                    stage2_samples.append(sample)

        # Read stage3 reasoning
        stage3_file = meta_file.parent / "stage3_reasoning.jsonl"
        if stage3_file.exists():
            with open(stage3_file, "r", encoding="utf-8") as f:
                for line in f:
                    sample = json.loads(line)
                    if "image" in sample:
                        image_filename = sample.pop("image")
                        sample["images"] = [f"frames/{image_filename}"]
                    if add_prefix:
                        sample["images"] = [
                            f"{dataset_name}/{img}" for img in sample.get("images", [])
                        ]
                    stage3_samples.append(sample)

    # Write combined files
    print(f"\nTotal samples: {len(all_samples)}")

    with open(output_dir / "all_samples.jsonl", "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    if stage1_samples:
        with open(output_dir / "stage1_pretrain.jsonl", "w", encoding="utf-8") as f:
            for sample in stage1_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"Stage1: {len(stage1_samples)}")

    if stage2_samples:
        with open(output_dir / "stage2_instruct.jsonl", "w", encoding="utf-8") as f:
            for sample in stage2_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"Stage2: {len(stage2_samples)}")

    if stage3_samples:
        with open(output_dir / "stage3_reasoning.jsonl", "w", encoding="utf-8") as f:
            for sample in stage3_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"Stage3: {len(stage3_samples)}")

    # Copy frames if they exist
    frames_dir = input_dir / "frames"
    if frames_dir.exists():
        combined_frames_dir = output_dir / "frames"
        combined_frames_dir.mkdir(parents=True, exist_ok=True)

        for meta_file in tqdm(metadata_files, desc="Copying frames"):
            dataset_name = meta_file.parent.name
            src_frames = meta_file.parent / "frames"
            if src_frames.exists():
                for frame in list(src_frames.glob("*.png")) + list(
                    src_frames.glob("*.jpg")
                ):
                    dst_name = (
                        f"{dataset_name}_{frame.name}" if add_prefix else frame.name
                    )
                    dst_path = combined_frames_dir / dst_name
                    if not dst_path.exists():
                        import shutil

                        shutil.copy2(frame, dst_path)

        print(f"Frames copied to {combined_frames_dir}")

    print(f"\nCombined dataset saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Combine multiple datasets")
    parser.add_argument("--input", required=True, help="Input dataset folder")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument(
        "--prefix",
        action="store_true",
        help="Add dataset name prefix to image filenames",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return

    combine_dataset(input_dir, output_dir, args.prefix)


if __name__ == "__main__":
    main()
