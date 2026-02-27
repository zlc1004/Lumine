"""
Convert Q&A metadata to training format.

This script takes the output from generate_qa_bedrock_simple.py and converts it
to a standard training format suitable for fine-tuning language models.

Input format:
{
    "id": "...",
    "text": "...",
    "image": "...",
    "qa_pairs": [{"question": "...", "answer": "..."}],
    "model": "..."
}

Output format (VeOmni/OpenAI chat format):
{
    "id": "...",
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "..."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
    ]
}

Or simple text format:
{
    "id": "...",
    "question": "...",
    "answer": "..."
}

Usage:
    python metadata2training.py -i base_metadata.jsonl -o training.jsonl
    python metadata2training.py -i base_metadata.jsonl -o training.jsonl --format simple
    python metadata2training.py -i base_metadata.jsonl -o training.jsonl --format veomni --include-image
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List

from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Also log to file
file_handler = logging.FileHandler("metadata2training.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
)
logger.addHandler(file_handler)


def convert_to_simple_format(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert a single entry to simple Q&A format.

    Returns a list of training examples (one per Q&A pair).
    """
    entry_id = entry.get("id", "unknown")
    qa_pairs = entry.get("qa_pairs", [])

    if not isinstance(qa_pairs, list):
        logger.warning(f"Skipping entry {entry_id}: qa_pairs is not a list")
        return []

    results = []
    for idx, qa in enumerate(qa_pairs):
        if not isinstance(qa, dict):
            logger.warning(f"Skipping Q&A pair {idx} in {entry_id}: not a dict")
            continue

        question = qa.get("question", "")
        answer = qa.get("answer", "")

        if not question or not answer:
            logger.warning(
                f"Skipping Q&A pair {idx} in {entry_id}: missing question or answer"
            )
            continue

        results.append(
            {
                "id": f"{entry_id}_qa{idx}",
                "question": question,
                "answer": answer,
            }
        )

    return results


def convert_to_veomni_format(
    entry: Dict[str, Any], include_image: bool = False
) -> List[Dict[str, Any]]:
    """
    Convert a single entry to VeOmni/OpenAI chat format.

    Returns a list of training examples (one per Q&A pair).
    """
    entry_id = entry.get("id", "unknown")
    qa_pairs = entry.get("qa_pairs", [])
    image_path = entry.get("image", "")

    if not isinstance(qa_pairs, list):
        logger.warning(f"Skipping entry {entry_id}: qa_pairs is not a list")
        return []

    results = []
    for idx, qa in enumerate(qa_pairs):
        if not isinstance(qa, dict):
            logger.warning(f"Skipping Q&A pair {idx} in {entry_id}: not a dict")
            continue

        question = qa.get("question", "")
        answer = qa.get("answer", "")

        if not question or not answer:
            logger.warning(
                f"Skipping Q&A pair {idx} in {entry_id}: missing question or answer"
            )
            continue

        # Build user message content
        user_content = []
        if include_image and image_path:
            user_content.append(
                {
                    "type": "image",
                    "image": image_path,
                }
            )
        user_content.append(
            {
                "type": "text",
                "text": question,
            }
        )

        # Build assistant message content
        assistant_content = [
            {
                "type": "text",
                "text": answer,
            }
        ]

        results.append(
            {
                "id": f"{entry_id}_qa{idx}",
                "messages": [
                    {
                        "role": "user",
                        "content": user_content,
                    },
                    {
                        "role": "assistant",
                        "content": assistant_content,
                    },
                ],
            }
        )

    return results


def convert_to_text_format(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert a single entry to text format (question + answer concatenated).

    Returns a list of training examples (one per Q&A pair).
    """
    entry_id = entry.get("id", "unknown")
    qa_pairs = entry.get("qa_pairs", [])

    if not isinstance(qa_pairs, list):
        logger.warning(f"Skipping entry {entry_id}: qa_pairs is not a list")
        return []

    results = []
    for idx, qa in enumerate(qa_pairs):
        if not isinstance(qa, dict):
            logger.warning(f"Skipping Q&A pair {idx} in {entry_id}: not a dict")
            continue

        question = qa.get("question", "")
        answer = qa.get("answer", "")

        if not question or not answer:
            logger.warning(
                f"Skipping Q&A pair {idx} in {entry_id}: missing question or answer"
            )
            continue

        results.append(
            {
                "id": f"{entry_id}_qa{idx}",
                "text": f"Question: {question}\n\nAnswer: {answer}",
            }
        )

    return results


def process_file(
    input_file: str,
    output_file: str,
    format_type: str = "veomni",
    include_image: bool = False,
):
    """Process the input file and convert to training format."""

    # Count total lines
    logger.info(f"Reading input file: {input_file}")
    total_lines = 0
    with open(input_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    logger.info(f"Found {total_lines} entries in input file")

    # Process entries
    total_qa_pairs = 0
    skipped_entries = 0
    invalid_qa_count = 0
    error_entries = 0

    with (
        open(input_file, "r", encoding="utf-8") as infile,
        open(output_file, "w", encoding="utf-8") as outfile,
    ):
        for line_num, line in enumerate(
            tqdm(infile, total=total_lines, desc="Converting"), 1
        ):
            try:
                entry = json.loads(line)
                entry_id = entry.get("id", f"line_{line_num}")

                # Check if entry has an error
                if "error" in entry:
                    error_msg = entry.get("error", "Unknown error")
                    logger.warning(f"Skipping entry {entry_id}: {error_msg}")
                    error_entries += 1
                    skipped_entries += 1
                    continue

                # Check if entry has qa_pairs
                if "qa_pairs" not in entry:
                    logger.warning(
                        f"Line {line_num} (ID: {entry_id}): Missing qa_pairs field"
                    )
                    skipped_entries += 1
                    continue

                # Check if qa_pairs is valid
                qa_pairs = entry.get("qa_pairs")
                if not isinstance(qa_pairs, list):
                    logger.warning(
                        f"Line {line_num} (ID: {entry_id}): qa_pairs is not a list (type: {type(qa_pairs).__name__})"
                    )
                    skipped_entries += 1
                    continue

                if len(qa_pairs) == 0:
                    logger.debug(
                        f"Line {line_num} (ID: {entry_id}): Empty qa_pairs list"
                    )
                    skipped_entries += 1
                    continue

                # Convert based on format
                if format_type == "simple":
                    converted = convert_to_simple_format(entry)
                elif format_type == "text":
                    converted = convert_to_text_format(entry)
                else:  # veomni
                    converted = convert_to_veomni_format(entry, include_image)

                # Write converted entries
                for item in converted:
                    outfile.write(json.dumps(item, ensure_ascii=False) + "\n")
                    total_qa_pairs += 1

                if len(converted) < len(qa_pairs):
                    invalid_qa_count += len(qa_pairs) - len(converted)

            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Failed to parse JSON - {e}")
                skipped_entries += 1
                continue
            except Exception as e:
                logger.error(f"Line {line_num}: Unexpected error - {e}")
                skipped_entries += 1
                continue

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("Conversion Summary")
    logger.info("=" * 80)
    logger.info(f"Total input entries: {total_lines}")
    logger.info(f"Total Q&A pairs generated: {total_qa_pairs}")
    logger.info(f"Skipped entries: {skipped_entries}")
    if error_entries > 0:
        logger.info(f"  - Entries with errors: {error_entries}")
    if invalid_qa_count > 0:
        logger.info(f"Invalid Q&A pairs skipped: {invalid_qa_count}")
    logger.info(f"Output file: {output_file}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Q&A metadata to training format"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input JSONL file (output from generate_qa_bedrock_simple.py)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output JSONL file in training format",
    )
    parser.add_argument(
        "--format",
        choices=["veomni", "simple", "text"],
        default="veomni",
        help="Output format (default: veomni)",
    )
    parser.add_argument(
        "--include-image",
        action="store_true",
        help="Include image paths in output (only for veomni format)",
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Banner
    logger.info("=" * 80)
    logger.info("Metadata to Training Format Converter")
    logger.info("=" * 80)
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Format: {args.format}")
    if args.format == "veomni":
        logger.info(f"Include images: {args.include_image}")
    logger.info("")

    # Process
    process_file(
        args.input,
        args.output,
        format_type=args.format,
        include_image=args.include_image,
    )

    logger.info("")
    logger.info("Conversion completed successfully!")


if __name__ == "__main__":
    main()
