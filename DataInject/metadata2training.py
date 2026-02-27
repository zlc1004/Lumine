"""
Convert Q&A metadata to training format.

This script takes the output from generate_qa_bedrock_simple.py and converts it
to simple Q&A format for VeOmni knowledge injection training.

Input format:
{
    "id": "...",
    "text": "...",
    "image": "...",
    "qa_pairs": [{"question": "...", "answer": "..."}],
    "model": "..."
}

Output format (simple):
{
    "id": "...",
    "question": "...",
    "answer": "..."
}

Usage:
    python metadata2training.py -i base_metadata.jsonl -o training.jsonl
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

file_handler = logging.FileHandler("metadata2training.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
)
logger.addHandler(file_handler)


def convert_entry(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert a single metadata entry to simple Q&A pairs.

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


def process_file(input_file: str, output_file: str):
    """Process the input file and convert to simple training format."""

    logger.info(f"Reading input file: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    logger.info(f"Found {total_lines} entries in input file")

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

                if "error" in entry:
                    logger.warning(f"Skipping entry {entry_id}: {entry['error']}")
                    error_entries += 1
                    skipped_entries += 1
                    continue

                if "qa_pairs" not in entry:
                    logger.warning(
                        f"Line {line_num} (ID: {entry_id}): Missing qa_pairs field"
                    )
                    skipped_entries += 1
                    continue

                qa_pairs = entry.get("qa_pairs")
                if not isinstance(qa_pairs, list) or len(qa_pairs) == 0:
                    skipped_entries += 1
                    continue

                converted = convert_entry(entry)

                for item in converted:
                    outfile.write(json.dumps(item, ensure_ascii=False) + "\n")
                    total_qa_pairs += 1

                if len(converted) < len(qa_pairs):
                    invalid_qa_count += len(qa_pairs) - len(converted)

            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Failed to parse JSON - {e}")
                skipped_entries += 1
            except Exception as e:
                logger.error(f"Line {line_num}: Unexpected error - {e}")
                skipped_entries += 1

    logger.info("")
    logger.info("=" * 80)
    logger.info("Conversion Summary")
    logger.info("=" * 80)
    logger.info(f"Total input entries:       {total_lines}")
    logger.info(f"Total Q&A pairs generated: {total_qa_pairs}")
    logger.info(f"Skipped entries:           {skipped_entries}")
    if error_entries > 0:
        logger.info(f"  - Entries with errors:   {error_entries}")
    if invalid_qa_count > 0:
        logger.info(f"Invalid Q&A pairs skipped: {invalid_qa_count}")
    logger.info(f"Output file:               {output_file}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Q&A metadata to simple training format for VeOmni knowledge injection"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input JSONL file (output from generate_qa_bedrock_simple.py)",
    )
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("Metadata to Training Format Converter")
    logger.info("=" * 80)
    logger.info(f"Input file:  {args.input}")
    logger.info(f"Output file: {args.output}")
    logger.info("")

    process_file(args.input, args.output)

    logger.info("")
    logger.info("Conversion completed successfully!")


if __name__ == "__main__":
    main()
