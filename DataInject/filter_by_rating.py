"""
Filter rated Q&A pairs by quality threshold.

Reads a rated knowledge JSONL file and splits it into two files based on rating:
- Entries with rating <= threshold go to "bad" output
- Entries with rating > threshold go to "good" output

Usage:
    python filter_by_rating.py -i veomni_knowledge.jsonl -r 25 \\
        -o veomni_knowledge_good.jsonl -e veomni_knowledge_bad.jsonl
    
    python filter_by_rating.py -i veomni_knowledge.jsonl -r 0 \\
        -o veomni_knowledge_positive.jsonl -e veomni_knowledge_negative.jsonl
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def filter_by_rating(
    input_file: str,
    output_good: str,
    output_bad: str,
    min_rating: int,
):
    """Filter entries by rating threshold."""

    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    logger.info(f"Reading from: {input_file}")
    logger.info(f"Min rating threshold: {min_rating}")
    logger.info(f"Good output (rating > {min_rating}): {output_good}")
    logger.info(f"Bad output (rating <= {min_rating}): {output_bad}")
    logger.info("")

    # Read and classify entries
    entries = []
    unrated_count = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
                entries.append(entry)

                if entry.get("rating") is None:
                    unrated_count += 1

            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Failed to parse JSON - {e}")

    total = len(entries)
    logger.info(f"Loaded {total} entries")

    if unrated_count > 0:
        logger.warning(
            f"{unrated_count} entries have no rating (will be treated as bad)"
        )

    # Split by rating
    good_entries = []
    bad_entries = []

    for entry in entries:
        rating = entry.get("rating")

        # Treat unrated (None) as bad
        if rating is None or rating <= min_rating:
            bad_entries.append(entry)
        else:
            good_entries.append(entry)

    # Write outputs
    logger.info("")
    logger.info(f"Writing {len(good_entries)} good entries to {output_good}")
    with open(output_good, "w", encoding="utf-8") as f:
        for entry in good_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Writing {len(bad_entries)} bad entries to {output_bad}")
    with open(output_bad, "w", encoding="utf-8") as f:
        for entry in bad_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Summary statistics
    good_ratings = [e["rating"] for e in good_entries if e.get("rating") is not None]
    bad_ratings = [e["rating"] for e in bad_entries if e.get("rating") is not None]

    logger.info("")
    logger.info("=" * 80)
    logger.info("Filter Summary")
    logger.info("=" * 80)
    logger.info(f"Total entries:        {total}")
    logger.info(
        f"Good entries:         {len(good_entries)} ({len(good_entries) / total * 100:.1f}%)"
    )
    logger.info(
        f"Bad entries:          {len(bad_entries)} ({len(bad_entries) / total * 100:.1f}%)"
    )
    logger.info(f"Unrated (as bad):     {unrated_count}")

    if good_ratings:
        logger.info(
            f"Good rating avg:      {sum(good_ratings) / len(good_ratings):.1f}"
        )
        logger.info(f"Good rating range:    {min(good_ratings)} to {max(good_ratings)}")

    if bad_ratings:
        logger.info(f"Bad rating avg:       {sum(bad_ratings) / len(bad_ratings):.1f}")
        logger.info(f"Bad rating range:     {min(bad_ratings)} to {max(bad_ratings)}")

    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Filter rated Q&A pairs by quality threshold"
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Input JSONL file with ratings"
    )
    parser.add_argument(
        "-r",
        "--min-rating",
        type=int,
        required=True,
        help="Minimum rating threshold (entries <= this go to bad output)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output file for good entries (rating > threshold)",
    )
    parser.add_argument(
        "-e",
        "--output-bad",
        required=True,
        help="Output file for bad entries (rating <= threshold)",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Q&A Quality Filter")
    logger.info("=" * 80)

    filter_by_rating(
        args.input,
        args.output,
        args.output_bad,
        args.min_rating,
    )

    logger.info("")
    logger.info("Filtering completed successfully!")


if __name__ == "__main__":
    main()
