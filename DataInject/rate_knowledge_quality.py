"""
Rate Q&A pair quality for Genshin Impact knowledge injection training.

Reads base_metadata.jsonl and veomni_knowledge.jsonl, joins them by ID,
and rates each Q&A pair using the original source text as context. 
Writes results back with a new "rating" key added.

Supports resuming: already-rated entries are skipped.

Usage:
    python rate_knowledge_quality.py \\
        --base genshinPlayData/base_metadata.jsonl \\
        --knowledge genshinPlayData/veomni_knowledge.jsonl
    python rate_knowledge_quality.py \\
        --base genshinPlayData/base_metadata.jsonl \\
        --knowledge genshinPlayData/veomni_knowledge.jsonl \\
        --output genshinPlayData/veomni_knowledge_rated.jsonl \\
        --base-url http://localhost:8000/v1 \\
        --model Qwen/Qwen3-VL-32B-Instruct \\
        --concurrent 32
"""

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from tqdm.asyncio import tqdm_asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress HTTP request logs from openai/httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

file_handler = logging.FileHandler("rate_knowledge_quality.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
)
logger.addHandler(file_handler)


RATING_PROMPT = """\
You are evaluating Q&A pairs that will be used to train an AI agent to play Genshin Impact.

Rate the following Q&A pair on a scale from -100 to 100 using your best judgment for the EXACT rating value.

DO NOT use round numbers like 50, 75, 85, etc. Use precise ratings like 34, 67, -23, 91, etc.

Rating guidelines (but use any integer in the range):
  -100 to -75: Completely wrong, harmful, or totally irrelevant to Genshin Impact
   -74 to -25: Misleading, very vague, or barely on-topic
   -24 to  24: Trivial information with little training value, or very generic
    25 to  74: Useful Genshin-specific knowledge (items, quests, mechanics, lore)
    75 to 100: Critical gameplay info, precise details, or important lore an agent must know

Original source text (for context):
{source_text}

Q&A pair to rate:
Question: {question}
Answer: {answer}

Respond with ONLY a JSON object in this exact format, no other text:
{{"rating": <integer from -100 to 100>, "reason": "<one sentence>"}}

Remember: Use SPECIFIC ratings (e.g., 67, -82, 38), NOT round multiples of 5 or 10."""


async def rate_entry(
    client,
    entry: Dict[str, Any],
    source_text: str,
    model: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Rate a single Q&A entry with source context. Returns the entry with a 'rating' key added."""
    question = entry.get("question", "")
    answer = entry.get("answer", "")

    # Truncate source text if too long
    max_context = 2000
    truncated_source = source_text[:max_context]
    if len(source_text) > max_context:
        truncated_source += "... [truncated]"

    prompt = RATING_PROMPT.format(
        source_text=truncated_source, question=question, answer=answer
    )

    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=128,
                    temperature=0.0,
                )
                raw = response.choices[0].message.content.strip()

                # Strip markdown code fences if present
                if raw.startswith("```"):
                    raw = re.sub(r"^```[^\n]*\n?", "", raw)
                    raw = re.sub(r"\n?```$", "", raw)
                    raw = raw.strip()

                parsed = json.loads(raw)
                rating = int(parsed["rating"])
                rating = max(-100, min(100, rating))  # Clamp to valid range

                return {**entry, "rating": rating}

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Parse error for id={entry.get('id', '?')} "
                        f"(attempt {attempt + 1}/{max_retries}): {e} — retrying"
                    )
                    await asyncio.sleep(1)
                else:
                    logger.error(
                        f"Failed to parse rating for id={entry.get('id', '?')} "
                        f"after {max_retries} attempts: {e}"
                    )
                    return {**entry, "rating": None}

            except Exception as e:
                wait = 2**attempt
                if attempt < max_retries - 1:
                    logger.warning(
                        f"API error for id={entry.get('id', '?')} "
                        f"(attempt {attempt + 1}/{max_retries}): {e} — retrying in {wait}s"
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        f"API error for id={entry.get('id', '?')} "
                        f"after {max_retries} attempts: {e}"
                    )
                    return {**entry, "rating": None}

    return {**entry, "rating": None}  # Should never reach here


async def run(args):
    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.error("openai package not found. Install with: pip install openai")
        sys.exit(1)

    base_path = Path(args.base)
    knowledge_path = Path(args.knowledge)

    if not base_path.exists():
        logger.error(f"Base metadata file not found: {args.base}")
        sys.exit(1)
    if not knowledge_path.exists():
        logger.error(f"Knowledge file not found: {args.knowledge}")
        sys.exit(1)

    # Output defaults to same file as knowledge (in-place update)
    output_path = Path(args.output) if args.output else knowledge_path

    # Load base metadata (source text) indexed by ID
    logger.info(f"Loading base metadata from {base_path}")
    base_metadata = {}
    with open(base_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                # Extract base ID (remove _qa suffix if present)
                entry_id = entry.get("id", "")
                base_id = (
                    entry_id.rsplit("_qa", 1)[0] if "_qa" in entry_id else entry_id
                )
                # Store original text
                base_metadata[base_id] = entry.get("text", "")

    logger.info(f"Loaded {len(base_metadata)} base metadata entries")

    # Load knowledge Q&A entries
    logger.info(f"Loading Q&A entries from {knowledge_path}")
    entries = []
    with open(knowledge_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    total = len(entries)
    logger.info(f"Loaded {total} Q&A entries")

    # Split into already-rated and pending
    rated = [e for e in entries if e.get("rating") is not None]
    pending = [e for e in entries if e.get("rating") is None]

    if rated:
        logger.info(f"Resuming: {len(rated)} already rated, {len(pending)} pending")
    else:
        logger.info(f"Rating all {len(pending)} entries")

    if not pending:
        logger.info("Nothing to do — all entries already have ratings.")
        return

    # Apply limit if in test mode
    if args.limit and args.limit > 0:
        original_pending = len(pending)
        pending = pending[: args.limit]
        logger.info(
            f"Test mode: limiting to first {len(pending)} of {original_pending} pending entries"
        )

    # Build mapping of entries to source text
    entry_sources = {}
    missing_sources = 0
    for entry in pending:
        entry_id = entry.get("id", "")
        base_id = entry_id.rsplit("_qa", 1)[0] if "_qa" in entry_id else entry_id
        source_text = base_metadata.get(base_id, "")
        if not source_text:
            missing_sources += 1
            source_text = "[Source text not found]"
        entry_sources[entry_id] = source_text

    if missing_sources > 0:
        logger.warning(
            f"{missing_sources} entries missing source text in base_metadata.jsonl"
        )

    # Set up client
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    client = AsyncOpenAI(base_url=args.base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrent)

    logger.info(f"Endpoint: {args.base_url}")
    logger.info(f"Model:    {args.model}")
    logger.info(f"Concurrent requests: {args.concurrent}")

    # Rate pending entries with source context
    tasks = [
        rate_entry(
            client, entry, entry_sources[entry.get("id", "")], args.model, semaphore
        )
        for entry in pending
    ]
    newly_rated = await tqdm_asyncio.gather(*tasks, desc="Rating Q&A pairs")

    # Merge: preserve original order
    id_to_result = {e.get("id"): e for e in newly_rated}
    all_entries = rated + newly_rated

    # Restore original order
    pending_ids = {e.get("id") for e in pending}
    final_entries = []
    for entry in entries:
        eid = entry.get("id")
        if eid in pending_ids:
            final_entries.append(id_to_result.get(eid, entry))
        else:
            final_entries.append(entry)

    # Write atomically via temp file
    tmp = output_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for entry in final_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    shutil.move(str(tmp), str(output_path))

    # Summary stats
    ratings = [e["rating"] for e in final_entries if e.get("rating") is not None]
    failed = sum(1 for e in final_entries if e.get("rating") is None)

    logger.info("")
    logger.info("=" * 80)
    logger.info("Rating Summary")
    logger.info("=" * 80)
    logger.info(f"Total entries:   {total}")
    logger.info(f"Newly rated:     {len(newly_rated)}")
    logger.info(f"Failed (None):   {failed}")
    if ratings:
        logger.info(f"Rating avg:      {sum(ratings) / len(ratings):.1f}")
        logger.info(f"Rating min/max:  {min(ratings)} / {max(ratings)}")
        buckets = {
            "Excellent (75-100)": sum(1 for r in ratings if r >= 75),
            "Good     (25-74) ": sum(1 for r in ratings if 25 <= r < 75),
            "Neutral  (-24-24)": sum(1 for r in ratings if -24 <= r < 25),
            "Poor     (-74--25)": sum(1 for r in ratings if -74 <= r < -24),
            "Terrible (-100--75)": sum(1 for r in ratings if r < -74),
        }
        for label, count in buckets.items():
            pct = count / len(ratings) * 100
            logger.info(f"  {label}: {count:>6} ({pct:.1f}%)")
    logger.info(f"Output file:     {output_path}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Rate Q&A pair quality for Genshin Impact knowledge training"
    )
    parser.add_argument(
        "--base",
        required=True,
        help="Base metadata JSONL file (base_metadata.jsonl with source text)",
    )
    parser.add_argument(
        "--knowledge",
        required=True,
        help="Knowledge Q&A JSONL file (veomni_knowledge.jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSONL file (default: overwrites knowledge file in-place)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible endpoint base URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-32B-Instruct",
        help="Model name to use for rating (default: Qwen/Qwen3-VL-32B-Instruct)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (default: OPENAI_API_KEY env var, or 'EMPTY' for local)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=32,
        help="Max concurrent requests (default: 32)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Test mode: only rate first N entries (default: rate all)",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Q&A Quality Rater")
    logger.info("=" * 80)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
