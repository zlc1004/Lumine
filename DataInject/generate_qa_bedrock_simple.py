"""
Generate Q&A pairs from Genshin Impact lore using AWS Bedrock.

This is a simplified version with direct async processing, progress bars, and detailed logging.

Usage:
    python generate_qa_bedrock_simple.py --use-openai-api --model google.gemma-3-12b-it --input final_veomni_training.jsonl --output genshin_qa_dataset.jsonl

Requirements:
    - AWS_SESSION_TOKEN environment variable (for OpenAI API mode)
    - openai package installed
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from loguru import logger
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
)
logger.add(
    "qa_generation.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG",
    rotation="100 MB",
)


# Custom prompt for Genshin Impact Q&A generation
GENSHIN_QA_PROMPT = """You are an expert on Genshin Impact lore, gameplay mechanics, and items.

Generate 2-3 diverse question-answer pairs about the following Genshin Impact content to train a multimodal AI assistant.

CRITICAL RULES:
1. Only use information explicitly provided in the text below
2. If information is missing or incomplete (e.g., "located in ." or blank fields), state that the information is "not available" or "not specified" WITHOUT referencing "the document", "the text", or "this description"
3. Write answers as if you are directly answering the player's question with your game knowledge
4. Be natural and conversational - never say "the document states", "according to the text", etc.
5. If critical information is missing, you can say "The specific [location/quest/details] are not specified" but phrase it naturally

Focus on:
1. What the item/character/location is
2. Where it can be found or obtained (if specified)
3. Its gameplay purpose or lore significance
4. Any related characters, quests, or regions

You must output ONLY valid JSON in this exact format (no markdown, no extra text):
{{
  "qa_pairs": [
    {{
      "question": "First question here",
      "answer": "Detailed answer here based only on the provided information"
    }},
    {{
      "question": "Second question here",
      "answer": "Detailed answer here based only on the provided information"
    }},
    {{
      "question": "Third question here",
      "answer": "Detailed answer here based only on the provided information"
    }}
  ]
}}

Content:
{document}

Generate the Q&A pairs in JSON format:"""


class BedrockQAGenerator:
    """Async Q&A generator using AWS Bedrock OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_concurrent: int = 10,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        """Initialize the generator."""
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.max_concurrent = max_concurrent
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Statistics
        self.total_processed = 0
        self.total_errors = 0
        self.json_retry_success = 0  # Count successful retries after JSON errors

    async def setup(self):
        """Initialize the OpenAI client."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            logger.error("openai package not found. Install with: pip install openai")
            raise

        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        logger.info(f"✓ OpenAI client initialized with base_url={self.base_url}")
        logger.info(f"✓ Model: {self.model}")
        logger.info(f"✓ Max concurrent requests: {self.max_concurrent}")
        logger.info(f"✓ Temperature: {self.temperature}, Max tokens: {self.max_tokens}")

    async def generate_qa(
        self, entry: Dict[str, Any], retry_count: int = 3
    ) -> Dict[str, Any]:
        """Generate Q&A pairs for a single entry with retry logic."""
        entry_id = entry.get("id", "unknown")
        text = entry.get("text", "")

        if not text:
            logger.warning(f"Skipping entry {entry_id}: no text content")
            return {**entry, "qa_pairs": [], "error": "No text content"}

        # Create prompt
        prompt = GENSHIN_QA_PROMPT.format(document=text[:2000])  # Limit text length
        messages = [{"role": "user", "content": prompt}]

        # Rate limiting with semaphore
        async with self.semaphore:
            for attempt in range(retry_count):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )

                    raw_response = response.choices[0].message.content

                    # Try to parse JSON response
                    try:
                        # Remove markdown code blocks if present
                        cleaned_response = raw_response.strip()
                        if cleaned_response.startswith("```"):
                            # Extract JSON from markdown code block
                            lines = cleaned_response.split("\n")
                            cleaned_response = "\n".join(
                                line
                                for line in lines
                                if not line.strip().startswith("```")
                            )

                        parsed_response = json.loads(cleaned_response)
                        qa_pairs = parsed_response.get("qa_pairs", [])

                        if not isinstance(qa_pairs, list):
                            raise ValueError("qa_pairs is not a list")

                        if len(qa_pairs) == 0:
                            raise ValueError("qa_pairs list is empty")

                        # Success! Valid JSON parsed
                        if attempt > 0:
                            # This was a successful retry
                            self.json_retry_success += 1
                            logger.info(
                                f"✓ JSON retry successful for {entry_id} on attempt {attempt + 1}"
                            )

                        logger.debug(
                            f"✓ Generated {len(qa_pairs)} Q&A pairs for {entry_id}"
                        )

                        self.total_processed += 1

                        return {
                            **entry,
                            "qa_pairs": qa_pairs,
                            "model": self.model,
                        }

                    except (json.JSONDecodeError, ValueError) as parse_error:
                        # JSON parsing failed - treat as retriable error
                        if attempt < retry_count - 1:
                            logger.warning(
                                f"Invalid JSON for {entry_id} (attempt {attempt + 1}/{retry_count}): "
                                f"{parse_error}. Retrying API call..."
                            )
                            await asyncio.sleep(1)  # Brief pause before retry
                            continue  # Retry the API call
                        else:
                            # Final attempt failed, store raw response as fallback
                            logger.error(
                                f"✗ Failed to get valid JSON for {entry_id} after {retry_count} attempts. "
                                f"Storing raw response."
                            )
                            self.total_errors += 1
                            return {
                                **entry,
                                "qa_pairs": raw_response,  # Store as string fallback
                                "model": self.model,
                                "error": f"JSON parse error: {parse_error}",
                            }

                except Exception as e:
                    # API call failed
                    if attempt < retry_count - 1:
                        wait_time = 2**attempt  # Exponential backoff
                        logger.warning(
                            f"API error for {entry_id} (attempt {attempt + 1}/{retry_count}): {e}. "
                            f"Retrying in {wait_time}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(
                            f"✗ Failed to generate Q&A for {entry_id} after {retry_count} attempts: {e}"
                        )
                        self.total_errors += 1
                        return {
                            **entry,
                            "qa_pairs": [],
                            "error": str(e),
                        }

    async def process_batch(
        self, entries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process a batch of entries concurrently with progress bar."""
        logger.info(f"Processing {len(entries)} entries...")

        tasks = [self.generate_qa(entry) for entry in entries]

        # Use tqdm for progress tracking
        results = []
        for coro in tqdm_asyncio.as_completed(
            tasks, total=len(tasks), desc="Generating Q&A"
        ):
            result = await coro
            results.append(result)

        return results


def extract_text_from_veomni(input_file: str, output_file: str) -> int:
    """
    Extract text content from VeOmni format and prepare for Q&A generation.

    Converts from:
    {"id": "...", "messages": [{"role": "assistant", "content": [{"type": "text", "text": "..."}]}]}

    To:
    {"id": "...", "text": "...", "image": "..."}
    """
    logger.info(f"Extracting text from {input_file}...")

    extracted = []
    total_lines = 0

    # Count total lines first for progress bar
    with open(input_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    logger.info(f"Found {total_lines} entries in input file")

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(
            tqdm(f, total=total_lines, desc="Extracting text"), 1
        ):
            try:
                entry = json.loads(line)

                # Extract text from the assistant message
                text_content = None
                image_path = ""

                if "messages" in entry:
                    # Extract image from user message
                    for msg in entry["messages"]:
                        if msg["role"] == "user":
                            for content in msg.get("content", []):
                                if content.get("type") == "image":
                                    image_path = content.get("image", "")
                                    break

                    # Extract text from assistant message
                    for msg in entry["messages"]:
                        if msg["role"] == "assistant":
                            for content in msg.get("content", []):
                                if content.get("type") == "text":
                                    text_content = content.get("text", "")
                                    break
                            break

                if text_content:
                    extracted.append(
                        {
                            "id": entry.get("id", f"entry_{line_num}"),
                            "text": text_content,
                            "image": image_path,
                        }
                    )

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error processing line {line_num}: {e}")
                continue

    # Write extracted text
    logger.info(f"Writing {len(extracted)} extracted entries to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in extracted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(
        f"✓ Extracted {len(extracted)} entries (skipped {total_lines - len(extracted)})"
    )
    return len(extracted)


def merge_qa_results(
    extracted_file: str, results: List[Dict[str, Any]], final_output_file: str
):
    """
    Merge the Q&A results with the original data.
    """
    logger.info("Merging Q&A results with original data...")

    # Write results
    success_count = 0
    error_count = 0
    json_format_count = 0
    text_format_count = 0

    with open(final_output_file, "w", encoding="utf-8") as f:
        for item in tqdm(results, desc="Writing results"):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

            qa_pairs = item.get("qa_pairs")
            if qa_pairs:
                success_count += 1
                if isinstance(qa_pairs, list):
                    json_format_count += 1
                else:
                    text_format_count += 1
            else:
                error_count += 1

    logger.info(f"✓ Wrote {len(results)} entries to {final_output_file}")
    logger.info(f"  - Successfully generated Q&A: {success_count}")
    logger.info(f"    • JSON format: {json_format_count}")
    logger.info(f"    • Text format: {text_format_count}")
    logger.info(f"  - Errors or empty: {error_count}")


async def main():
    parser = argparse.ArgumentParser(description="Generate Q&A pairs using AWS Bedrock")
    parser.add_argument(
        "--model",
        default="anthropic.claude-3-5-haiku-20241022-v1:0",
        help="Bedrock model ID",
    )
    parser.add_argument(
        "--input",
        default="final_veomni_training.jsonl",
        help="Input file (VeOmni format)",
    )
    parser.add_argument(
        "--output",
        default="genshin_qa_dataset.jsonl",
        help="Output file for Q&A dataset",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent requests (default: 10)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens per response (default: 512)",
    )
    parser.add_argument(
        "--use-openai-api",
        action="store_true",
        help="Use OpenAI-compatible API endpoint",
    )
    parser.add_argument(
        "--base-url",
        default="https://bedrock-mantle.us-west-2.api.aws/v1",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (uses AWS_SESSION_TOKEN env var if not provided)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Process only first N entries (for testing)",
    )

    args = parser.parse_args()

    # Banner
    logger.info("=" * 80)
    logger.info("AWS Bedrock Q&A Generation Pipeline")
    logger.info("=" * 80)

    # Validate input file
    if not Path(args.input).exists():
        logger.error(f"✗ Input file not found: {args.input}")
        sys.exit(1)

    logger.info(f"✓ Input file: {args.input}")
    logger.info(f"✓ Output file: {args.output}")

    # Get API key
    api_key = args.api_key or os.getenv("AWS_SESSION_TOKEN") or os.getenv("AWS_TOKEN")
    if not api_key and args.use_openai_api:
        logger.error("✗ API key required. Set AWS_SESSION_TOKEN or use --api-key")
        sys.exit(1)

    # Step 1: Extract text from VeOmni format
    logger.info("")
    logger.info("Step 1/3: Extracting text from VeOmni format...")
    logger.info("-" * 80)

    extracted_file = "extracted_text.jsonl"
    total_entries = extract_text_from_veomni(args.input, extracted_file)

    if total_entries == 0:
        logger.error("✗ No entries extracted. Check input file format.")
        sys.exit(1)

    # Load extracted entries
    entries = []
    with open(extracted_file, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))

    # Limit batch size if specified
    if args.batch_size:
        entries = entries[: args.batch_size]
        logger.info(f"✓ Limited to first {len(entries)} entries for testing")

    # Step 2: Generate Q&A pairs
    logger.info("")
    logger.info(f"Step 2/3: Generating Q&A pairs for {len(entries)} entries...")
    logger.info("-" * 80)

    generator = BedrockQAGenerator(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        max_concurrent=args.max_concurrent,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    await generator.setup()

    results = await generator.process_batch(entries)

    logger.info(f"✓ Processing complete!")
    logger.info(f"  - Total processed: {generator.total_processed}")
    logger.info(f"  - Total errors: {generator.total_errors}")
    if generator.json_retry_success > 0:
        logger.info(f"  - JSON retry successes: {generator.json_retry_success}")
        logger.info(f"    (Entries that succeeded after JSON parse failures)")

    # Step 3: Merge and save results
    logger.info("")
    logger.info("Step 3/3: Merging results and saving...")
    logger.info("-" * 80)

    merge_qa_results(extracted_file, results, args.output)

    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ Pipeline completed successfully!")
    logger.info("=" * 80)
    logger.info(f"Output saved to: {args.output}")
    logger.info(f"Log saved to: qa_generation.log")


if __name__ == "__main__":
    asyncio.run(main())
