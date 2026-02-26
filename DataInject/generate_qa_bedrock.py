"""
Generate Q&A pairs from Genshin Impact lore using AWS Bedrock.

This script uses NeMo Curator's synthetic data generation pipeline with AWS Bedrock
to create instruction-tuning Q&A pairs from the cleaned wiki text.

Usage with boto3 (requires AWS credentials):
    python generate_qa_bedrock.py --model anthropic.claude-3-5-haiku-20241022-v1:0 --input final_veomni_training.jsonl --output genshin_qa_dataset.jsonl

Usage with OpenAI-compatible API (requires AWS_SESSION_TOKEN):
    python generate_qa_bedrock.py --use-openai-api --base-url https://bedrock-mantle.us-west-2.api.aws/v1 --model anthropic.claude-3-5-haiku-20241022-v1:0 --input final_veomni_training.jsonl --output genshin_qa_dataset.jsonl

Requirements:
    - For boto3: AWS credentials configured (via ~/.aws/credentials or environment variables)
    - For OpenAI API: AWS_SESSION_TOKEN or AWS_TOKEN environment variable
    - boto3 installed (for boto3 mode)
    - openai installed (for OpenAI API mode)
    - NeMo Curator installed
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

# Import NeMo Curator components
from nemo_curator.backends.xenna.executor import XennaExecutor
from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader.jsonl import JsonlReaderStage
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.tasks import DocumentBatch

# Import our custom Bedrock client
from bedrock_client import AsyncBedrockClient, AsyncBedrockOpenAIClient

# Import BaseSyntheticStage to create custom Q&A generator
import pandas as pd
from dataclasses import dataclass
from nemo_curator.stages.synthetic.nemotron_cc.base import BaseSyntheticStage


# Custom prompt for Genshin Impact Q&A generation
GENSHIN_QA_PROMPT = """You are an expert on Genshin Impact lore, gameplay mechanics, and items.

Given the following item/character/quest description from the Genshin Impact wiki, generate 2-3 diverse question-answer pairs that would help train a multimodal AI to understand this content.

Focus on:
1. What the item/character/location is
2. Where it can be found or obtained
3. Its gameplay purpose or lore significance
4. Any related characters, quests, or regions

Format your response as:
Question: [question here]
Answer: [detailed answer here]

Question: [question here]
Answer: [detailed answer here]

Document:
{document}

Generate the Q&A pairs now:"""


@dataclass
class GenshinQAStage(BaseSyntheticStage):
    """Custom stage for generating Genshin Impact Q&A pairs."""

    system_prompt: str = None
    prompt: str = GENSHIN_QA_PROMPT
    input_field: str = "text"
    output_field: str = "qa_pairs"

    @property
    def name(self) -> str:
        return "GenshinQAGeneration"


def extract_text_from_veomni(input_file: str, output_file: str):
    """
    Extract text content from VeOmni format and prepare for Q&A generation.

    Converts from:
    {"id": "...", "messages": [{"role": "assistant", "content": [{"type": "text", "text": "..."}]}]}

    To:
    {"id": "...", "text": "..."}
    """
    logger.info(f"Extracting text from {input_file}...")

    extracted = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line)

                # Extract text from the assistant message
                text_content = None
                if "messages" in entry:
                    for msg in entry["messages"]:
                        if msg["role"] == "assistant":
                            for content in msg["content"]:
                                if content["type"] == "text":
                                    text_content = content["text"]
                                    break
                            break

                if text_content:
                    extracted.append(
                        {
                            "id": entry.get("id", f"entry_{line_num}"),
                            "text": text_content,
                            "image": entry.get("messages", [{}])[0]
                            .get("content", [{}])[0]
                            .get("image", ""),
                        }
                    )

            except Exception as e:
                logger.warning(f"Failed to parse line {line_num}: {e}")
                continue

    # Write extracted text
    with open(output_file, "w", encoding="utf-8") as f:
        for item in extracted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Extracted {len(extracted)} entries to {output_file}")
    return len(extracted)


def merge_qa_results(extracted_file: str, qa_output_file: str, final_output_file: str):
    """
    Merge the Q&A results back with the original data.
    """
    logger.info("Merging Q&A results with original data...")

    # Load extracted data with images
    extracted_data = {}
    with open(extracted_file, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            extracted_data[entry["id"]] = entry

    # Load Q&A results and merge
    merged = []
    with open(qa_output_file, "r", encoding="utf-8") as f:
        for line in f:
            qa_entry = json.loads(line)
            entry_id = qa_entry.get("id", "")

            if entry_id in extracted_data:
                original = extracted_data[entry_id]
                merged.append(
                    {
                        "id": entry_id,
                        "text": qa_entry.get("text", ""),
                        "qa_pairs": qa_entry.get("qa_pairs", ""),
                        "image": original.get("image", ""),
                    }
                )

    # Write merged results
    with open(final_output_file, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Merged {len(merged)} entries to {final_output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate Q&A pairs using AWS Bedrock")
    parser.add_argument(
        "--model",
        default="anthropic.claude-3-5-haiku-20241022-v1:0",
        help="Bedrock model ID (default: Claude 3.5 Haiku)",
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
        "--region",
        default="us-east-1",
        help="AWS region (default: us-east-1)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent Bedrock requests (default: 10)",
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
        "--profile",
        default=None,
        help="AWS profile name (optional, from ~/.aws/credentials)",
    )
    parser.add_argument(
        "--use-openai-api",
        action="store_true",
        help="Use OpenAI-compatible API endpoint instead of boto3",
    )
    parser.add_argument(
        "--base-url",
        default="https://bedrock-mantle.us-west-2.api.aws/v1",
        help="OpenAI-compatible API base URL (default: us-west-2)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for OpenAI-compatible endpoint (uses AWS_SESSION_TOKEN env var if not provided)",
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Step 1: Extract text from VeOmni format
    extracted_file = "extracted_text.jsonl"
    total_entries = extract_text_from_veomni(args.input, extracted_file)

    logger.info(
        f"Starting Q&A generation for {total_entries} entries using {args.model}"
    )
    logger.info(f"Concurrency: {args.max_concurrent}, Temperature: {args.temperature}")

    # Step 2: Initialize Bedrock client
    if args.use_openai_api:
        logger.info(f"Using OpenAI-compatible API: {args.base_url}")
        client = AsyncBedrockOpenAIClient(
            base_url=args.base_url,
            api_key=args.api_key,
            max_concurrent_requests=args.max_concurrent,
            max_retries=3,
            base_delay=1.0,
        )
    else:
        logger.info(f"Using boto3 Bedrock client in region {args.region}")
        client = AsyncBedrockClient(
            region_name=args.region,
            profile_name=args.profile,
            max_concurrent_requests=args.max_concurrent,
            max_retries=3,
            base_delay=1.0,
        )

    generation_config = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=0.9,
    )

    # Step 3: Create NeMo Curator pipeline
    qa_output_file = "qa_generated.jsonl"

    pipeline = Pipeline(
        [
            JsonlReaderStage(extracted_file),
            GenshinQAStage(
                client=client,
                model_name=args.model,
                generation_config=generation_config,
            ),
            JsonlWriter(qa_output_file),
        ]
    )

    # Step 4: Execute pipeline with XennaExecutor
    logger.info("Initializing XennaExecutor for distributed processing...")
    executor = XennaExecutor(memory="1800GB", rmm_pool_size="200GB")

    try:
        pipeline.execute(executor=executor)
        logger.info("✅ Q&A generation completed successfully!")

        # Step 5: Merge results back with image data
        merge_qa_results(extracted_file, qa_output_file, args.output)

        logger.info(f"✅ Final dataset saved to {args.output}")
        logger.info("You can now use this dataset for instruction-tuning VeOmni!")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise
    finally:
        executor.shutdown()


if __name__ == "__main__":
    main()
