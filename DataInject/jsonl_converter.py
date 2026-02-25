import os
import json
import argparse
import re
import tempfile
import logging
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def clean_wiki_text(text):
    """
    Structural Cleaning: Regex-strip everything after 'Navigation' or 'Other Languages'.
    Retains only the lore, location (How to Obtain), and version history.
    """
    if not text:
        return ""
    # Strip everything after 'Other Languages' or 'Navigation' blocks
    text = re.split(r"Other Languages|Navigation", text)[0]
    return text.strip()


def run_nemo_pipeline(input_jsonl, output_dir, use_gpu=True, generate_synthetic=False):
    """
    Runs the NeMo Curator pipeline for cleaning and quality filtering.
    """
    try:
        from nemo_curator.pipeline import Pipeline
        from nemo_curator.backends.xenna import XennaExecutor
        from nemo_curator.core.client import RayClient
        from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
        from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
        from nemo_curator.stages.text.filters.heuristic_filter import WordCountFilter
        from nemo_curator.stages.text.filters.fasttext_filter import FastTextLangId
        from nemo_curator.stages.text.modules import Filter, Modify, ScoreFilter
    except ImportError as e:
        logger.error(f"NeMo Curator import failed: {e}")
        return False

    logger.info("Initializing NeMo Curator Pipeline...")

    pipeline = Pipeline(
        name="genshin_wiki_cleaning", description="Clean and filter Genshin Wiki data"
    )

    # 1. Read Dataset
    pipeline.add_stage(JsonlReader(input_jsonl))

    # 2. Structural Cleaning (Map/Modify)
    def cleaning_fn(text):
        return clean_wiki_text(text)

    pipeline.add_stage(Modify(cleaning_fn, input_fields="content"))

    # 3. Quality Filtering
    # WordCountFilter (min_words=15)
    pipeline.add_stage(ScoreFilter(WordCountFilter(min_words=15), text_field="content"))

    # 4. Information Density (Custom Filter)
    def high_info_density_fn(text):
        if not text:
            return False
        lines = text.split("\n")
        version_lines = [l for l in lines if re.search(r"Version \d+\.\d+", l)]
        if len(lines) > 0 and len(version_lines) / len(lines) > 0.5:
            return False
        return True

    pipeline.add_stage(Filter(high_info_density_fn, filter_field="content"))

    # 5. Write Output
    pipeline.add_stage(JsonlWriter(output_dir))

    logger.info("Executing pipeline...")
    executor = XennaExecutor()
    pipeline.run(executor)

    return True


def convert_to_veomni_format(cleaned_dir, final_output_jsonl, media_entries):
    """
    Converts the cleaned text data and original media entries into the final VeOmni JSONL format.
    """
    logger.info("Finalizing VeOmni JSONL format...")

    text_count = 0
    with open(final_output_jsonl, "w", encoding="utf-8") as out:
        if os.path.isdir(cleaned_dir):
            shards = [
                os.path.join(cleaned_dir, f)
                for f in os.listdir(cleaned_dir)
                if f.endswith(".jsonl")
            ]
            for shard in tqdm(shards, desc="Processing text shards"):
                with open(shard, "r", encoding="utf-8") as f:
                    for line in f:
                        entry = json.loads(line)
                        content = entry.get("content", "")
                        filename = entry.get("file", "unknown")
                        title = filename.replace(".txt", "").replace("_", " ")

                        ve_entry = {
                            "id": f"wiki_lore_{filename}",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": f"Tell me about {title}.",
                                        }
                                    ],
                                },
                                {
                                    "role": "assistant",
                                    "content": [{"type": "text", "text": content}],
                                },
                            ],
                        }
                        out.write(json.dumps(ve_entry) + "\n")
                        text_count += 1

        for entry in tqdm(media_entries, desc="Processing media"):
            filename = entry.get("file")
            path = entry.get("path")
            desc = entry.get("desc", "")
            ve_entry = {
                "id": f"wiki_img_{filename}",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": path},
                            {"type": "text", "text": "What is in this image?"},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": desc}]},
                ],
            }
            out.write(json.dumps(ve_entry) + "\n")

    logger.info(
        f"Successfully processed {text_count} text documents and {len(media_entries)} images."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distributed NeMo Curator Wiki Data Processor"
    )
    parser.add_argument("metadata_json", help="Path to the custom metadata.json")
    parser.add_argument("--output", required=True, help="Output .jsonl file path")
    parser.add_argument(
        "--no-gpu", action="store_true", help="Disable GPU acceleration"
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Attempt synthetic QA generation"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run on a small subset for testing"
    )
    args = parser.parse_args()

    if not os.path.exists(args.metadata_json):
        logger.error(f"Metadata file not found: {args.metadata_json}")
        exit(1)

    # Load initial metadata
    with open(args.metadata_json, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    text_entries = [e for e in data_list if e.get("type") == "text"]
    media_entries = [e for e in data_list if e.get("type") == "media"]

    if args.test:
        logger.info("Test mode enabled: processing only first 100 text entries")
        text_entries = text_entries[:100]
        # Also limit media entries for a quick test
        media_entries = media_entries[:10]

    if not text_entries:
        logger.warning("No text entries found to process.")
        # If there's no text but there is media, we can still run the formatting part
        if media_entries:
            convert_to_veomni_format(None, args.output, media_entries)
        exit(0)

    # Use a persistent directory for intermediate NeMo processing
    nemo_tmp_dir = os.path.join(os.getcwd(), "nemo_tmp")
    os.makedirs(nemo_tmp_dir, exist_ok=True)

    raw_text_jsonl = os.path.join(nemo_tmp_dir, "raw_text.jsonl")
    logger.info(f"Preparing {len(text_entries)} text entries for NeMo Curator...")

    with open(raw_text_jsonl, "w", encoding="utf-8") as f:
        for entry in text_entries:
            f.write(json.dumps(entry) + "\n")

    cleaned_text_dir = os.path.join(nemo_tmp_dir, "cleaned_text")
    if os.path.exists(cleaned_text_dir):
        import shutil

        shutil.rmtree(cleaned_text_dir)

    success = run_nemo_pipeline(
        raw_text_jsonl, cleaned_text_dir, use_gpu=not args.no_gpu
    )

    if success:
        convert_to_veomni_format(cleaned_text_dir, args.output, media_entries)
        print(f"\nFinal dataset saved to: {args.output}")
    else:
        logger.error("NeMo Curator pipeline failed.")
