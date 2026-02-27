#!/usr/bin/env python3
"""
Parallel Game Frame Annotation Script
======================================
Uses vLLM with data parallelism to annotate Genshin Impact gameplay frames
with action sequences for Stage 2 SFT training.

Usage:
    python annotate_frames_parallel.py --frames_dir ./genshin_frames --output genshin_actions.jsonl
"""

import os
import json
import base64
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import aiohttp
from tqdm.asyncio import tqdm_asyncio


@dataclass
class AnnotationConfig:
    """Configuration for the annotation process."""

    port: int = 8000  # Single vLLM endpoint with data parallelism
    max_concurrent: int = 64  # Max concurrent requests
    timeout: int = 60  # Request timeout in seconds
    max_tokens: int = 512  # Max tokens to generate per response


ANNOTATION_PROMPT = """You are an AI assistant that analyzes Genshin Impact gameplay screenshots and generates action sequences.

Given this screenshot, describe what actions the player should take. Format your response as a precise action sequence:

<|action_start|>X Y Z ; key1 key2 key3 ; key4 key5 key6<|action_end|>

Where:
- X Y Z: Camera movement (integers, 0 if no movement)
- key1-key3: First action frame keys (e.g., W A S D for movement, Space for jump, E for skill)
- key4-key6: Second action frame keys
- Continue pattern for more frames as needed

Example outputs:
- Walking forward: <|action_start|>0 0 0 ; W ; W ; W ; W ; W<|action_end|>
- Jump and attack: <|action_start|>0 0 0 ; Space ; Space ; J ; J ; J<|action_end|>
- Look right and move: <|action_start|>50 0 0 ; D ; D ; D ; D<|action_end|>

Analyze the screenshot and provide ONLY the action sequence, no other text."""


def encode_image_to_base64(image_path: Path) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


async def annotate_frame(
    session: aiohttp.ClientSession,
    image_path: Path,
    config: AnnotationConfig,
) -> Dict[str, Any]:
    """
    Annotate a single frame using vLLM data parallel endpoint.

    Args:
        session: aiohttp session for making requests
        image_path: Path to the image file
        config: Annotation configuration

    Returns:
        Dictionary with annotation results
    """
    url = f"http://localhost:{config.port}/v1/chat/completions"

    # Prepare the image
    image_base64 = encode_image_to_base64(image_path)

    # Create the request payload (OpenAI-compatible format)
    payload = {
        "model": "Qwen/Qwen3-VL-32B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ANNOTATION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            }
        ],
        "max_tokens": config.max_tokens,
        "temperature": 0.1,  # Low temperature for more consistent outputs
        "top_p": 0.9,
    }

    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=config.timeout)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                return {
                    "image": str(image_path),
                    "success": False,
                    "error": f"HTTP {response.status}: {error_text}",
                }

            result = await response.json()
            action_sequence = result["choices"][0]["message"]["content"].strip()

            return {
                "image": str(image_path),
                "action_sequence": action_sequence,
                "success": True,
            }

    except asyncio.TimeoutError:
        return {
            "image": str(image_path),
            "success": False,
            "error": f"Timeout after {config.timeout}s",
        }
    except Exception as e:
        return {
            "image": str(image_path),
            "success": False,
            "error": str(e),
        }


async def check_health(port: int) -> bool:
    """Check if vLLM server is healthy."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://localhost:{port}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                return response.status == 200
    except:
        return False


async def wait_for_server(config: AnnotationConfig) -> None:
    """Wait for vLLM server to be ready."""
    print("Waiting for vLLM server to be ready...")

    while True:
        if await check_health(config.port):
            print("Server ready!")
            break
        print(f"  Server not ready yet, retrying...")
        await asyncio.sleep(5)


async def annotate_all_frames(
    frames_dir: Path, output_path: Path, config: AnnotationConfig
) -> None:
    """
    Annotate all frames using vLLM data parallel endpoint.

    Args:
        frames_dir: Directory containing game frame images
        output_path: Path to save JSONL output
        config: Annotation configuration
    """
    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = [
        f for f in frames_dir.iterdir() if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No images found in {frames_dir}")
        return

    print(f"Found {len(image_files)} images to annotate")

    # Wait for vLLM server to be ready
    await wait_for_server(config)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(config.max_concurrent)

    async def annotate_with_semaphore(session, image_path):
        async with semaphore:
            return await annotate_frame(session, image_path, config)

    # Process all images with progress bar
    async with aiohttp.ClientSession() as session:
        tasks = [annotate_with_semaphore(session, img) for img in image_files]

        results = await tqdm_asyncio.gather(*tasks, desc="Annotating frames")

    # Save results to JSONL
    success_count = 0
    error_count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            if result["success"]:
                # Write in VeOmni format (matches genshinPlayData/veomni_pretrain.jsonl)
                training_sample = {
                    "instruction": "What actions should be taken in this game screenshot?",
                    "images": [result["image"]],  # Array of image paths
                    "answer": result["action_sequence"],
                }
                f.write(json.dumps(training_sample, ensure_ascii=False) + "\n")
                success_count += 1
            else:
                # Log errors
                print(
                    f"Error on {result['image']}: {result.get('error', 'Unknown error')}"
                )
                error_count += 1

    print(f"\nAnnotation complete!")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate game frames using vLLM data parallel deployment"
    )
    parser.add_argument(
        "--frames_dir",
        type=Path,
        default=Path("./genshin_frames"),
        help="Directory containing game frame images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./genshin_actions.jsonl"),
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="vLLM server port (default: 8000)"
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=64,
        help="Max concurrent requests (default: 64)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)",
    )

    args = parser.parse_args()

    config = AnnotationConfig(
        port=args.port,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
    )

    # Run the annotation
    asyncio.run(annotate_all_frames(args.frames_dir, args.output, config))


if __name__ == "__main__":
    main()
