import os
import json
import argparse
import webdataset as wds
from tqdm import tqdm


def pack_to_wds(input_jsonl, output_dir, shard_size=1000):
    """
    Packs JSONL entries and their corresponding images into WebDataset (.tar) shards.
    VeOmni/Lumine paradigm: Each sample is a dict with .json, .png (optional), and .txt.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract the base directory of the input_jsonl to find images
    # We assume images are in the same relative structure as the converter saw
    # If not, the JSONL stores absolute paths which we will use.

    pattern = os.path.join(output_dir, "shard-%06d.tar")
    sink = wds.ShardWriter(pattern, maxcount=shard_size)

    with open(input_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(tqdm(lines, desc="Sharding to WDS")):
        data = json.loads(line)
        sample_id = data.get("id", f"sample_{i}")

        # Construct the WDS sample
        # We store the full JSON metadata and the actual binary image
        sample = {
            "__key__": sample_id,
            "json": json.dumps(data).encode("utf-8"),
        }

        # Check for image in the Lumine format
        # data['messages'][0]['content'][0]['image'] usually contains the path
        try:
            img_path = data["messages"][0]["content"][0].get("image")
            if img_path and os.path.exists(img_path):
                with open(img_path, "rb") as img_f:
                    sample["png"] = img_f.read()
        except (IndexError, KeyError):
            pass

        sink.write(sample)

    sink.close()
    print(f"Successfully packed {len(lines)} items into {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JSONL to WebDataset Sharder for VeOmni"
    )
    parser.add_argument("input_jsonl", help="The metadata .jsonl file")
    parser.add_argument("--output", required=True, help="Directory to save .tar shards")
    parser.add_argument("--shard-size", type=int, default=1000, help="Items per shard")
    args = parser.parse_args()

    pack_to_wds(args.input_jsonl, args.output, args.shard_size)
