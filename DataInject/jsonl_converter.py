import os
import json
import argparse
import glob


def convert_to_jsonl(input_dir, output_file):
    images_dir = os.path.join(input_dir, "images")
    texts_dir = os.path.join(input_dir, "texts")

    with open(output_file, "w", encoding="utf-8") as out:
        # Scenario 1: File Metadata (Namespace 6) - Pairs image with its own name/short desc
        image_files = glob.glob(os.path.join(images_dir, "*"))
        for img_path in image_files:
            filename = os.path.basename(img_path)
            clean_name = filename.replace("_", " ").split(".")[0]

            # Lumine / VeOmni JSONL Format
            entry = {
                "id": f"wiki_img_{filename}",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_path},
                            {
                                "type": "text",
                                "text": f"What is depicted in this image from the Genshin Wiki?",
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": f"This image shows {clean_name}."}
                        ],
                    },
                ],
            }
            out.write(json.dumps(entry) + "\n")

        # Scenario 2: Lore/Knowledge (Namespace 0) - Pure text injection
        text_files = glob.glob(os.path.join(texts_dir, "*.txt"))
        for txt_path in text_files:
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                continue

            title = os.path.basename(txt_path).replace("_", " ").replace(".txt", "")

            entry = {
                "id": f"wiki_lore_{title}",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Tell me about {title} in Genshin Impact.",
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": content}],
                    },
                ],
            }
            out.write(json.dumps(entry) + "\n")

    print(f"Successfully created {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Wiki folder to JSONL for VeOmni"
    )
    parser.add_argument("input_dir", help="Input folder (output from downloader)")
    parser.add_argument("--output", required=True, help="Output .jsonl file path")
    args = parser.parse_args()

    convert_to_jsonl(args.input_dir, args.output)
