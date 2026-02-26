import os
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def map_text_to_images(
    metadata_json, output_jsonl, cleaned_jsonl=None, xml_mapping=None
):
    """
    Matches text lore with corresponding image assets.
    Uses an optional XML mapping for higher accuracy.
    Creates a combined JSONL for multi-modal training.
    """
    logger.info(f"Loading metadata from {metadata_json}...")
    with open(metadata_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    xml_map = {}
    if xml_mapping and os.path.exists(xml_mapping):
        logger.info(f"Loading XML mapping from {xml_mapping}...")
        with open(xml_mapping, "r", encoding="utf-8") as f:
            xml_map = json.load(f)

    texts = {}
    images_by_name = {}  # "Item Plain Vase Ocher.png" -> entry
    images_by_basename = {}  # "Item_Plain_Vase_Ocher" -> entry

    # Extract all media entries from metadata_json
    for entry in data:
        if entry.get("type") == "media":
            filename = entry.get("file", "")
            basename = os.path.splitext(filename)[0]
            images_by_name[filename] = entry
            images_by_basename[basename] = entry
            # Also index by normalized name (spaces replaced with underscores)
            images_by_name[filename.replace(" ", "_")] = entry
            images_by_basename[basename.replace(" ", "_")] = entry

    # If cleaned_jsonl is provided, use it for text entries
    if cleaned_jsonl and os.path.exists(cleaned_jsonl):
        logger.info(f"Loading cleaned text from {cleaned_jsonl}...")
        with open(cleaned_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    entry_id = entry.get("id", "")
                    if "wiki_lore_" in entry_id:
                        filename = entry_id.replace("wiki_lore_", "")
                        basename = os.path.splitext(filename)[0]
                        # Extract content from assistant message
                        content = entry["messages"][1]["content"][0]["text"]
                        texts[basename] = {
                            "content": content,
                            "file": filename,
                            "title": basename.replace("_", " "),
                        }
                except (IndexError, KeyError, json.JSONDecodeError):
                    continue
    else:
        # Fallback to original metadata_json for text
        for entry in data:
            if entry.get("type") == "text":
                filename = entry.get("file", "")
                basename = os.path.splitext(filename)[0]
                texts[basename] = entry
                texts[basename]["title"] = basename.replace("_", " ")

    logger.info(
        f"Found {len(texts)} text entries and {len(images_by_name)} image entries (including aliases)."
    )

    matches = 0
    with open(output_jsonl, "w", encoding="utf-8") as out:
        for basename, text_entry in texts.items():
            content = text_entry.get("content", "")
            title = text_entry.get("title", "")

            img_entry = None

            # 1. Try XML mapping first
            if title in xml_map:
                img_name = xml_map[title]
                img_entry = images_by_name.get(img_name) or images_by_name.get(
                    img_name.replace(" ", "_")
                )

            # 2. Try direct basename match if XML failed
            if not img_entry:
                img_entry = images_by_basename.get(basename) or images_by_basename.get(
                    basename.replace(" ", "_")
                )

            if img_entry:
                matches += 1
                img_path = img_entry.get("path")
                ve_entry = {
                    "id": f"multi_{basename}",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img_path},
                                {
                                    "type": "text",
                                    "text": f"What can you tell me about the item shown in this image?",
                                },
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": content}],
                        },
                    ],
                }
            else:
                # Text only fallback
                ve_entry = {
                    "id": f"text_{basename}",
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

    logger.info(
        f"Successfully mapped {matches} text-image pairs. Total entries: {len(texts)}"
    )
    logger.info(f"Output saved to {output_jsonl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map Text Lore to Image Assets")
    parser.add_argument("metadata_json", help="Path to metadata.json")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--cleaned_jsonl", help="Path to cleaned lore JSONL (optional)")
    parser.add_argument("--xml_mapping", help="Path to xml_mapping.json (optional)")
    args = parser.parse_args()

    map_text_to_images(
        args.metadata_json, args.output, args.cleaned_jsonl, args.xml_mapping
    )
