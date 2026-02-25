import os
import json
import argparse
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


import magic


def process_image(img_path):
    """Worker function to process image into custom metadata format using MIME type detection."""
    if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
        return None

    try:
        mime = magic.from_file(img_path, mime=True)
        if not mime.startswith("image/"):
            return None
    except Exception:
        return None

    filename = os.path.basename(img_path)
    clean_name = filename.split(".")[0].replace("_", " ")
    return {
        "type": "media",
        "file": filename,
        "path": img_path,
        "desc": f"This image shows {clean_name}",
    }


def process_lore(txt_path):
    """Worker function to process lore into custom metadata format."""
    filename = os.path.basename(txt_path)
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return None

    # Ignore redirect pages
    if content.lower().startswith("redirect"):
        return None

    return {"type": "text", "file": filename, "content": content}


def extract_metadata(input_dir, output_file):
    """
    Extracts Wiki folder contents into a custom metadata format:
    [{"filename": "data"}, ...]
    """
    images_dir = os.path.join(input_dir, "images")
    texts_dir = os.path.join(input_dir, "texts")

    image_files = glob.glob(os.path.join(images_dir, "*"))
    text_files = glob.glob(os.path.join(texts_dir, "*.txt"))

    num_cores = cpu_count()
    print(f"Parallelizing metadata extraction across {num_cores} cores...")

    all_data = []

    # Parallel Image Processing
    if image_files:
        with Pool(processes=num_cores) as pool:
            results = pool.imap_unordered(process_image, image_files, chunksize=100)
            for res in tqdm(results, total=len(image_files), desc="Mapping Images"):
                if res:
                    all_data.append(res)

    # Parallel Lore Processing
    if text_files:
        with Pool(processes=num_cores) as pool:
            results = pool.imap_unordered(process_lore, text_files, chunksize=100)
            for res in tqdm(results, total=len(text_files), desc="Mapping Lore"):
                if res:
                    all_data.append(res)

    print(f"Saving metadata to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)

    print(f"Successfully created {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-threaded Wiki Metadata Extractor"
    )
    parser.add_argument("input_dir", help="Input folder from downloader")
    parser.add_argument("--output", required=True, help="Output metadata.json path")
    args = parser.parse_args()

    extract_metadata(args.input_dir, args.output)
