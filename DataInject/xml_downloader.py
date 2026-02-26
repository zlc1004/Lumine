import os
import argparse
import asyncio
import aiohttp
import lxml.etree as ET
import mwparserfromhell
from urllib.parse import quote
from tqdm import tqdm
import re
from multiprocessing import Pool, cpu_count
import hashlib

# Concurrency control
MAX_CONCURRENT_DOWNLOADS = 50
semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)


def clean_filename(filename):
    """Sanitizes a string to be a safe filename for Windows and Linux."""
    return re.sub(r'[<>:"/\\|?*\'\x00-\x1f]', "_", filename).strip(". ")


def get_fandom_static_url(filename, wiki_name="gensin-impact"):
    """
    Constructs the direct static.wikia.nocookie.net URL using MD5 hashing.
    Pattern: https://static.wikia.nocookie.net/{wiki_name}/images/{h1}/{h1}{h2}/{filename}
    """
    # Fandom uses spaces as underscores in the URL/Hash
    link_name = filename.replace("File:", "").strip().replace(" ", "_")

    # Calculate MD5 hash of the filename
    md5_hash = hashlib.md5(link_name.encode("utf-8")).hexdigest()
    h1 = md5_hash[0]
    h2 = md5_hash[1]

    return f"https://static.wikia.nocookie.net/{wiki_name}/images/{h1}/{h1}{h2}/{quote(link_name)}"


async def download_image(session, filename, output_dir, pbar):
    """Downloads an image using the direct static URL (bypassing Special:FilePath)."""
    # Try the most common wiki name variants if one fails
    wiki_variants = ["gensin-impact", "genshin-impact"]

    async with semaphore:
        for wiki in wiki_variants:
            url = get_fandom_static_url(filename, wiki)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            }

            try:
                async with session.get(url, timeout=20, headers=headers) as response:
                    if response.status == 200:
                        clean_name = clean_filename(
                            filename.replace("File:", "").strip().replace(" ", "_")
                        )
                        filepath = os.path.join(output_dir, clean_name)

                        if not os.path.exists(filepath):
                            content = await response.read()
                            if content:
                                with open(filepath, "wb") as f:
                                    f.write(content)
                        pbar.update(1)
                        return  # Success
                    elif response.status == 404:
                        continue  # Try next variant
                    else:
                        break  # Other error, stop
            except Exception:
                break

        pbar.update(1)  # Final update if all variants fail


def process_page_worker(data):
    """CPU-intensive task: Cleans wikicode and extracts image links."""
    title, ns, text = data
    image_refs = []
    clean_text = None

    if ns == "0":  # Lore
        clean_text = mwparserfromhell.parse(text).strip_code()
        wikicode = mwparserfromhell.parse(text)
        for file_ref in wikicode.filter_wikilinks():
            if file_ref.title.startswith("File:"):
                image_refs.append(str(file_ref.title))
    elif ns == "6":  # File
        image_refs.append(title)

    return title, ns, clean_text, image_refs


def get_namespace(xml_path):
    with open(xml_path, "rb") as f:
        for event, elem in ET.iterparse(f, events=("start-ns",)):
            return f"{{{elem[1]}}}"
    return "{http://www.mediawiki.org/xml/export-0.11/}"


async def process_xml(xml_path, output_dir):
    xml_ns = get_namespace(xml_path)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "texts"), exist_ok=True)

    print("Counting pages in XML...")
    total_pages = 0
    with open(xml_path, "rb") as f:
        for event, elem in ET.iterparse(f, events=("end",), tag=f"{xml_ns}page"):
            total_pages += 1
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

    image_list = []
    print(f"Parallel processing {total_pages} pages across {cpu_count()} cores...")

    def page_generator():
        context = ET.iterparse(xml_path, events=("end",), tag=f"{xml_ns}page")
        for event, elem in context:
            title = elem.findtext(f".//{xml_ns}title")
            ns = elem.findtext(f".//{xml_ns}ns")
            revision = elem.find(f".//{xml_ns}revision")
            text = revision.findtext(f"{xml_ns}text") if revision is not None else ""
            yield (title, ns, text)
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

    with Pool(processes=cpu_count()) as pool:
        results = pool.imap_unordered(
            process_page_worker, page_generator(), chunksize=50
        )
        with tqdm(total=total_pages, desc="Processing Wiki", unit="page") as pbar:
            for title, ns, clean_text, image_refs in results:
                if clean_text:
                    safe_title = clean_filename(title)
                    with open(
                        os.path.join(output_dir, "texts", f"{safe_title}.txt"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        f.write(clean_text)
                if image_refs:
                    image_list.extend(image_refs)
                pbar.update(1)

    image_list = list(set(image_list))
    print(f"Found {len(image_list)} unique images. Starting direct static download...")

    async with aiohttp.ClientSession() as session:
        with tqdm(total=len(image_list), desc="Downloading Images", unit="img") as pbar:
            tasks = [
                download_image(session, img, os.path.join(output_dir, "images"), pbar)
                for img in image_list
            ]
            await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-threaded Wiki XML Downloader (Static URL Edition)"
    )
    parser.add_argument("xml_path", help="Path to the MediaWiki XML dump")
    parser.add_argument("--output", required=True, help="Output folder")
    args = parser.parse_args()
    asyncio.run(process_xml(args.xml_path, args.output))
