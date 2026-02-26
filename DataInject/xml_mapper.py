import lxml.etree as ET
import json
import re
import os
import argparse


def parse_xml_mapping(xml_path, output_json):
    """
    Parses MediaWiki XML to:
    1. Build a redirect map for Files (Namespace 6).
    2. Extract image links from Infoboxes in Namespace 0.
    """
    ns_map = {"mw": "http://www.mediawiki.org/xml/export-0.11/"}

    file_redirects = {}  # "File:Old Name.png" -> "Target Name.png"
    item_to_image = {}  # "Item Name" -> "Image Name.png"

    print(f"Starting XML parse of {xml_path}...")

    # Use iterparse to be memory efficient
    context = ET.iterparse(
        xml_path, events=("end",), tag="{http://www.mediawiki.org/xml/export-0.11/}page"
    )

    page_count = 0
    mapping_count = 0
    redirect_count = 0

    for event, elem in context:
        page_count += 1
        if page_count % 100000 == 0:
            print(f"Processed {page_count} pages...")

        ns = elem.find("mw:ns", ns_map).text
        title = elem.find("mw:title", ns_map).text

        # Case 1: Namespace 6 (File) - Check for redirects
        if ns == "6":
            redirect_elem = elem.find("mw:redirect", ns_map)
            if redirect_elem is not None:
                target = redirect_elem.attrib.get("title", "")
                if target.startswith("File:"):
                    target = target[5:]  # Strip "File:"
                source = title
                if source.startswith("File:"):
                    source = source[5:]  # Strip "File:"
                file_redirects[source] = target
                redirect_count += 1

        # Case 2: Namespace 0 (Main) - Extract Infobox image
        elif ns == "0":
            revision = elem.find("mw:revision", ns_map)
            if revision is not None:
                text_elem = revision.find("mw:text", ns_map)
                if text_elem is not None and text_elem.text:
                    content = text_elem.text
                    # Match |image = filename.png
                    # Handles various spacing and line endings
                    match = re.search(r"\|\s*image\s*=\s*([^|\n}]+)", content)
                    if match:
                        img_val = match.group(1).strip()
                        if img_val:
                            item_to_image[title] = img_val
                            mapping_count += 1

        # Memory management
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    print(
        f"Finished parsing. Found {redirect_count} file redirects and {mapping_count} item-image links."
    )

    # Final Resolve: Apply redirects to item_to_image
    resolved_mapping = {}
    for item, img in item_to_image.items():
        # Clean up image name (sometimes users include "File:" prefix in templates)
        if img.startswith("File:"):
            img = img[5:]

        # Follow redirect if it exists
        final_img = file_redirects.get(img, img)
        resolved_mapping[item] = final_img

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(resolved_mapping, f, indent=2)

    print(f"Successfully saved resolved mapping to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Item-to-Image mapping from MediaWiki XML"
    )
    parser.add_argument(
        "--xml",
        default="../assets/gensinimpact_pages_current.xml",
        help="Path to the XML dump",
    )
    parser.add_argument(
        "--output", default="xml_mapping.json", help="Path to output JSON"
    )
    args = parser.parse_args()

    if os.path.exists(args.xml):
        parse_xml_mapping(args.xml, args.output)
    else:
        print(f"Error: XML file not found at {args.xml}")
