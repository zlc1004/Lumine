#!/bin/bash
set -e

# 1. Update requirements
echo "[1/4] Updating requirements..."
./.conda/bin/pip install -r requirements.txt

echo "[2/4] Extracting metadata and mapping XML..."
echo "  -> Running metadata_extractor.py"
./.conda/bin/python metadata_extractor.py output --output wiki_metadata.json
echo "  -> Running xml_mapper.py"
./.conda/bin/python xml_mapper.py --xml ../assets/gensinimpact_pages_current.xml --output xml_mapping.json

# 2. Run NeMo Curator Pipeline
echo "[3/4] Running NeMo Curator Cleaning Pipeline..."
./.conda/bin/python jsonl_converter.py wiki_metadata.json --output genshin_clean_lore.jsonl

# 3. Run Multi-modal Mapping
echo "[4/4] Mapping cleaned lore to image assets..."
# Note: map_assets.py currently uses wiki_metadata.json for images
# We want it to use the CLEANED lore from genshin_clean_lore.jsonl
# I'll update map_assets.py to support reading from the cleaned JSONL
./.conda/bin/python map_assets.py wiki_metadata.json --output final_veomni_training.jsonl --cleaned_jsonl genshin_clean_lore.jsonl --xml_mapping xml_mapping.json

echo "DONE! Final dataset saved to: final_veomni_training.jsonl"
