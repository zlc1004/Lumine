#!/bin/bash

# Check if we have enough arguments
if [ "$#" -lt 3 ]; then
    echo "Usage: ./upload.sh \"ssh root@ip -p 22 -i key\" <file> <remote_path>"
    exit 1
fi

SSH_STRING="$1"
FILE_TO_UPLOAD="$2"
REMOTE_DEST="$3"

# Parse components
USER_HOST=$(echo "$SSH_STRING" | grep -oE '[a-zA-Z0-9._%+-]+@[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+')
PORT=$(echo "$SSH_STRING" | grep -oP '(?<=-p\s)\d+')
ID_FILE=$(echo "$SSH_STRING" | grep -oP '(?<=-i\s)[^\s]+')
PORT=${PORT:-22}
BASENAME=$(basename "$FILE_TO_UPLOAD")

echo "----------------------------------------"
echo "Target: $USER_HOST"
echo "Splitting & Uploading: $BASENAME"
echo "----------------------------------------"
# 1. Check if chunks already exist to skip splitting
echo "----------------------------------------"
echo "[1/3] Checking for existing chunks..."

# Check if there are any files matching the pattern in the chunks folder
CHUNKS_EXIST=false
if [ -d "./chunks_tmp" ]; then
    # Using 'find' to see if at least one part file exists
    if [ -n "$(find ./chunks_tmp -maxdepth 1 -name 'part_*' -print -quit)" ]; then
        CHUNKS_EXIST=true
    fi
fi

if [ "$CHUNKS_EXIST" = true ]; then
    echo ">> Found existing chunks in ./chunks_tmp. Skipping split step."
else
    echo ">> No chunks found. Splitting file into 500MB pieces..."
    mkdir -p ./chunks_tmp
    split -b 500M -d "$FILE_TO_UPLOAD" ./chunks_tmp/part_
fi

# 2. Upload chunks in parallel (using 8 parallel streams)
# Added --ignore-existing so it doesn't re-upload a chunk if the transfer was interrupted
echo "[2/3] Uploading chunks in parallel..."
find ./chunks_tmp -type f | xargs -P 8 -I {} rsync -v -e "ssh -p $PORT -i $ID_FILE -o StrictHostKeyChecking=no -c aes128-gcm@openssh.com" --no-owner --no-group --no-perms --ignore-existing {} "$USER_HOST":"$REMOTE_DEST"

# 3. Combine chunks on the remote server
echo "[3/3] Recombining chunks on remote server..."
ssh -p "$PORT" -i "$ID_FILE" "$USER_HOST" "cd $REMOTE_DEST && cat part_* > $BASENAME && rm part_*"

# Optional: Cleanup local chunks only if recombination succeeded
if [ $? -eq 0 ]; then
    rm -rf ./chunks_tmp
    echo "----------------------------------------"
    echo "Upload and Recombination Complete!"
else
    echo "!! Recombination failed on remote server. Local chunks preserved."
fi