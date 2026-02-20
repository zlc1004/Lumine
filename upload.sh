#!/bin/bash

# Check if we have enough arguments
if [ "$#" -lt 3 ]; then
    echo "Usage: ./upload.sh \"ssh root@ip -p 22 -i key\" <file> <remote_path>"
    exit 1
fi

SSH_STRING="$1"
FILE_TO_UPLOAD="$2"
REMOTE_DEST="$3"

# Parse the components using grep/regex
# 1. Look for user@host
USER_HOST=$(echo "$SSH_STRING" | grep -oE '[a-zA-Z0-9._%+-]+@[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+')
# 2. Look for the port number following -p
PORT=$(echo "$SSH_STRING" | grep -oP '(?<=-p\s)\d+')
# 3. Look for the identity file following -i
ID_FILE=$(echo "$SSH_STRING" | grep -oP '(?<=-i\s)[^\s]+')

# Default to port 22 if none found
PORT=${PORT:-22}

echo "----------------------------------------"
echo "Target: $USER_HOST"
echo "Port:   $PORT"
echo "Key:    $ID_FILE"
echo "----------------------------------------"

# Execute rsync with the parsed info
rsync -avzP -e "ssh -p $PORT -i $ID_FILE -o StrictHostKeyChecking=no" "$FILE_TO_UPLOAD" "$USER_HOST":"$REMOTE_DEST"