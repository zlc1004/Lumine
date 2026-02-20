# 3. Extract the checkpoint using pigz (Multi-threaded gzip)
if [ -f "hf_ckpt.tar.gz" ]; then
    echo "--- Extracting Checkpoint ---"
    # Using pv to see progress and pigz for speed
    tar -I pigz -xf hf_ckpt.tar.gz
else
    echo "!!! hf_ckpt.tar.gz not found in current directory !!!"
fi