# 3. Extract the checkpoint using pigz (Multi-threaded gzip)
if [ -f "hf_ckpt.tar.gz" ]; then
    echo "--- Extracting Checkpoint ---"
    # Using pv to see progress and pigz for speed
    pv hf_ckpt.tar.gz | tar -I pigz -xvf -
else
    echo "!!! hf_ckpt.tar.gz not found in current directory !!!"
fi