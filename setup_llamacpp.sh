#!/bin/bash
# Setup llama.cpp for GGUF conversion and quantization

echo "Cloning llama.cpp..."
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

echo "Building llama.cpp..."
# Use CUDA for faster quantization if possible, otherwise standard make
if command -v nvcc &> /dev/null
then
    echo "CUDA detected, building with CUDA support..."
    cmake -B build -DGGML_CUDA=ON
    cmake --build build --config Release -j $(nproc)
    # Create symlinks to the binaries in the llama.cpp root for the quantize script
    ln -s build/bin/llama-quantize . 2>/dev/null
    ln -s build/bin/llama-cli . 2>/dev/null
else
    echo "CUDA not detected, building with CPU only..."
    make -j $(nproc)
fi

echo "Installing python dependencies..."
# Use uv if available, otherwise standard pip
if command -v uv &> /dev/null
then
    uv pip install -r requirements.txt
else
    pip install -r requirements.txt
fi

echo "llama.cpp setup complete."
