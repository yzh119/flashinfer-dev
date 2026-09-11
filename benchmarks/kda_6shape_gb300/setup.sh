#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}

# Use an environment with CUDA-enabled PyTorch already installed.
packages=(setuptools ninja numpy packaging requests click tqdm tabulate einops
    nvidia-ml-py 'apache-tvm-ffi>=0.1.6,<0.2' 'cupti-python>=13')
if command -v uv >/dev/null; then
    uv pip install --python "$PYTHON" "${packages[@]}"
else
    "$PYTHON" -m pip install "${packages[@]}"
fi
mkdir -p "$HERE/.deps"
checkout() {
    local name=$1 url=$2 revision=$3
    local dest="$HERE/.deps/$name"
    if [[ ! -d "$dest/.git" ]]; then
        git init "$dest"
        git -C "$dest" remote add origin "$url"
        git -C "$dest" fetch --depth 1 origin "$revision"
        git -C "$dest" checkout --detach "$revision"
    fi
    test "$(git -C "$dest" rev-parse HEAD)" = "$revision"
}
checkout flashinfer https://github.com/flashinfer-ai/flashinfer.git 9f1f3ea7807799a4b01face909b64ddd416ebe18
checkout FlashKDA https://github.com/MoonshotAI/FlashKDA.git 1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b
git -C "$HERE/.deps/flashinfer" submodule update --init --depth 1 3rdparty/cutlass 3rdparty/spdlog
git -C "$HERE/.deps/FlashKDA" submodule update --init --depth 1 cutlass

# Same source-data links as an editable FlashInfer installation; no package replacement.
mkdir -p "$HERE/.deps/flashinfer/flashinfer/data"
for name in cutlass spdlog; do
    ln -sfn "../../3rdparty/$name" "$HERE/.deps/flashinfer/flashinfer/data/$name"
done
for name in csrc include; do
    ln -sfn "../../$name" "$HERE/.deps/flashinfer/flashinfer/data/$name"
done
cd "$HERE/.deps/FlashKDA"
FLASH_KDA_CUDA_ARCHS=103a NVCC_THREADS=${NVCC_THREADS:-4} MAX_JOBS=${MAX_JOBS:-4} \
    "$PYTHON" setup.py build_ext --inplace
echo "Ready: bash $HERE/run.sh"
