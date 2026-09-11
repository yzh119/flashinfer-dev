#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export FLASHINFER_CUDA_ARCH_LIST=10.3a
export MAX_JOBS=${MAX_JOBS:-4}
export PYTHONPATH="$HERE/.deps/flashinfer:$HERE/.deps/FlashKDA"
"${PYTHON:-python}" -u "$HERE/benchmark.py" "$@" 2>&1 | tee "$HERE/run.log"
