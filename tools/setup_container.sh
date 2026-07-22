#!/usr/bin/env bash
# Reproduce jm-test-arcadia from ubuntu:24.04 — vLLM-private build for gfx1260 (ROCm pip-SDK)
set -euo pipefail

# ── 1. Base toolchain ──────────────────────────────────────────────
# pkg-config + libdrm-dev were the two missing pieces vs. the original history
apt-get update && apt-get install -y --no-install-recommends \
    build-essential clang-19 lld ccache ninja-build cmake \
    git curl libcurl4-openssl-dev rsync ssh wget ca-certificates \
    python3 python3-pip python3-dev python3-venv \
    less vim libzstd-dev numactl libelf1 m4 jq \
    pkg-config libdrm-dev \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── 2. Python venv ─────────────────────────────────────────────────
export VENV=/opt/venv
python3 -m venv "$VENV"
export PATH="$VENV/bin:$PATH"
python -m pip install --upgrade pip setuptools PyYAML

# ── 3. torch + ROCm SDK (installed as wheels, NOT /opt/rocm) ────────
pip install --index https://rocm.genesis.amd.com/whl/gfx1260 torch "rocm[libraries,devel]"

# ── 4. Triton from source (needed to RUN vLLM; skip if build-only) ──
#     Requires SSH access to AMD-internal GitHub.
pip uninstall -y triton pytorch-triton pytorch-triton-rocm
pip install cmake requests conan
git clone git@github.com:AMD-Triton/triton-arcadia.git
git clone git@github.com:AMD-Lightning-Internal/llvm-project.git
SHA=$(jq -r '.llvm_hash' triton-arcadia/cmake/llvm-info.json)
( cd llvm-project && git checkout "$SHA" && mkdir -p build && cd build && \
  cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld;clang" \
        -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" ../llvm && ninja )
export LLVM_BUILD_DIR="$PWD/llvm-project/build"
( cd triton-arcadia && \
  LLVM_INCLUDE_DIRS="$LLVM_BUILD_DIR/include" \
  LLVM_LIBRARY_DIR="$LLVM_BUILD_DIR/lib" \
  LLVM_SYSPATH="$LLVM_BUILD_DIR" pip install -v -e . )

# ── 5. ROCm env for the build (the .jm_buildenv.sh fix) ─────────────
export ROCM_PATH="$(rocm-sdk path --root)"
export PATH="$(rocm-sdk path --bin):$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
export HIP_PATH="$ROCM_PATH"

# ── 6. Build vLLM (editable). --no-deps avoids the triton-pin re-check ──
cd /app/vllm-private
python setup.py develop --no-deps
