#!/usr/bin/env bash
# gfx1260 dependency + aiter setup — run ON TOP of base container
set -euo pipefail

VLLM_DIR=/app/vllm-private
GENESIS_INDEX=https://rocm.genesis.amd.com/whl/gfx1260

# venv + wheel-based ROCm env ─────────────────────────────────
source /opt/venv/bin/activate
export ROCM_PATH="$(rocm-sdk path --root)"
export PATH="$(rocm-sdk path --bin):$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
export HIP_PATH="$ROCM_PATH"

# Rust frontend ────────────────────────
pip install -r "$VLLM_DIR/requirements/build/rust.txt"

# Runtime deps, protecting the ROCm torch/triton stack ──────
TORCH_BEFORE=$(pip show torch 2>/dev/null   | awk -F': ' '/^Version/{print $2}')
TRITON_BEFORE=$(pip show triton 2>/dev/null | awk -F': ' '/^Version/{print $2}')

pip install -r "$VLLM_DIR/requirements/rocm.txt"

TORCH_AFTER=$(pip show torch 2>/dev/null   | awk -F': ' '/^Version/{print $2}')
if [ -n "$TORCH_BEFORE" ] && [ "$TORCH_AFTER" != "$TORCH_BEFORE" ]; then
  echo "[deps] torch clobbered ($TORCH_BEFORE -> $TORCH_AFTER); restoring ROCm build"
  pip install --force-reinstall --no-deps --index-url "$GENESIS_INDEX" "torch==$TORCH_BEFORE"
fi

TRITON_AFTER=$(pip show triton 2>/dev/null | awk -F': ' '/^Version/{print $2}')
if [ -n "$TRITON_BEFORE" ] && [ "$TRITON_AFTER" != "$TRITON_BEFORE" ] && [ -d /triton-arcadia ]; then
  echo "[deps] triton clobbered ($TRITON_BEFORE -> $TRITON_AFTER); restoring custom triton-arcadia"
  pip uninstall -y triton pytorch-triton pytorch-triton-rocm || true
  ( cd /triton-arcadia && pip install -e . --no-build-isolation --no-deps )
fi

# aiter: CK-free rebuild
# Remove stale CK-baked JIT artifacts
if [ -d /root/aiter ]; then
  rm -f  /root/aiter/aiter/jit/module_*.so
  rm -rf /root/aiter/aiter/jit/build /root/aiter/amd_aiter.egg-info /root/.aiter /tmp/aiter_configs
  ( cd /root/aiter && AITER_USE_SYSTEM_TRITON=1 ENABLE_CK=0 PREBUILD_KERNELS=0 python setup.py develop )
fi

# Build vLLM (editable) for gfx1260
cd "$VLLM_DIR"
VLLM_TARGET_DEVICE=rocm AITER_USE_SYSTEM_TRITON=1 \
GPU_ARCHS=gfx1260 PYTORCH_ROCM_ARCH=gfx1260 MAX_JOBS="$(nproc)" \
  python -m pip install -e . --no-build-isolation --no-deps

echo
echo "[deps] Done. Before serving, export the runtime env 'setup_container.sh'"
