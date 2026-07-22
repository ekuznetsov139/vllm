# Source this to enter the vLLM runtime environment:
#   source /app/vllm-private/scripts/gfx1260_runtime_env.sh
# venv + wheel-based ROCm + FFM
source /opt/venv/bin/activate

# wheel-based ROCm toolchain
export ROCM_PATH="$(rocm-sdk path --root)"
export PATH="$(rocm-sdk path --bin):$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
export HIP_PATH="$ROCM_PATH"

# FFM / HSA functional-model runtime
export HSA_MODEL_LIB=/ffm/libhsakmtmodel.so
export HSA_MODEL_TOML=/ffm/ffm_config.toml
export HSA_MODEL_ARGS=ffm_enable_time_slicing
export HSA_MODEL_TOPOLOGY=/ffm/topology/arcadia
export HSA_MODEL_NUM_THREADS=448
export HSA_ENABLE_SDMA=0
export HSA_ENABLE_INTERRUPT=0

# disable CK
export ENABLE_CK=0

# vLLM detects arch via amdsmi, which on the FFM sim reports the HOST's real
# Force the true arch
export VLLM_ROCM_GCN_ARCH=gfx1260

# flydsl's SMEM_CAPACITY_MAP has no gfx1260 key
# FLYDSL_GPU_ARCH; gfx1250 has the identical 320KB LDS and is the same RDNA4 family
export FLYDSL_GPU_ARCH=gfx1250

# HF cache
export HF_HOME=/data/huggingface-cache

echo "[gfx1260_env] venv=$(which python)  ROCM_PATH=$ROCM_PATH"
echo "[gfx1260_env] FFM arcadia (gfx1260) simulator ready"
