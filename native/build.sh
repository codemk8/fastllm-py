#!/usr/bin/env bash
# Build native/libfastllm_kernels.so:
#   * fastllm's Marlin INT4 GEMM + GPTQ->Marlin repack kernels
#       (compiled directly from the unmodified fastllm source tree)
#   * an extracted, raw-pointer FP8-E4M3 block-128 GEMV + quantize kernel
#       (native/fastllm_fp8_block128.cu)
#
# No CMake, no cuBLAS -- links against the CUDA runtime only.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
FASTLLM="$REPO/fastllm"

NVCC="${NVCC:-/usr/local/cuda-12.9/bin/nvcc}"
ARCH="${ARCH:-sm_89}"

MARLIN_SRC="$FASTLLM/src/devices/cuda/linear/fastllm-marlin.cu"
FP8_SRC="$HERE/fastllm_fp8_block128.cu"
OUT="$HERE/libfastllm_kernels.so"

INCLUDES=(
    -I"$FASTLLM/include"
    -I"$FASTLLM/include/devices/cuda"
    -I"$FASTLLM/third_party/json11"
)

echo "[build] nvcc: $NVCC   arch: $ARCH"
echo "[build] compiling Marlin kernel ($MARLIN_SRC)"
"$NVCC" -c -Xcompiler -fPIC -arch="$ARCH" -O3 -std=c++17 \
    "${INCLUDES[@]}" "$MARLIN_SRC" -o "$HERE/fastllm-marlin.o"

echo "[build] compiling FP8 block128 kernels ($FP8_SRC)"
"$NVCC" -c -Xcompiler -fPIC -arch="$ARCH" -O3 -std=c++17 \
    "$FP8_SRC" -o "$HERE/fastllm_fp8_block128.o"

echo "[build] linking $OUT"
"$NVCC" -shared -Xcompiler -fPIC -arch="$ARCH" \
    "$HERE/fastllm-marlin.o" "$HERE/fastllm_fp8_block128.o" \
    -lcudart -o "$OUT"

rm -f "$HERE/fastllm-marlin.o" "$HERE/fastllm_fp8_block128.o"
echo "[build] done: $OUT"
ls -l "$OUT"
