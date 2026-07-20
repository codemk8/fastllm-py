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

# Inject a stream-accepting GEMM entry into a COPY of the upstream file
# (upstream tree is never modified). See native/marlin_stream_entry.inc.
MARLIN_PATCHED="$HERE/_marlin_patched.cu"
echo "[build] patching Marlin copy with stream entry"
python3 - "$MARLIN_SRC" "$HERE/marlin_stream_entry.inc" "$MARLIN_PATCHED" <<'PY'
import sys
src, inc, out = sys.argv[1:4]
text = open(src).read()
snippet = open(inc).read()
marker = "#undef TORCH_CHECK"
assert marker in text, "expected #undef TORCH_CHECK marker in fastllm-marlin.cu"
assert "FastllmCudaMarlinHalfInt4GemmStream" not in text, "already patched?"
text = text.replace(marker, snippet + marker, 1)
open(out, "w").write(text)
print("  -> injected FastllmCudaMarlinHalfInt4GemmStream")
PY

echo "[build] compiling Marlin kernel ($MARLIN_PATCHED)"
"$NVCC" -c -Xcompiler -fPIC -arch="$ARCH" -O3 -std=c++17 \
    "${INCLUDES[@]}" "$MARLIN_PATCHED" -o "$HERE/fastllm-marlin.o"

echo "[build] compiling FP8 block128 kernels ($FP8_SRC)"
"$NVCC" -c -Xcompiler -fPIC -arch="$ARCH" -O3 -std=c++17 \
    "$FP8_SRC" -o "$HERE/fastllm_fp8_block128.o"

echo "[build] linking $OUT"
"$NVCC" -shared -Xcompiler -fPIC -arch="$ARCH" \
    "$HERE/fastllm-marlin.o" "$HERE/fastllm_fp8_block128.o" \
    -lcudart -o "$OUT"

rm -f "$HERE/fastllm-marlin.o" "$HERE/fastllm_fp8_block128.o" "$MARLIN_PATCHED"
echo "[build] done: $OUT"
ls -l "$OUT"
