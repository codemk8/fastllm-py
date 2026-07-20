// Standalone, raw-pointer extraction of fastllm's FP8-E4M3 block-128 GEMV +
// weight-quantize kernels, with no dependency on fastllm::Data, cuBLAS, or the
// monolithic fastllm-cuda.cu global state (see docs/fastllm-internals.md §7.2).
//
// All kernel/launcher bodies below are copied verbatim (or with the minimal
// change of dropping the fastllm::Data host wrapper) from:
//   fastllm/src/devices/cuda/linear/fastllm-linear-fp8.cu
// with the origin line noted at each block. Do NOT edit the numeric behaviour
// of these kernels -- they must stay bit-identical to fastllm so that weights
// quantized here decode the same way the upstream engine decodes them.

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <algorithm>

// ---------------------------------------------------------------------------
// union_half4: origin fastllm/include/devices/cuda/fastllm-cuda.cuh:46-53
// ---------------------------------------------------------------------------
typedef union __align__(16) _union_half_4 {
    uint2 in;
    half out[4];
    half2 out2[2];
    __device__ _union_half_4() {
      // Do nothing
    }
} union_half4;

// ---------------------------------------------------------------------------
// FastllmFp8QuantLoad: origin fastllm-linear-fp8.cu:24-38
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ float FastllmFp8QuantLoad(const T *input, size_t index);

template <>
__device__ __forceinline__ float FastllmFp8QuantLoad(const half *input, size_t index) {
    return __half2float(input[index]);
}

template <>
__device__ __forceinline__ float FastllmFp8QuantLoad(const __nv_bfloat16 *input, size_t index) {
    return __bfloat162float(input[index]);
}

// ---------------------------------------------------------------------------
// Quantize kernel (T weight -> FP8_E4M3_BLOCK_128 interleaved layout).
// origin fastllm-linear-fp8.cu:40-85
// ---------------------------------------------------------------------------
template <typename T>
__global__ void FastllmQuantizeLinearWeightFP8E4M3Block128Kernel(
        const T *input, uint8_t *output, int rows, int columns,
        int packedRowBytes, int blocksPerRow, int totalBlocks) {
    constexpr int warpsPerBlock = 8;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    int tile = blockIdx.x * warpsPerBlock + warp;
    const int tileStride = gridDim.x * warpsPerBlock;

    for (; tile < totalBlocks; tile += tileStride) {
        int row = tile / blocksPerRow;
        int columnBlock = tile - row * blocksPerRow;
        int columnStart = columnBlock * 128;
        float localMax = 0.0f;
#pragma unroll
        for (int i = lane; i < 128; i += 32) {
            float value = FastllmFp8QuantLoad(
                input, (size_t)row * columns + columnStart + i);
            localMax = fmaxf(localMax, fabsf(value));
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            localMax = fmaxf(
                localMax, __shfl_down_sync(0xffffffff, localMax, offset));
        }
        float scale = __shfl_sync(0xffffffff, localMax, 0) / 448.0f;
        if (!(scale > 0.0f)) {
            scale = 1.0f;
        }

        uint8_t *packedBlock = output + (size_t)row * packedRowBytes +
                               (size_t)columnBlock * (128 + sizeof(float));
        if (lane == 0) {
            *reinterpret_cast<float*>(packedBlock + 128) = scale;
        }
        float inverseScale = 1.0f / scale;
#pragma unroll
        for (int i = lane; i < 128; i += 32) {
            float value = FastllmFp8QuantLoad(
                input, (size_t)row * columns + columnStart + i);
            packedBlock[i] = (uint8_t)__nv_cvt_float_to_fp8(
                value * inverseScale, __NV_SATFINITE, __NV_E4M3);
        }
    }
}

// ---------------------------------------------------------------------------
// GEMV kernel: one warp per output row (faster variant).
// origin fastllm-linear-fp8.cu:705-775
// ---------------------------------------------------------------------------
template <int WARPS_PER_BLOCK, int PART>
__global__ void FastllmGemvHalfFP8E4M3Block128KernelWarpMultiRow(
        const half * __restrict__ A, const uint8_t * __restrict__ B, half * __restrict__ C,
        const half * __restrict__ bias, int m, int k, int perRow) {
    const int warpId = threadIdx.x >> 5;
    const int laneId = threadIdx.x & 31;
    const int st = blockIdx.x * WARPS_PER_BLOCK + warpId;
    if (st >= k) return;

    const float magicScaleConstant = exp2f(8.0f);
    const int block_size = 128;
    const int numBlocks = (m - 1) / block_size + 1;
    const uint8_t *baseB = B + (size_t)st * perRow;

    float acc[PART];
#pragma unroll
    for (int x = 0; x < PART; x++) acc[x] = 0.0f;

    for (int blk = 0; blk < numBlocks; blk++) {
        const int blkStart = blk * block_size;
        const uint8_t *blkData = baseB + (size_t)blk * (block_size + sizeof(float));
        const float blkScale = *(const float*)(blkData + block_size);
        const int localOff = laneId * 4;
        const int i = blkStart + localOff;

        if (i + 3 < m) {
            uint32_t bb = *(const uint32_t*)(blkData + localOff);
            half2 B01 = make_half2(__short_as_half((short)((((bb >> 0)  & 0x80) << 8) | (((bb >> 0)  & 0x7F) << 7))),
                                   __short_as_half((short)((((bb >> 8)  & 0x80) << 8) | (((bb >> 8)  & 0x7F) << 7))));
            half2 B23 = make_half2(__short_as_half((short)((((bb >> 16) & 0x80) << 8) | (((bb >> 16) & 0x7F) << 7))),
                                   __short_as_half((short)((((bb >> 24) & 0x80) << 8) | (((bb >> 24) & 0x7F) << 7))));
#pragma unroll
            for (int x = 0; x < PART; x++) {
                union_half4 regA;
                regA.in = *reinterpret_cast<const uint2*>(A + (size_t)x * m + i);
                half2 p = __hadd2(__hmul2(regA.out2[0], B01), __hmul2(regA.out2[1], B23));
                acc[x] += (__half2float(p.x) + __half2float(p.y)) * blkScale;
            }
        } else {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                if (i + j >= m) break;
                uint8_t bv = blkData[localOff + j];
                float bf = __half2float(__short_as_half((short)(((bv & 0x80) << 8) | ((bv & 0x7F) << 7)))) * blkScale;
#pragma unroll
                for (int x = 0; x < PART; x++) {
                    acc[x] += __half2float(A[(size_t)x * m + i + j]) * bf;
                }
            }
        }
    }

#pragma unroll
    for (int x = 0; x < PART; x++) {
        float v = acc[x];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, off);
        }
        acc[x] = v;
    }

    if (laneId == 0) {
#pragma unroll
        for (int x = 0; x < PART; x++) {
            float r = acc[x] * magicScaleConstant;
            if (bias != nullptr) r += (float)bias[st];
            C[st + (size_t)k * x] = (half)r;
        }
    }
}

// ---------------------------------------------------------------------------
// fp16 launcher: origin fastllm-linear-fp8.cu:777-813
// ---------------------------------------------------------------------------
static void LaunchFastllmGemmFp16FP8E4M3Block128(half *input, uint8_t *weight, half *output, half *bias, int n, int m, int k, int perRow) {
    constexpr int W = 8; // 8 warps (256 threads) per block
    const int grid = (k + W - 1) / W;
#define FASTLLM_FP8_B128_WARP_LAUNCH(PARTVAL, AOFF, COFF) \
    FastllmGemvHalfFP8E4M3Block128KernelWarpMultiRow<W, PARTVAL> <<< grid, W * 32 >>>( \
        input + (AOFF) * m, weight, output + (COFF) * k, bias, m, k, perRow)

    switch (n) {
        case 1:  FASTLLM_FP8_B128_WARP_LAUNCH(1, 0, 0);  return;
        case 2:  FASTLLM_FP8_B128_WARP_LAUNCH(2, 0, 0);  return;
        case 3:  FASTLLM_FP8_B128_WARP_LAUNCH(3, 0, 0);  return;
        case 4:  FASTLLM_FP8_B128_WARP_LAUNCH(4, 0, 0);  return;
        case 5:  FASTLLM_FP8_B128_WARP_LAUNCH(5, 0, 0);  return;
        case 6:  FASTLLM_FP8_B128_WARP_LAUNCH(6, 0, 0);  return;
        case 7:  FASTLLM_FP8_B128_WARP_LAUNCH(7, 0, 0);  return;
        case 8:  FASTLLM_FP8_B128_WARP_LAUNCH(8, 0, 0);  return;
        default: break;
    }

    {
        int i = 0;
        for (; i + 7 < n; i += 8) {
            FASTLLM_FP8_B128_WARP_LAUNCH(8, i, i);
        }
        for (; i + 3 < n; i += 4) {
            FASTLLM_FP8_B128_WARP_LAUNCH(4, i, i);
        }
        for (; i + 1 < n; i += 2) {
            FASTLLM_FP8_B128_WARP_LAUNCH(2, i, i);
        }
        for (; i < n; i++) {
            FASTLLM_FP8_B128_WARP_LAUNCH(1, i, i);
        }
        return;
    }
#undef FASTLLM_FP8_B128_WARP_LAUNCH
}

// ---------------------------------------------------------------------------
// extern "C" ctypes entry points (new thin shims; no fastllm dependency).
// ---------------------------------------------------------------------------
extern "C" {

// Quantize a [rows, columns] fp16 (isBf16==0) or bf16 (isBf16!=0) weight matrix,
// already resident on the current CUDA device, into the interleaved
// FP8_E4M3_BLOCK_128 layout at `output`.
//   output size = rows * (columns + (columns/128)*sizeof(float)) bytes.
// columns must be a multiple of 128. Returns 0 on success, non-zero on error.
int fastllm_fp8_block128_quantize(const void *input, void *output,
                                  int rows, int columns, int isBf16) {
    if (rows <= 0 || columns <= 0 || (columns % 128) != 0) {
        return 1;
    }
    int blocksPerRow = columns / 128;
    int packedRowBytes = columns + blocksPerRow * (int)sizeof(float);
    int totalBlocks = rows * blocksPerRow;

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device) != cudaSuccess) {
        cudaGetLastError();
        return 2;
    }
    int grid = std::max(1, std::min(totalBlocks, props.multiProcessorCount * 8));
    if (isBf16 == 0) {
        FastllmQuantizeLinearWeightFP8E4M3Block128Kernel<<<grid, 256>>>(
            (const half *)input, (uint8_t *)output, rows, columns,
            packedRowBytes, blocksPerRow, totalBlocks);
    } else {
        FastllmQuantizeLinearWeightFP8E4M3Block128Kernel<<<grid, 256>>>(
            (const __nv_bfloat16 *)input, (uint8_t *)output, rows, columns,
            packedRowBytes, blocksPerRow, totalBlocks);
    }
    cudaError_t st = cudaGetLastError();
    if (st == cudaSuccess) st = cudaDeviceSynchronize();
    if (st != cudaSuccess) {
        printf("fastllm_fp8_block128_quantize kernel error: %s\n", cudaGetErrorString(st));
        return 3;
    }
    return 0;
}

// packedRowBytes for a given input-feature count m.
int fastllm_fp8_block128_packed_row_bytes(int m) {
    return m + (m / 128) * (int)sizeof(float);
}

// GEMV/GEMM: C[n, k] = A[n, m] @ dequant(W)[k, m]^T   (+ bias[k] if non-null).
//   A:      device half, row-major [n, m]
//   weight: device uint8, k rows of `perRow` bytes each in FP8_E4M3_BLOCK_128 layout
//   C:      device half, row-major [n, k]
//   bias:   device half [k] or nullptr
//   perRow: bytes per weight row = m + (m/128)*4  (see *_packed_row_bytes)
// Runs on the default stream and synchronizes. Returns 0 on success.
int fastllm_fp8_block128_gemv_fp16(const void *A, const void *weight, void *C,
                                   const void *bias, int n, int m, int k, int perRow) {
    if (n <= 0 || m <= 0 || k <= 0 || (m % 128) != 0) {
        return 1;
    }
    LaunchFastllmGemmFp16FP8E4M3Block128((half *)A, (uint8_t *)weight, (half *)C,
                                         (half *)bias, n, m, k, perRow);
    cudaError_t st = cudaGetLastError();
    if (st == cudaSuccess) st = cudaDeviceSynchronize();
    if (st != cudaSuccess) {
        printf("fastllm_fp8_block128_gemv_fp16 kernel error: %s\n", cudaGetErrorString(st));
        return 2;
    }
    return 0;
}

} // extern "C"
