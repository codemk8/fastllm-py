// Injected at build time by native/build.sh into a *copy* of fastllm's
// fastllm-marlin.cu (the upstream tree is never modified). Identical to
// FastllmCudaMarlinHalfInt4Gemm except it takes an explicit CUDA stream
// pointer instead of hardcoding the legacy default stream (0). Running
// Marlin on a real stream removes the device-wide barrier that stream 0
// imposes on blocking streams, and — critically — makes the GEMM
// capturable into a CUDA graph. See docs/next-optimizations.md #1.
extern "C" bool FastllmCudaMarlinHalfInt4GemmStream(const void *a, const uint32_t *b_q_weight,
                                                    const void *b_scales, const uint32_t *b_zeros,
                                                    void *c, int size_m, int size_n, int size_k,
                                                    int group_size, int *workspace, void *stream_ptr) {
    if (!FastllmCudaMarlinCurrentDeviceSupported()) {
        return false;
    }
    if (size_m <= 0 || size_n <= 0 || size_k <= 0 ||
        (group_size != 32 && group_size != 128) ||
        size_k % group_size != 0) {
        return false;
    }
    if (size_n % marlin::min_thread_n != 0 || size_k % marlin::min_thread_k != 0) {
        return false;
    }
    int dev = 0;
    cudaGetDevice(&dev);
    int num_groups = size_k / group_size;
    marlin::marlin_mm<half>(
        (const half*)a, b_q_weight, (half*)c, nullptr,
        const_cast<half*>((const half*)b_scales), const_cast<uint32_t*>(b_zeros),
        nullptr, nullptr, nullptr, size_m, size_n, size_k, workspace,
        vllm::kU4, false, false, true, num_groups, group_size, dev, (cudaStream_t)stream_ptr,
        -1, -1, -1, marlin::max_par, false, false);
    return true;
}

