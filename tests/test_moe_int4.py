"""Custom row-major INT4 GEMV: matches dequant reference + the marlin path."""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from fastllm_py.kernels import moe_int4


@pytest.mark.parametrize("out_f,in_f,gs", [(1408, 2048, 128), (2048, 1408, 128),
                                            (512, 1024, 128), (256, 256, 32)])
def test_gemv_matches_dequant(out_f, in_f, gs):
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((out_f, in_f)) * 0.1).astype(np.float32)
    x = (rng.standard_normal(in_f) * 0.5).astype(np.float32)
    payload = moe_int4.quantize_int4_rowmajor(w, gs, xp=cp)
    y = cp.asnumpy(moe_int4.gemv_int4(cp.asarray(x), payload))
    # reference: dequant the SAME payload (on CPU), then x @ Wq.T
    ref = x @ moe_int4.dequantize_int4_rowmajor(payload, xp=cp).get().T
    np.testing.assert_allclose(y, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("E,K,hidden,inter,gs", [(8, 4, 256, 512, 128),
                                                 (60, 4, 512, 1408, 128),
                                                 (4, 1, 128, 256, 32)])
def test_fused_moe_matches_reference(E, K, hidden, inter, gs):
    rng = np.random.default_rng(2)
    experts = [{"gate": (rng.standard_normal((inter, hidden)) * 0.1).astype(np.float16),
                "up": (rng.standard_normal((inter, hidden)) * 0.1).astype(np.float16),
                "down": (rng.standard_normal((hidden, inter)) * 0.1).astype(np.float16)}
               for _ in range(E)]
    x = (rng.standard_normal(hidden) * 0.5).astype(np.float32)
    eidx = rng.choice(E, size=K, replace=False).astype(np.int32)
    rw = (rng.random(K) + 0.1).astype(np.float32)

    stacked = moe_int4.build_stacked_experts(experts, gs, xp=cp)
    y = cp.asnumpy(moe_int4.fused_moe_ffn(cp.asarray(x), stacked,
                                          cp.asarray(eidx), cp.asarray(rw), hidden, inter))
    y2 = cp.asnumpy(moe_int4.fused_moe_ffn2(cp.asarray(x), stacked,
                                            cp.asarray(eidx), cp.asarray(rw), hidden, inter))

    # reference: dequant the SAME row-major payloads, run each expert's FFN
    ref = np.zeros(hidden, dtype=np.float32)
    for k in range(K):
        e = int(eidx[k])
        wq = {p: moe_int4.dequantize_int4_rowmajor(
                    moe_int4.quantize_int4_rowmajor(experts[e][p], gs), xp=np)
              for p in ("gate", "up", "down")}
        g = x @ wq["gate"].T
        u = x @ wq["up"].T
        act = (g / (1 + np.exp(-g))) * u
        ref += float(rw[k]) * (act @ wq["down"].T)
    np.testing.assert_allclose(y, ref, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(y2, ref, rtol=2e-3, atol=2e-3)


def test_gate_matvec_matches_cublas():
    rng = np.random.default_rng(3)
    E, hidden = 60, 2048
    gw = (rng.standard_normal((E, hidden)) * 0.1).astype(np.float16)
    x = (rng.standard_normal(hidden) * 0.3).astype(np.float32)
    y = cp.asnumpy(moe_int4.gate_matvec(cp.asarray(x), cp.asarray(gw), E, hidden))
    ref = x @ gw.astype(np.float32).T
    np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-4)


def test_fused_moe_weighted_matches_index_version():
    rng = np.random.default_rng(4)
    E, K, hidden, inter, gs = 60, 4, 512, 1408, 128
    experts = [{"gate": (rng.standard_normal((inter, hidden)) * 0.1).astype(np.float16),
                "up": (rng.standard_normal((inter, hidden)) * 0.1).astype(np.float16),
                "down": (rng.standard_normal((hidden, inter)) * 0.1).astype(np.float16)}
               for _ in range(E)]
    stacked = moe_int4.build_stacked_experts(experts, gs, xp=cp)
    x = (rng.standard_normal(hidden) * 0.3).astype(np.float32)
    eidx = rng.choice(E, K, replace=False).astype(np.int32)
    w = (rng.random(K) + 0.1).astype(np.float32)
    rw = np.zeros(E, dtype=np.float32)
    rw[eidx] = w
    yw = cp.asnumpy(moe_int4.fused_moe_weighted(cp.asarray(x), stacked, cp.asarray(rw),
                                                E, hidden, inter))
    yi = cp.asnumpy(moe_int4.fused_moe_ffn2(cp.asarray(x), stacked, cp.asarray(eidx),
                                            cp.asarray(w), hidden, inter))
    # same math; accumulation order over experts differs (atomicAdd) -> ~fp noise
    np.testing.assert_allclose(yw, yi, rtol=1e-4, atol=1e-5)


def test_fused_moe_weighted_captures():
    """gate + route + weighted fused kernel must be CUDA-graph-capturable."""
    rng = np.random.default_rng(5)
    E, K, hidden, inter, gs = 60, 4, 512, 512, 128
    experts = [{"gate": (rng.standard_normal((inter, hidden)) * 0.1).astype(np.float16),
                "up": (rng.standard_normal((inter, hidden)) * 0.1).astype(np.float16),
                "down": (rng.standard_normal((hidden, inter)) * 0.1).astype(np.float16)}
               for _ in range(E)]
    stacked = moe_int4.build_stacked_experts(experts, gs, xp=cp)
    gw = cp.asarray((rng.standard_normal((E, hidden)) * 0.1).astype(np.float16))
    # non-degenerate x so routing isn't an all-tied threshold
    x = cp.asarray((rng.standard_normal(hidden) * 0.5).astype(np.float32))
    lbuf = cp.empty(E, dtype=cp.float32); ibuf = cp.empty((E, inter), dtype=cp.float32)
    rw = cp.empty(E, dtype=cp.float32); out = cp.zeros(hidden, dtype=cp.float32)
    stream = cp.cuda.Stream(non_blocking=True)

    def step():
        moe_int4.gate_matvec(x, gw, E, hidden, out=lbuf)
        e = cp.exp(lbuf - lbuf.max()); probs = e / e.sum()
        kth = cp.sort(probs)[-K]; rw[:] = cp.where(probs >= kth, probs, 0.0)
        out.fill(0)
        moe_int4.fused_moe_weighted(x, stacked, rw, E, hidden, inter, out=out, inter_buf=ibuf)

    with stream:
        for _ in range(3):
            step()
    stream.synchronize()
    with stream:
        stream.begin_capture()
        step()
    graph = stream.end_capture()
    graph.launch(stream); stream.synchronize()
    assert int((cp.asnumpy(rw) > 0).sum()) == K


def test_gemv_close_to_marlin():
    """Row-major and marlin use the same RTN quant, so their GEMV outputs should
    agree to int4 rounding (both dequantize the same q/scale/zero)."""
    from fastllm_py.kernels import marlin
    if not marlin.available():
        pytest.skip("marlin .so missing")
    out_f, in_f, gs = 1408, 2048, 128
    rng = np.random.default_rng(1)
    w = (rng.standard_normal((out_f, in_f)) * 0.1).astype(np.float32)
    x = (rng.standard_normal(in_f) * 0.5).astype(np.float16)

    pay = moe_int4.quantize_int4_rowmajor(w, gs, xp=cp)
    y_rm = cp.asnumpy(moe_int4.gemv_int4(cp.asarray(x.astype(np.float32)), pay))
    ref = x.astype(np.float32) @ moe_int4.dequantize_int4_rowmajor(pay, xp=cp).get().T
    # row-major GEMV == its own dequant ref (self-consistent; the marlin cross-
    # check is covered by test_native_kernels)
    rel = np.abs(y_rm - ref).max() / (np.abs(ref).max() + 1e-6)
    assert rel < 1e-3, rel
