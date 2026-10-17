"""Fused Triton mHC (hyper-connection) pre/post for XPU, drop-in for sglang mhc._mhc_pre_torch / _mhc_post_torch.
pre: split-K GEMV (x @ fn^T, sum x^2) -> finalize (rsqrt, sigmoid pre/post, softmax + Sinkhorn on n x n comb, layer_input)."""
import torch, triton, triton.language as tl

@triton.jit
def _pre_gemv(x_ptr, fn_ptr, pmix_ptr, psq_ptr, S, K: tl.constexpr, M: tl.constexpr, MP: tl.constexpr,
              CK: tl.constexpr, BK: tl.constexpr):
    s = tl.program_id(0); p = tl.program_id(1); NS = tl.num_programs(1)
    rm = tl.arange(0, MP); mm = rm < M
    acc = tl.zeros([MP], dtype=tl.float32); sq = tl.zeros([BK], dtype=tl.float32)
    for k0 in range(p * CK, p * CK + CK, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + s * K + rk).to(tl.float32)
        f = tl.load(fn_ptr + rm[:, None] * K + rk[None, :], mask=mm[:, None], other=0.0)
        acc += tl.sum(f * x[None, :], axis=1); sq += x * x
    tl.store(pmix_ptr + (p * S + s) * MP + rm, acc)
    tl.store(psq_ptr + p * S + s, tl.sum(sq, axis=0))

@triton.jit
def _pre_fin(res_ptr, pmix_ptr, psq_ptr, scale_ptr, base_ptr, post_ptr, comb_ptr, out_ptr, S, NS,
             rms_eps, pre_eps, sk_eps, post_mult, N: tl.constexpr, H: tl.constexpr, MP: tl.constexpr,
             BH: tl.constexpr, ITERS: tl.constexpr):
    s = tl.program_id(0); hb = tl.program_id(1)
    rn = tl.arange(0, N)
    pre = tl.zeros([N], dtype=tl.float32); post = tl.zeros([N], dtype=tl.float32)
    comb = tl.zeros([N, N], dtype=tl.float32); sq = 0.0
    for p in range(NS):
        base = pmix_ptr + (p * S + s) * MP
        pre += tl.load(base + rn); post += tl.load(base + N + rn)
        comb += tl.load(base + 2 * N + rn[:, None] * N + rn[None, :])
        sq += tl.load(psq_ptr + p * S + s)
    r = 1.0 / tl.sqrt(sq / (N * H) + rms_eps)
    s0 = tl.load(scale_ptr); s1 = tl.load(scale_ptr + 1); s2 = tl.load(scale_ptr + 2)
    pre = tl.sigmoid(pre * r * s0 + tl.load(base_ptr + rn)) + pre_eps
    post = post_mult * tl.sigmoid(post * r * s1 + tl.load(base_ptr + N + rn))
    comb = comb * r * s2 + tl.load(base_ptr + 2 * N + rn[:, None] * N + rn[None, :])
    comb = tl.exp(comb - tl.max(comb, axis=1)[:, None])
    comb = comb / tl.sum(comb, axis=1)[:, None] + sk_eps
    comb = comb / (tl.sum(comb, axis=0)[None, :] + sk_eps)
    for _ in tl.static_range(ITERS - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + sk_eps)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + sk_eps)
    if hb == 0:
        tl.store(post_ptr + s * N + rn, post)
        tl.store(comb_ptr + s * N * N + rn[:, None] * N + rn[None, :], comb)
    rh = hb * BH + tl.arange(0, BH)
    res = tl.load(res_ptr + s * N * H + rn[:, None] * H + rh[None, :]).to(tl.float32)
    tl.store(out_ptr + s * H + rh, tl.sum(pre[:, None] * res, axis=0).to(tl.bfloat16))

@triton.jit
def _post(x_ptr, res_ptr, post_ptr, comb_ptr, out_ptr, N: tl.constexpr, H: tl.constexpr, BH: tl.constexpr):
    s = tl.program_id(0); hb = tl.program_id(1)
    rn = tl.arange(0, N); rh = hb * BH + tl.arange(0, BH)
    x = tl.load(x_ptr + s * H + rh).to(tl.float32)
    res = tl.load(res_ptr + s * N * H + rn[:, None] * H + rh[None, :]).to(tl.float32)
    po = tl.load(post_ptr + s * N + rn)
    cb = tl.load(comb_ptr + s * N * N + rn[:, None] * N + rn[None, :])          # [j, i]
    mix = tl.sum(cb[:, :, None] * res[:, None, :], axis=0)                        # [i, BH]
    out = po[:, None] * x[None, :] + mix
    tl.store(out_ptr + s * N * H + rn[:, None] * H + rh[None, :], out.to(tl.bfloat16))

NSPLIT = 16

def mhc_pre_triton(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat):
    s, n, h = residual.shape
    K = n * h; M = fn.shape[0]; MP = triton.next_power_of_2(M)
    res = residual.contiguous(); fn = fn.contiguous().float()
    dev = res.device
    pmix = torch.empty(NSPLIT, s, MP, device=dev, dtype=torch.float32)
    psq = torch.empty(NSPLIT, s, device=dev, dtype=torch.float32)
    _pre_gemv[(s, NSPLIT)](res, fn, pmix, psq, s, K=K, M=M, MP=MP, CK=K // NSPLIT, BK=256)
    post = torch.empty(s, n, device=dev, dtype=torch.float32)
    comb = torch.empty(s, n, n, device=dev, dtype=torch.float32)
    out = torch.empty(s, h, device=dev, dtype=residual.dtype)
    BH = 512
    _pre_fin[(s, h // BH)](res, pmix, psq, hc_scale.float(), hc_base.float(), post, comb, out, s, NSPLIT,
                           float(rms_eps), float(hc_pre_eps), float(hc_sinkhorn_eps), float(hc_post_mult_value),
                           N=n, H=h, MP=MP, BH=BH, ITERS=int(sinkhorn_repeat))
    return post.unsqueeze(-1), comb, out

def mhc_post_triton(x, residual, post_layer_mix, comb_res_mix):
    s, n, h = residual.shape
    out = torch.empty(s, n, h, device=x.device, dtype=x.dtype)
    BH = 512
    _post[(s, h // BH)](x.contiguous(), residual.contiguous(), post_layer_mix.reshape(s, n).float().contiguous(),
                        comb_res_mix.reshape(s, n, n).float().contiguous(), out, N=n, H=h, BH=BH)
    return out

def install():
    import os
    from sglang.kernels.ops.layernorm import mhc
    if os.environ.get("GLM_MHC_TRITON", "1") != "1": return
    tp, tq = mhc._mhc_pre_torch, mhc._mhc_post_torch
    def pre(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat):
        s, n, h = residual.shape
        if residual.device.type != "xpu" or h % 512 or (n * h) % (NSPLIT * 256) or residual.dtype != torch.bfloat16:
            return tp(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)
        return mhc_pre_triton(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)
    def post(x, residual, post_layer_mix, comb_res_mix):
        s, n, h = residual.shape
        if x.device.type != "xpu" or h % 512 or x.dtype != torch.bfloat16:
            return tq(x, residual, post_layer_mix, comb_res_mix)
        return mhc_post_triton(x, residual, post_layer_mix, comb_res_mix)
    mhc._mhc_pre_torch = pre; mhc._mhc_post_torch = post
    print("GLM_OPT: mhc triton installed", flush=True)
