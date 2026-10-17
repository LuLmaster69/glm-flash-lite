import torch, time, sys
sys.path.insert(0, "/opt/glmopt")
from sglang.kernels.ops.layernorm import mhc
import mhc_triton as T
d = "xpu"; torch.manual_seed(0); n, h = 4, 4096; M = (2 + n) * n
for s in (1, 2, 4, 8, 37, 300):
    res = (torch.randn(s, n, h, device=d) * 3).bfloat16(); fn = torch.randn(M, n * h, device=d) * 0.02
    sc = torch.tensor([0.5, 0.7, 1.3], device=d); bs = torch.randn(M, device=d) * 0.3
    args = (res, fn, sc, bs, 1e-6, 1e-6, 1e-6, 2.0, 20)
    a = mhc._mhc_pre_torch(*args); b = T.mhc_pre_triton(*args)
    errs = [((x.float() - y.float()).abs().max() / (x.float().abs().max() + 1e-9)).item() for x, y in zip(a, b)]
    x = torch.randn(s, h, device=d).bfloat16()
    pa = mhc._mhc_post_torch(x, res, a[0], a[1]); pb = T.mhc_post_triton(x, res, a[0], a[1])
    pe = ((pa.float() - pb.float()).abs().max() / pa.float().abs().max()).item()
    print("s=%d pre rel err post/comb/layer_in=%s post_out=%.2e" % (s, ["%.2e" % e for e in errs], pe))
s = 1; res = torch.randn(s, n, h, device=d).bfloat16(); fn = torch.randn(M, n * h, device=d) * 0.02
sc = torch.tensor([0.5, 0.7, 1.3], device=d); bs = torch.randn(M, device=d) * 0.3; args = (res, fn, sc, bs, 1e-6, 1e-6, 1e-6, 2.0, 20)
x = torch.randn(s, h, device=d).bfloat16()
for name, f in (("torch_pre", lambda: mhc._mhc_pre_torch(*args)), ("triton_pre", lambda: T.mhc_pre_triton(*args)),
                ("torch_post", lambda: mhc._mhc_post_torch(x, res, args[0][:, :, :1].float(), torch.rand(1, n, n, device=d))),
                ("triton_post", lambda: T.mhc_post_triton(x, res, args[0][:, :, :1].float(), torch.rand(1, n, n, device=d)))):
    for _ in range(5): f()
    torch.xpu.synchronize(); t = time.perf_counter()
    for _ in range(200): f()
    torch.xpu.synchronize(); print(name, f"{(time.perf_counter() - t) / 200 * 1e6:.1f} us")
