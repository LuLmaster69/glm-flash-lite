"""test_ftx.py: standalone CPU-tier handshake + numerics test on one B70 (no server). One real layer (3) on the CPU tier;
all picks forced to the CPU; checks job publish (GPU -> host), done flag (host -> GPU), partial vs tier_forward."""
import os, sys, time, json
sys.path.insert(0, "/opt/glmopt")
os.environ.setdefault("FTX_BUILD", "/opt/glmopt-build/ct")
import torch
import ftx_tier as F
from exl3xpu import moe_offload

dev = torch.device("xpu", 0)
X = moe_offload.ops()
Hx = F._build_host()
torch.ops.load_library("/opt/glmopt/ftx/_ftx.so")
E, H, I, K = 288, 4096, 2048, 3
ctrl = X.host_alloc(64).view(torch.int64); ctrl.zero_()
hx = X.host_alloc(4 * H * 2).view(torch.float16)
picks = X.host_alloc(4 * 8 * 3 * 4).view(torch.int32)
hout = X.host_alloc(4 * H * 4).view(torch.float32)
cpus = list(range(4, 24))
Hx.tier_init(len(cpus), cpus, ctrl, hx, picks, hout, -1, H, I, 1, 1)
ck = F._Ckpt(os.environ["EXL3_MODEL_PATH"])
pre = "model.language_model.layers.3.mlp.experts."
tr, sc = [], {k: [] for k in ("suh_g", "svh_g", "suh_u", "svh_u", "suh_d", "svh_d")}
for e in range(E):
    row = []
    for pj, tag in (("gate_proj", "g"), ("up_proj", "u"), ("down_proj", "d")):
        t = ck.get(f"{pre}{e}.{pj}.trellis").contiguous(); row.append(t)
        sc["suh_" + tag].append(ck.get(f"{pre}{e}.{pj}.suh").to(torch.float16))
        sc["svh_" + tag].append(ck.get(f"{pre}{e}.{pj}.svh").to(torch.float16))
    tr.append(row)
print("trellis", tr[0][0].dtype, tuple(tr[0][0].shape), tuple(tr[0][2].shape), "suh", sc["suh_g"][0].dtype, flush=True)
ptrs = torch.tensor([[t.data_ptr() for t in row] for row in tr], dtype=torch.int64)
st = {k: torch.stack(v).contiguous() for k, v in sc.items()}
Hx.tier_add_layer(0, ptrs, st["suh_g"], st["svh_g"], st["suh_u"], st["svh_u"], st["suh_d"], st["svh_d"])

torch.manual_seed(0)
x = (torch.randn(1, H) * 0.5).to(torch.bfloat16)
ids = torch.randperm(E)[:8].to(torch.int32).view(1, 8)
w = torch.softmax(torch.randn(1, 8), -1).float()
ref = Hx.tier_forward(0, x.float().to(torch.float16).float(), ids, w, -1)
print("ref norm", float(ref.norm()), flush=True)
Hx.tier_start(cpus[0])

slot_of = torch.full((E,), -1, dtype=torch.int32, device=dev)
score = torch.zeros(E, dtype=torch.float32, device=dev)
pol = torch.tensor([0.40, 0.023, 0.11, 0.104, 0.2, 32, 8, 0], dtype=torch.float32, device=dev)   # force_n = 8
seq = torch.zeros(1, dtype=torch.int64, device=dev); dflag = torch.zeros_like(seq); stats = torch.zeros(8, dtype=torch.int64, device=dev)
xd, idd, wd = x.to(dev), ids.to(dev), w.to(dev)
for it in range(3):
    ids2, w2 = torch.ops.ftx.split(idd, wd, xd, slot_of, 0, E, score, pol, ctrl.data_ptr(), hx.data_ptr(), picks.data_ptr(), seq, dflag, stats)
    torch.xpu.synchronize()
    print(f"[{it}] after split: ctrl={ctrl.tolist()[:5]} dflag={dflag.item()} ids2={ids2.tolist()} hx0={hx[:4].tolist()} x0={x[0,:4].tolist()}", flush=True)
    t0 = time.time()
    while ctrl[1].item() < ctrl[0].item() and time.time() - t0 < 5: time.sleep(0.0005)
    print(f"[{it}] host sees done: ctrl1={ctrl[1].item()} after {1e3*(time.time()-t0):.2f} ms", flush=True)
    y = torch.zeros(1, H, dtype=torch.bfloat16, device=dev)
    torch.ops.ftx.combine(y, dflag, ctrl.data_ptr(), hout.data_ptr(), stats, 400000)
    torch.xpu.synchronize()
    out = y.float().cpu()
    print(f"[{it}] stats={stats.tolist()} rel_err_vs_ref={float((out - ref).norm() / ref.norm()):.4e} hout_vs_ref={float((hout[:H].view(1,H) - ref).norm() / ref.norm()):.4e}", flush=True)
# probe A: GPU->host visibility of the job publish (no host sync)
import threading
Hx.tier_stop()
for it in range(3):
    c0 = ctrl[0].item()
    ids2, w2 = torch.ops.ftx.split(idd, wd, xd, slot_of, 0, E, score, pol, ctrl.data_ptr(), hx.data_ptr(), picks.data_ptr(), seq, dflag, stats)
    t0 = time.time()
    while ctrl[0].item() == c0 and time.time() - t0 < 2: pass
    print(f"A[{it}] host saw ctrl0 {c0}->{ctrl[0].item()} after {1e3*(time.time()-t0):.3f} ms (no sync)", flush=True)
    torch.xpu.synchronize()
# probe B: host->GPU visibility while the GPU spins; host writes ctrl[1] = ctrl[0] after 5 ms
for spin in [400000 | (m << 40) for m in (0, 1, 2)]:
    ids2, w2 = torch.ops.ftx.split(idd, wd, xd, slot_of, 0, E, score, pol, ctrl.data_ptr(), hx.data_ptr(), picks.data_ptr(), seq, dflag, stats)
    torch.xpu.synchronize()
    s = ctrl[0].item(); st0 = stats.tolist()
    def wr():
        time.sleep(0.005); ctrl[1] = s
    th = threading.Thread(target=wr); t0 = time.time()
    torch.ops.ftx.combine(y, dflag, ctrl.data_ptr(), hout.data_ptr(), stats, spin)
    th.start(); torch.xpu.synchronize(); th.join()
    st1 = stats.tolist()
    print(f"B[mode {spin >> 40}] combine done after {1e3*(time.time()-t0):.2f} ms, polls={st1[4]-st0[4]}, timeouts={st1[6]-st0[6]}", flush=True)
