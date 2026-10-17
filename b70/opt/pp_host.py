"""GLM_PP_HOST=1: pipeline-parallel hand-off through host memory (gloo CPU group) instead of oneCCL device P2P.
On this EPYC board the two B70s sit on different root complexes; device-side P2P stalled a ccs job past the xe 5 s
timeout (card 48, 2026-10-15 12:44). Decode moves ~32 KB/token, so the host path costs microseconds."""
import torch

def _dev():
    return torch.device("xpu", torch.xpu.current_device())

def install(ps):
    G = ps.GroupCoordinator
    o_send, o_recv, o_sr = G.send_tensor_dict, G.recv_tensor_dict, getattr(G, "send_recv_tensor_dict", None)
    def to_cpu(d):
        return {k: (v.to("cpu") if isinstance(v, torch.Tensor) and not v.is_cpu else v) for k, v in d.items()}
    def to_dev(d):
        if d is None: return d
        dev = _dev()
        return {k: (v.to(dev, non_blocking=False) if isinstance(v, torch.Tensor) and v.is_cpu else v) for k, v in d.items()}
    def send(self, tensor_dict, *a, **k):
        if self.world_size == 1: return o_send(self, tensor_dict, *a, **k)
        return o_send(self, to_cpu(tensor_dict), *a, **k)
    def recv(self, *a, **k):
        return to_dev(o_recv(self, *a, **k))
    G.send_tensor_dict, G.recv_tensor_dict = send, recv
    if o_sr is not None:
        def sr(self, send_tensor_dict, *a, **k):
            return to_dev(o_sr(self, to_cpu(send_tensor_dict), *a, **k))
        G.send_recv_tensor_dict = sr
    print("GLM_OPT: PP hand-off via host (gloo)", flush=True)
