"""Runtime refusal-direction projection (EXL3_ABLIT_RUNTIME): the yamz-labs "uncensor" spec, applied with no weight edit.

A spec (uncensor_spec.json: hidden, n_layers, per-layer strengths attn_w[L], mlp_w[L]) plus one unit direction r
(uncensor_direction.st: safetensors layout, tensor "r", fp32 [hidden]). Editing every writer as W' = (I - w r r^T) W
(attention o_proj, dense down_proj, shared + routed expert down_proj) is linear in the writer output, so it equals
projecting each block's attention output and MLP output before the mHC hc_apply mixes it into the four streams:

    y <- y - w_L * r (r . y)        every token row, prefill and decode, w_L = attn_w[L] or mlp_w[L]

The MLP output is complete when BlockSparseMLP.forward returns: the routed sum, the CPU-lane / NVMe->CPU partial
(cpu_split_combine, ft_combine, nv_combine) and the shared expert (fused kernel, k_overlap side stream or python add)
are all folded in before it. So the hook wraps block.attn.forward and block.mlp.forward of the main model's
TransformerBlocks (instance attributes): stock TransformerBlock.forward and k_hcfuse's fused forward both use the
projected y, and the deferred hc_apply holds the same tensor. The attention output of the prefill-only last KV block
is discarded by the block, so it is not projected. MTP draft blocks are never touched: only the main model is
passed in, and their checkpoint layer numbers are >= n_layers. Torch path: one fp32 GEMV + one rank-1 update, in
place, per hooked site.

The published GLM-5.3-Flash spec (yamz-labs/GLM-5.3-Flash-EXL3-Yamz-Uncensored @ 4614256a, spec sha256 fe9989cb...f8d97,
direction 7ca504e1...0c608) hooks attention of layers 24-44 (w 2.14-2.21) and MLP of layers 11-38 (w 0.90-1.49):
49 sites on 34 blocks. It was fitted on a 2.05 bpw quantisation and measured by its authors on their pack; on the
3.05 bpw checkpoint served here its effect and quality cost are not measured.

Resolve (the rules of the Kyojin engine's exllamav3/modules/ablit_runtime.py, so the same files and switches work):
  EXL3_ABLIT_RUNTIME=off|0|false|no|none, or serve.py --no-uncensor   off
  EXL3_ABLIT_RUNTIME=<spec.json>                                      that spec (missing or malformed: the load stops)
  unset                                                               <model dir>/uncensor_spec.json if present
Direction file: the spec's "r_file" (relative to the spec), else <spec>.safetensors or <spec dir>/uncensor_direction.st
(bundled: uncensor_direction.st first). No runtime on/off switch: exllamav3 reuses cached KV pages and recurrent
states across requests, which would mix the two settings; A/B by restarting with EXL3_ABLIT_RUNTIME=off.
No spec found (the default): nothing is installed and every mode keeps its exactness.

serve.py hooks: ablit.load(model_dir) before model load (fails fast), ablit.install(model) after the other patches.
"""
import hashlib
import json
import os
import re
import torch

BUNDLED = "uncensor_spec.json"
BUNDLED_DIRECTION = "uncensor_direction.st"   # safetensors layout; not *.safetensors so weight loaders never index it
OFF = ("off", "0", "false", "no", "none")
REQUIRED = ("hidden", "n_layers", "attn_w", "mlp_w")
EPS_W = 1e-9                                   # |w| at or below this is no edit
_LAYER = re.compile(r"\.layers\.(\d+)$")

SPEC = None    # loaded spec (load())
SITES = {"attn": [], "mlp": []}   # hooked layer numbers (install())
_R = {}        # (device, dtype) -> direction


def resolve(model_dir):
    """(spec path, source) or (None, reason); source / reason in env, bundled, off, none."""
    env = os.environ.get("EXL3_ABLIT_RUNTIME", "").strip()
    if env.lower() in OFF:
        return None, "off"
    if env:
        return env, "env"
    if model_dir and os.path.isfile(os.path.join(model_dir, BUNDLED)):
        return os.path.join(model_dir, BUNDLED), "bundled"
    return None, "none"


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load(model_dir):
    """Resolve, read and check the spec and its direction. Returns the loaded spec, or None when off."""
    global SPEC
    path, source = resolve(model_dir)
    if path is None:
        if source == "off" and model_dir and os.path.isfile(os.path.join(model_dir, BUNDLED)):
            print(f" -- ablit runtime: OFF (EXL3_ABLIT_RUNTIME=off); bundled {BUNDLED} in the model directory ignored",
                  flush=True)
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ablit runtime: spec file not found: {path} (source: {source})")
    try:
        with open(path) as f:
            spec = json.load(f)
    except ValueError as e:
        raise ValueError(f"ablit runtime: spec {path} is not valid JSON: {e}") from None
    missing = [k for k in REQUIRED if not isinstance(spec, dict) or k not in spec]
    if missing:
        raise ValueError(f"ablit runtime: spec {path} is missing field(s) {missing}")
    n = int(spec["n_layers"])
    if len(spec["attn_w"]) < n or len(spec["mlp_w"]) < n:
        raise ValueError(f"ablit runtime: spec {path}: attn_w/mlp_w shorter than n_layers {n}")
    d = os.path.dirname(os.path.abspath(path))
    r_file = spec.get("r_file")
    if r_file:
        r_file = r_file if os.path.isabs(r_file) else os.path.join(d, r_file)
    else:
        cands = [os.path.splitext(path)[0] + ".safetensors", os.path.join(d, BUNDLED_DIRECTION)]
        if source == "bundled":
            cands.reverse()
        r_file = next((c for c in cands if os.path.isfile(c)), cands[0])
    if not os.path.isfile(r_file):
        raise FileNotFoundError(f"ablit runtime: direction file not found: {r_file} (spec {path})")
    try:
        from safetensors import safe_open
        with safe_open(r_file, "pt") as f:
            r = f.get_tensor("r").double()
    except Exception as e:
        raise ValueError(f"ablit runtime: cannot read direction 'r' from {r_file}: {e}") from None
    if r.dim() != 1 or r.shape[0] != int(spec["hidden"]):
        raise ValueError(f"ablit runtime: direction shape {tuple(r.shape)} != hidden {spec['hidden']} ({r_file})")
    norm = float(r.norm())
    if not 0.0 < norm < float("inf"):   # also NaN
        raise ValueError(f"ablit runtime: direction in {r_file} has norm {norm}")
    SPEC = {"path": path, "source": source, "r_file": r_file, "spec": spec, "r": (r / norm).float(),
            "r_norm": norm, "sha256": {"spec": _sha256(path), "direction": _sha256(r_file)}}
    _R.clear()
    print(f" -- ablit runtime: spec {path} active "
          f"({'bundled in the model directory' if source == 'bundled' else 'from EXL3_ABLIT_RUNTIME'})", flush=True)
    print(f" -- ablit runtime: direction {r_file} (norm {norm:.9f}, renormalised), sha256 spec "
          f"{SPEC['sha256']['spec']}, direction {SPEC['sha256']['direction']}", flush=True)
    if abs(norm - 1.0) > 1e-3:
        print(f" !! ablit runtime: direction norm {norm:.6f} is not 1; the strengths assume a unit vector", flush=True)
    return SPEC


def _dir(dev, dt):
    k = (dev, dt)
    r = _R.get(k)
    if r is None:
        r = _R[k] = SPEC["r"].to(device=dev, dtype=dt)
    return r


def project(y, w):
    """y -= w * r (r . y) over the last dim, in place (a contiguous copy when y is not contiguous). Returns y."""
    if not y.is_contiguous():
        y = y.contiguous()
    r32 = _dir(y.device, torch.float)
    y2 = y.view(-1, r32.shape[0])
    if y2.dtype == torch.float:
        y2.addr_(torch.mv(y2, r32), r32, alpha=-w)
    else:   # GLM-5.3 attn / mlp outputs are fp32; kept for other output dtypes (dot in fp32)
        y2.addr_(torch.mv(y2.float(), r32).to(y2.dtype), _dir(y.device, y2.dtype), alpha=-w)
    return y


def _wrap(f, w, attn):
    def fwd(*a, **k):
        y = f(*a, **k)
        if not isinstance(y, torch.Tensor):
            return y
        if attn:
            params = a[1] if len(a) > 1 else k.get("params") or {}
            if params.get("prefill") and not params.get("export_state_layers"):
                return y   # last KV block in a prefill: TransformerBlock returns before reading this output
        return project(y, w)
    return fwd


def _layer(b):
    m = _LAYER.search(getattr(b, "key", None) or "")
    return int(m.group(1)) if m else None


def install(model):
    """Wrap the attn / mlp forwards of the main model's decoder blocks that have a nonzero strength. Returns the
    number of hooked sites. Pass the main model only, never the MTP draft model."""
    if SPEC is None:
        return 0
    spec = SPEC["spec"]
    n = int(spec["n_layers"])
    hidden = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden is not None and int(hidden) != int(spec["hidden"]):
        raise ValueError(f"ablit runtime: spec hidden {spec['hidden']} != model hidden_size {hidden} ({SPEC['path']})")
    SITES["attn"].clear()
    SITES["mlp"].clear()
    blocks = 0
    for b in model.modules:
        if type(b).__name__ != "TransformerBlock":
            continue
        L = _layer(b)
        key = getattr(b, "key", "") or ""
        if L is None or L >= n or key.split(".")[0] == "mtp" or ".mtp." in key:
            continue
        hooked = False
        for site in ("attn", "mlp"):
            m = getattr(b, site, None)
            w = float(spec[f"{site}_w"][L])
            if m is None or abs(w) <= EPS_W:
                continue
            m.forward = _wrap(m.forward, w, site == "attn")
            SITES[site].append(L)
            hooked = True
        if hooked:
            blocks += 1
            dev = getattr(b, "device", None)
            if dev is not None:
                _dir(torch.device(dev), torch.float)   # direction on the block's device now, not inside a forward
    sites = len(SITES["attn"]) + len(SITES["mlp"])
    print(f" -- ablit runtime: {blocks} hooked blocks, {sites} sites (attention {_span(SITES['attn'])}, "
          f"MLP {_span(SITES['mlp'])})", flush=True)
    if not sites:
        print(f" !! ablit runtime: spec {SPEC['path']} matched no decoder block; the model is unchanged", flush=True)
    return sites


def _span(ls):
    if not ls:
        return "none"
    return f"layers {ls[0]}-{ls[-1]}" if ls == list(range(ls[0], ls[-1] + 1)) else f"layers {ls}"


def summary():
    if SPEC is None:
        return {"active": False}
    spec = SPEC["spec"]
    return {"active": True, "spec": SPEC["path"], "source": SPEC["source"], "direction": SPEC["r_file"],
            "sha256": SPEC["sha256"],
            "attn": {str(L): float(spec["attn_w"][L]) for L in SITES["attn"]},
            "mlp": {str(L): float(spec["mlp_w"][L]) for L in SITES["mlp"]}}
