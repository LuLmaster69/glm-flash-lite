"""Post-import hooks for GLM-5.3-Flash on XPU: install /opt/glmopt patches when their target modules load."""
import importlib, importlib.abc, importlib.util, os, sys
sys.path.insert(0, "/opt/glmopt")
HOOKS = {}
def _mhc(mod):
    import mhc_triton; mhc_triton.install()
HOOKS["sglang.kernels.ops.layernorm.mhc"] = _mhc
def _pp(mod):
    if os.environ.get("GLM_PP_HOST", "1") == "1":
        import pp_host; pp_host.install(mod)
HOOKS["sglang.srt.distributed.parallel_state"] = _pp
def _moe(mod):
    # GLM shapes (H 4096, I 2048): the DPAS prefill MoE kernel (M >= 256) runs away on B70 (2026-10-15, A9-A11);
    # route every M through the validated GEMV path until DPAS is fixed. GLM_MOE_PREFILL_MIN_M overrides.
    v = int(os.environ.get("GLM_MOE_PREFILL_MIN_M", str(1 << 30)))
    orig = mod.ops
    def ops():
        o = orig()
        if not getattr(ops, "_set", False):
            o.moe_set_prefill_min_m(v); ops._set = True
            print(f"GLM_OPT: moe prefill_min_m={v}", flush=True)
        return o
    mod.ops = ops
def _ftx(mod):
    # GLM NextN draft (DeepseekV3ForCausalLMNextN) names its block 'model.decoder.*' plus 'model.eh_proj/enorm/hnorm';
    # the checkpoint (mtp.safetensors) has them as layers.<num_hidden_layers>.*. Map them so the plugin finds the EXL3
    # modules (else the draft MoE is built unquantized: 4.5 GiB bf16 per expert tensor -> OOM).
    N = os.environ.get("EXL3_NEXTN_LAYER", "45")
    orig_nk = mod.norm_key
    def norm_key(k):
        parts = k.split(".")
        if "decoder" in parts and "visual" not in parts:
            return f"layers.{N}." + ".".join(parts[parts.index("decoder") + 1:])
        if parts[-1] in ("eh_proj", "enorm", "hnorm") or parts[-2:] == ["shared_head", "norm"]:
            i = parts.index("eh_proj") if "eh_proj" in parts else len(parts) - (2 if parts[-2] == "shared_head" else 1)
            return f"layers.{N}." + ".".join(parts[i:])
        return orig_nk(k)
    mod.norm_key = norm_key
    if os.environ.get("FTX", "0") == "1":
        import ftx_tier; ftx_tier.install(mod)
HOOKS["exl3xpu.sglang_plugin"] = _ftx
def _nextn(mod):
    import torch
    # mla_unfuse turns off the q_a/kv_a cat-fusion on the TARGET class only; the NextN draft's load_weights fuses them
    # unless the attribute exists -> EXL3 trellis/mul1 tensors hit torch.cat ("zero-dimensional tensor"). Same rule.
    cls = mod.Glm5NextForConditionalGenerationNextN
    orig = cls.__init__
    def __init__(self, *a, **k):
        orig(self, *a, **k)
        qc = getattr(self, "quant_config", None)
        try:
            if qc is not None and qc.get_name() == "exl3": self.fuse_qkv_a_proj = False
        except Exception: pass
        # eh_proj is a plain nn.Linear in DeepseekModelNextN but EXL3 in the checkpoint (trellis/suh/svh/mul1): the
        # loader silently skipped it and the draft ran on a randomly initialised projection (accept rate 0).
        try:
            from exl3xpu import sglang_plugin as P
            Cfg, _, Dense = P.classes()
            lin = getattr(self.model, "eh_proj", None)
            if isinstance(qc, Cfg) and isinstance(lin, torch.nn.Linear):
                info = qc.lookup("model.eh_proj")
                if info is not None:
                    self.model.eh_proj = Dense(lin.in_features, lin.out_features, info, "model.eh_proj",
                                               params_dtype=lin.weight.dtype)
                    print(f"GLM_OPT: NextN eh_proj -> EXL3 dense {info}", flush=True)
        except Exception as e:
            print("GLM_OPT: NextN eh_proj EXL3 swap failed", repr(e), flush=True)
    cls.__init__ = __init__
    orig_load = cls.load_weights
    def load_weights(self, weights, *a, **k):
        out = orig_load(self, weights, *a, **k)
        ep = getattr(self.model, "eh_proj", None)
        if ep is not None and hasattr(ep, "method") and hasattr(ep, "process_weights_after_loading"):
            ep.process_weights_after_loading()
            print("GLM_OPT: NextN eh_proj EXL3 weights processed", flush=True)
        return out
    cls.load_weights = load_weights
    def set_lm_head_from_target(self, target_lm_head):
        # EXL3 heads have no dense .weight (set_embed_and_head only swaps a placeholder): use the target's module
        self.lm_head = target_lm_head
        print("GLM_OPT: NextN draft shares the target EXL3 lm_head", flush=True)
    cls.set_lm_head_from_target = set_lm_head_from_target
    if os.environ.get("GLM_NANCHK", "0") == "1":
        import torch
        fwd0 = cls.forward
        st = {"n": 0}
        def forward(self, input_ids, positions, forward_batch, *a, **k):
            if st["n"] == 0:
                for n, m in self.named_modules():
                    w = getattr(m, "weight", None)
                    tr = getattr(m, "exl3_trellis", None)
                    if tr is not None:
                        print(f"GLM_DRAFT_W {n}: exl3 trellis {tuple(tr.shape)}", flush=True)
                    elif isinstance(w, torch.Tensor) and w.dim() >= 1 and len(list(m.children())) == 0:
                        wf = w.float() if w.numel() else w
                        print(f"GLM_DRAFT_W {n}: {type(m).__name__} {tuple(w.shape)} {w.dtype} "
                              f"mean={float(wf.mean()) if w.numel() else 'EMPTY':.4} std={float(wf.std()) if w.numel() > 1 else 0:.4}", flush=True)
            out = fwd0(self, input_ids, positions, forward_batch, *a, **k)
            if st["n"] < 12 and not torch.xpu.is_current_stream_capturing():
                st["n"] += 1
                lg = getattr(out, "next_token_logits", None)
                hs = getattr(forward_batch.spec_info, "hidden_states", None)
                print(f"GLM_DRAFT mode={forward_batch.forward_mode} ids={input_ids[:8].tolist()} pos={positions[:8].tolist()} "
                      f"hid={None if hs is None else (tuple(hs.shape), bool(torch.isfinite(hs).all()), round(float(hs.float().abs().mean()), 3))} "
                      f"logits_finite={None if lg is None else bool(torch.isfinite(lg).all())} "
                      f"top1={None if lg is None else lg.argmax(-1)[:8].tolist()}", flush=True)
            return out
        cls.forward = forward
HOOKS["sglang.srt.models.glm5_next_nextn"] = _nextn
def _topk_share(mod):
    # XPU DSA appends the kpool tail to topk indices (width index_topk + kpool - 1); the MTP seed buffer is
    # index_topk wide. Publish the pooled-history part only: the consumer re-appends its own rows' tail.
    cls = mod.IndexTopKShareState
    def publish(self):
        if self._topk_indices is None or not self.should_publish:
            return
        if self.enabled:
            self._forward_batch.spec_info.dsa_topk_indices = self._topk_indices
        seed_buf = self._seed_buf
        if seed_buf is not None:
            sel = self._forward_batch.spec_info.dsa_seed_topk_select
            src = self._topk_indices[: seed_buf.shape[0]] if sel is None else self._topk_indices[sel]
            if src.shape[-1] > seed_buf.shape[-1]:
                src = src[..., : seed_buf.shape[-1]]
            elif src.shape[-1] < seed_buf.shape[-1]:
                # buffer sized for tail-appended indices; the draft's indexer output has no tail: pad -1 (draft-only
                # proposals lose the <= kpool-1 always-select tail tokens; the target verifies with full attention)
                pad = src.new_full(src.shape[:-1] + (seed_buf.shape[-1] - src.shape[-1],), -1)
                src = __import__("torch").cat([src, pad], dim=-1)
            if not getattr(cls, "_glm_log", False):
                cls._glm_log = True
                print(f"GLM_OPT: topk seed src {tuple(self._topk_indices.shape)} -> buf {tuple(seed_buf.shape)}", flush=True)
            seed_buf[: src.shape[0]].copy_(src)
    cls.publish = publish
HOOKS["sglang.srt.layers.attention.index_topk_share"] = _topk_share
def _pad(mod):
    # seqlens_expand_triton is a device-agnostic Triton kernel guarded by `assert .is_cuda` (MTP verify metadata)
    import torch, triton
    kern = mod.seqlens_expand_kernel
    def seqlens_expand_triton(extend_seq_lens, seq_lens, total_len, max_q_len):
        N = extend_seq_lens.numel()
        offsets = torch.zeros(N + 1, device=extend_seq_lens.device, dtype=torch.int32)
        offsets[1:] = torch.cumsum(extend_seq_lens, dim=0)
        output = torch.empty(total_len, device=extend_seq_lens.device, dtype=torch.int32)
        kern[(N,)](extend_seq_lens, seq_lens, offsets, output, N, BLOCK=triton.next_power_of_2(max_q_len))
        return output
    mod.seqlens_expand_triton = seqlens_expand_triton
HOOKS["sglang.kernels.ops.attention.pad"] = _pad
def _sampler(mod):
    # grammar requests (tool-call parser) all-reduce token ids across TP even at TP=1; on XPU that initialises oneCCL,
    # which needs /dev/dri enumeration a 1-card container does not have. Nothing to sync with one rank: skip.
    orig = mod.Sampler._sync_token_ids_across_tp
    def sync(self, ids, info):
        g = getattr(self, "tp_sync_group", None)
        try:
            import torch.distributed as dist
            if g is not None and dist.get_world_size(group=g) == 1: return
        except Exception: pass
        return orig(self, ids, info)
    mod.Sampler._sync_token_ids_across_tp = sync
HOOKS["sglang.srt.layers.sampler"] = _sampler
for _n in ("exl3xpu.moe_offload",):
    HOOKS[_n] = _moe
if os.environ.get("GLM_SYNC_ASSERT", "0") == "1":
    # debug: XPU device asserts lose message + Python stack; check synchronously (outside graph capture) instead
    import torch, traceback
    _orig_aa = torch._assert_async
    def _assert_async(cond, msg=""):
        try:
            cap = torch.xpu.is_current_stream_capturing()
        except Exception:
            cap = False
        if cap:
            return _orig_aa(cond, msg) if msg else _orig_aa(cond)
        ok = bool(cond.all().item()) if hasattr(cond, "all") else bool(cond)
        if not ok:
            print(f"GLM_SYNC_ASSERT FAILED: {msg!r}", file=sys.stderr, flush=True)
            traceback.print_stack(file=sys.stderr)
            raise AssertionError(f"GLM_SYNC_ASSERT: {msg}")
    torch._assert_async = _assert_async
def _nanchk(mod):
    # debug (GLM_NANCHK=1, eager only): first decoder layer whose output goes non-finite in target_verify / decode
    if os.environ.get("GLM_NANCHK", "0") != "1":
        return
    import torch
    cls = mod.Glm5NextDecoderLayer
    orig = cls.forward
    st = {"n": 0}
    def forward(self, positions, hidden_states, forward_batch, residual, *a, **k):
        fin_in = bool(torch.isfinite(hidden_states).all())
        out = orig(self, positions, hidden_states, forward_batch, residual, *a, **k)
        fm = forward_batch.forward_mode
        if st["n"] < 6 and (fm.is_target_verify() or fm.is_decode()):
            h = out[0] if isinstance(out, tuple) else out
            if fin_in and not bool(torch.isfinite(h).all()):
                st["n"] += 1
                kind = "KDA" if getattr(self, "is_linear_attn", False) else "DSA/MLA"
                print(f"GLM_NANCHK first non-finite: layer {getattr(self, 'layer_id', '?')} ({kind}) mode={fm} rows={h.shape[0]} "
                      f"bad_rows={(~torch.isfinite(h.float()).all(-1)).nonzero().flatten().tolist()[:16]}", flush=True)
        return out
    cls.forward = forward
HOOKS["sglang.srt.models.glm5_next"] = _nanchk
class _F(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in HOOKS: return None
        sys.meta_path.remove(self)
        try: spec = importlib.util.find_spec(name)
        finally: sys.meta_path.insert(0, self)
        orig = spec.loader.exec_module
        def ex(mod, _n=name):
            orig(mod)
            try: HOOKS.pop(_n)(mod)
            except Exception as e: print("GLM_OPT hook failed", _n, repr(e), file=sys.stderr, flush=True)
        spec.loader.exec_module = ex
        return spec
if os.environ.get("GLM_OPT", "1") == "1":
    sys.meta_path.insert(0, _F())
