"""exl3 + MLA (q_lora_rank) fix.

SGLang's DeepseekV2AttentionMLA builds a single `fused_qkv_a_proj_with_mqa` linear whenever q_lora_rank is set, and
the model's load_weights fuses the checkpoint's `q_a_proj` + `kv_a_proj_with_mqa` into it with a raw `torch.cat`.
That only works for dense / awq / moe_wna16 weights. EXL3 stores each projection as separate trellis/suh/svh/mul1
tensors (the mul1 scalar is 0-dim), so the cat raises "zero-dimensional tensor cannot be concatenated". Qwen's
flash-next is GQA so this path never ran; GLM-5.3-Flash and DeepSeek-V4.1-Flash (both MLA) hit it.

Fix: for exl3 only, build SEPARATE `q_a_proj` and `kv_a_proj_with_mqa` exl3 linears instead of the fused one, and set
the owning model's `fuse_qkv_a_proj = False` so load_weights routes each as a standard single-matrix exl3 linear. The
attention forward already supports this: DeepseekV2AttentionMLA.prepare_qkv_latent concatenates the two projections'
OUTPUTS when `hasattr(self, "q_a_proj")`, which is mathematically identical to the fused projection.
"""
import logging

logger = logging.getLogger("exl3xpu")

_done = False


def _is_exl3(linear) -> bool:
    qm = getattr(linear, "quant_method", None)
    qc = getattr(qm, "quant_config", None)
    try:
        return qc is not None and qc.get_name() == "exl3"
    except Exception:
        return False


def install() -> None:
    global _done
    if _done:
        return
    try:
        from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.utils import add_prefix
    except Exception as e:  # pragma: no cover - SGLang without the MLA attention
        logger.info("exl3xpu: mla_unfuse not installed (%s)", e)
        return

    _orig_attn_init = DeepseekV2AttentionMLA.__init__

    def _attn_init(self, *args, **kwargs):
        prefix = kwargs.get("prefix")
        qc = kwargs.get("quant_config")
        for v in args:
            if prefix is None and isinstance(v, str):
                prefix = v
            if qc is None and hasattr(v, "get_name"):
                qc = v
        _orig_attn_init(self, *args, **kwargs)
        fused = getattr(self, "fused_qkv_a_proj_with_mqa", None)
        try:
            is_exl3 = qc is not None and qc.get_name() == "exl3"
        except Exception:
            is_exl3 = False
        if fused is None or not is_exl3:
            return
        fprefix = getattr(fused, "prefix", None) or add_prefix("fused_qkv_a_proj_with_mqa", prefix or "")
        base = fprefix.rsplit("fused_qkv_a_proj_with_mqa", 1)[0]  # "<...>self_attn."
        self.q_a_proj = ReplicatedLinear(
            self.hidden_size, self.q_lora_rank, bias=False, quant_config=qc,
            prefix=base + "q_a_proj")
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False, quant_config=qc,
            prefix=base + "kv_a_proj_with_mqa")
        del self.fused_qkv_a_proj_with_mqa
        self.has_fused_proj = False
        self.is_packed_weight = False
        self._use_min_latency_fused_a_gemm = False

    DeepseekV2AttentionMLA.__init__ = _attn_init

    # Model-level: stop glm5_next load_weights from manual-cat fusing q_a_proj + kv_a_proj for exl3. (DeepSeek-V4 uses
    # its own fuse path controlled by SGLANG_OPT_FUSE_WQA_WKV, so we deliberately do not touch it here.)
    patched = ["DeepseekV2AttentionMLA"]
    for mod, cls in (("sglang.srt.models.glm5_next", "Glm5NextForConditionalGeneration"),):
        try:
            m = __import__(mod, fromlist=[cls])
            k = getattr(m, cls)
        except Exception:
            continue
        _orig = k.__init__
        def _mk(orig):
            def _init(self, *a, **kw):
                orig(self, *a, **kw)
                qc = getattr(self, "quant_config", None)
                try:
                    if qc is not None and qc.get_name() == "exl3" and getattr(self, "fuse_qkv_a_proj", False):
                        self.fuse_qkv_a_proj = False
                except Exception:
                    pass
            return _init

        k.__init__ = _mk(_orig)
        patched.append(cls)

    _done = True
    logger.info("exl3xpu: mla_unfuse installed (exl3 MLA q_a/kv_a unfused): %s", ", ".join(patched))
