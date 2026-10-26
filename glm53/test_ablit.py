"""CPU checks for ablit.py (torch + safetensors, no GPU, no exllamav3): spec resolve and direction lookup, the
projection against the weight edit it replaces, and the hook placement on stand-in decoder blocks.
    python3 glm53/test_ablit.py
    ABLIT_REAL_DIR=<dir with uncensor_spec.json + uncensor_direction.st> python3 glm53/test_ablit.py   (+ published files)
"""
import json, os, sys, tempfile
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ablit

H, NL = 64, 6
PUB_SHA = {"spec": "fe9989cbce800f941ea46eac5200687fb961c51af607b4df9506c59c8b8f4d97",
           "direction": "7ca504e1fa94612c33451856e0bd33c483dd981462af7d9741b098772610c608"}


def raises(exc, f, *a):
    try:
        f(*a)
    except exc as e:
        return str(e)
    raise AssertionError(f"{f.__name__} did not raise {exc.__name__}")


def write_spec(d, name="uncensor_spec.json", attn_w=None, mlp_w=None, **extra):
    spec = {"hidden": H, "n_layers": NL, "attn_w": attn_w or [0, 0, 0, 2.0, 2.1, 2.2],
            "mlp_w": mlp_w or [0, 0.9, 1.2, 1.4, 0, 0], **extra}
    p = os.path.join(d, name)
    json.dump(spec, open(p, "w"))
    return p


def write_dir(path, r):
    save_file({"r": r}, path)


def env(v):
    if v is None:
        os.environ.pop("EXL3_ABLIT_RUNTIME", None)
    else:
        os.environ["EXL3_ABLIT_RUNTIME"] = v


def test_resolve_and_load():
    with tempfile.TemporaryDirectory() as d:
        env(None)
        assert ablit.resolve(d) == (None, "none") and ablit.load(d) is None
        spec = write_spec(d)
        r = torch.randn(H) * 3
        write_dir(os.path.join(d, "uncensor_direction.st"), r)
        assert ablit.resolve(d) == (spec, "bundled")
        for v in ("off", "OFF", "0", "false", "no", "none"):
            env(v)
            assert ablit.resolve(d) == (None, "off"), v
        env(None)
        s = ablit.load(d)
        assert s["source"] == "bundled" and s["r_file"].endswith("uncensor_direction.st")
        assert abs(float(s["r"].norm()) - 1.0) < 1e-6 and torch.allclose(s["r"], (r / r.norm()).float())
        # bundled prefers uncensor_direction.st, an explicit path prefers <spec>.safetensors
        write_dir(os.path.join(d, "uncensor_spec.safetensors"), -r)
        assert ablit.load(d)["r_file"].endswith("uncensor_direction.st")
        env(spec)
        s = ablit.load(d)
        assert s["source"] == "env" and s["r_file"].endswith("uncensor_spec.safetensors")
        # r_file in the spec wins, relative to the spec
        write_dir(os.path.join(d, "other.st"), r)
        env(write_spec(d, "custom.json", r_file="other.st"))
        assert ablit.load(d)["r_file"] == os.path.join(d, "other.st")
        # errors stop the load and name the file
        env(os.path.join(d, "missing.json"))
        assert "missing.json" in raises(FileNotFoundError, ablit.load, d)
        open(os.path.join(d, "bad.json"), "w").write("{")
        env(os.path.join(d, "bad.json"))
        assert "not valid JSON" in raises(ValueError, ablit.load, d)
        json.dump({"hidden": H}, open(os.path.join(d, "short.json"), "w"))
        env(os.path.join(d, "short.json"))
        assert "missing field" in raises(ValueError, ablit.load, d)
        env(write_spec(d, "short_w.json", attn_w=[1.0]))
        assert "shorter than n_layers" in raises(ValueError, ablit.load, d)
        write_dir(os.path.join(d, "wide.st"), torch.randn(H + 1))
        env(write_spec(d, "wide.json", r_file="wide.st"))
        assert "!= hidden" in raises(ValueError, ablit.load, d)
        write_dir(os.path.join(d, "zero.st"), torch.zeros(H))
        env(write_spec(d, "zero.json", r_file="zero.st"))
        assert "norm" in raises(ValueError, ablit.load, d)
        env(write_spec(d, "nodir.json"))   # no nodir.safetensors; uncensor_direction.st sits next to it -> found
        assert ablit.load(d)["r_file"].endswith("uncensor_direction.st")
        os.remove(os.path.join(d, "uncensor_direction.st"))
        env(write_spec(d, "nodir2.json"))
        assert "direction file not found" in raises(FileNotFoundError, ablit.load, d)
    env(None)
    print("resolve / load: ok")


def test_projection_is_the_weight_edit():
    """project(W h) == (W - w r r^T W) h, the edit the spec's "apply" field describes, for fp32 rows."""
    torch.manual_seed(0)
    r = torch.nn.functional.normalize(torch.randn(H, dtype=torch.double), dim=0)
    ablit.SPEC = {"r": r.float()}
    ablit._R.clear()
    W = torch.randn(H, 48, dtype=torch.double)
    h = torch.randn(3, 5, 48, dtype=torch.double)
    for w in (0.9, 1.0, 2.2, -0.5):
        y = (h @ W.T).float()
        ref = h @ (W - w * torch.outer(r, r) @ W).T
        out = ablit.project(y, w)
        assert out.data_ptr() == y.data_ptr(), "contiguous fp32 is projected in place"
        assert torch.allclose(out.double(), ref, rtol=1e-5, atol=1e-4), (w, (out.double() - ref).abs().max())
        # component along r scales by (1 - w), the orthogonal part is unchanged
        y0 = (h @ W.T)
        assert torch.allclose(out.double() @ r, (1 - w) * (y0 @ r), atol=1e-4)
        o_perp = out.double() - torch.outer((out.double() @ r).flatten(), r).view_as(out)
        y_perp = y0 - torch.outer((y0 @ r).flatten(), r).view_as(y0)
        assert torch.allclose(o_perp, y_perp, atol=1e-4)
    # non-contiguous input: a projected copy, same values
    y = (h @ W.T).float().transpose(0, 1)
    ref = ablit.project(y.contiguous().clone(), 2.0)
    out = ablit.project(y, 2.0)
    assert out.is_contiguous() and torch.equal(out, ref)
    # fp16 rows (not GLM-5.3's dtype, kept working)
    y = (h @ W.T).half()
    out = ablit.project(y.clone(), 1.0)
    before = float((y.float() @ r.float()).abs().max())
    assert out.dtype == torch.half and float((out.float() @ r.float()).abs().max()) < 1e-2 * before
    print("projection == weight edit: ok")


class Sub:
    """Stand-in attn / mlp: a linear writer, fp32 output (exllamav3 builds GLM-5.3's with out_dtype=torch.float)."""
    def __init__(self, W):
        self.W, self.raw = W, None

    def forward(self, x, params, out_dtype=None):
        y = (x.float() @ self.W.T).float()
        self.raw = y.clone()   # the writer output before any hook
        return y


class TransformerBlock:
    """Stand-in with stock exllamav3 1.5.1's call pattern: attn.forward, prefill early return, mlp.forward."""
    def __init__(self, key, attn, mlp, device="cpu"):
        self.key, self.attn, self.mlp, self.device = key, attn, mlp, device
        self.seen = {}

    def forward(self, x, params):
        y = self.attn.forward(x, params)
        self.seen["attn"] = y
        if params.get("prefill") and not params.get("export_state_layers"):
            return x
        x = x + y
        y = self.mlp.forward(x, params)
        self.seen["mlp"] = y
        return x + y


class Model:
    def __init__(self, blocks, hidden=H):
        self.modules = [object()] + blocks + [object()]
        self.config = type("C", (), {"hidden_size": hidden})()


def make_model(n=NL, extra=()):
    torch.manual_seed(1)
    blocks = [TransformerBlock(f"model.language_model.layers.{i}", Sub(torch.randn(H, H) * 0.1),
                               Sub(torch.randn(H, H) * 0.1)) for i in range(n)]
    for k in extra:
        blocks.append(TransformerBlock(k, Sub(torch.randn(H, H) * 0.1), Sub(torch.randn(H, H) * 0.1)))
    return Model(blocks)


def test_install():
    with tempfile.TemporaryDirectory() as d:
        env(None)
        write_spec(d, attn_w=[0, 0, 0, 2.0, 2.1, 1e-12], mlp_w=[0, 0.9, 1.2, 1.4, 0, 0])
        write_dir(os.path.join(d, "uncensor_direction.st"), torch.nn.functional.normalize(torch.randn(H), dim=0))
        ablit.load(d)
        r = ablit.SPEC["r"]
        # an MTP-numbered block (layers.NL) and an mtp.* key are never hooked
        model = make_model(extra=(f"model.language_model.layers.{NL}", "mtp.layers.0"))
        n = ablit.install(model)
        assert ablit.SITES == {"attn": [3, 4], "mlp": [1, 2, 3]} and n == 5, ablit.SITES   # 1e-12 counts as 0
        blocks = [b for b in model.modules if isinstance(b, TransformerBlock)]
        x = torch.randn(2, 3, H)

        def expect(raw, w):
            return raw - w * (raw @ r).unsqueeze(-1) * r

        for b in blocks:
            L = ablit._layer(b)
            b.forward(x.clone(), {})
            for site in ("attn", "mlp"):
                raw, seen = getattr(b, site).raw, b.seen[site]
                if L in ablit.SITES[site] and not b.key.startswith("mtp"):
                    w = ablit.SPEC["spec"][f"{site}_w"][L]
                    assert torch.allclose(seen, expect(raw, w), atol=1e-5), (b.key, site)
                    assert not torch.allclose(seen, raw, atol=1e-3), (b.key, site)
                else:
                    assert torch.equal(seen, raw), (b.key, site)
        # prefill on the last KV block: the discarded attn output is not projected; export_state forwards still are
        b = blocks[3]
        b.forward(x.clone(), {"prefill": True})
        assert torch.equal(b.seen["attn"], b.attn.raw)
        b.forward(x.clone(), {"prefill": True, "export_state_layers": [3]})
        assert torch.allclose(b.seen["attn"], expect(b.attn.raw, 2.0), atol=1e-5)
        # the mlp site has no prefill exception (its block only runs it when the output is used)
        b.forward(x.clone(), {})
        assert torch.allclose(b.seen["mlp"], expect(b.mlp.raw, 1.4), atol=1e-5)
        # summary
        s = ablit.summary()
        assert s["active"] and set(s["attn"]) == {"3", "4"} and set(s["mlp"]) == {"1", "2", "3"}
        # wrong model width stops the install
        assert "hidden_size" in raises(ValueError, ablit.install, Model([], hidden=H * 2))
    ablit.SPEC = None
    assert ablit.install(make_model()) == 0 and ablit.summary() == {"active": False}
    print("install / hook placement: ok")


def test_published(d):
    """The two published files: hashes, shape, and the 34 blocks / 49 sites the model card reports."""
    env(None)
    s = ablit.load(d)
    assert s["sha256"] == PUB_SHA, s["sha256"]
    assert s["spec"]["hidden"] == 4096 and s["spec"]["n_layers"] == 45 and abs(s["r_norm"] - 1.0) < 1e-6
    torch.manual_seed(2)
    blocks = [TransformerBlock(f"model.language_model.layers.{i}", Sub(torch.zeros(1, 1)), Sub(torch.zeros(1, 1)))
              for i in range(45)]
    blocks.append(TransformerBlock("model.language_model.layers.45", Sub(torch.zeros(1, 1)), Sub(torch.zeros(1, 1))))
    n = ablit.install(Model(blocks, hidden=4096))
    assert ablit.SITES["attn"] == list(range(24, 45)) and ablit.SITES["mlp"] == list(range(11, 39)), ablit.SITES
    assert n == 49 and len(set(ablit.SITES["attn"]) | set(ablit.SITES["mlp"])) == 34
    ws = [s["spec"]["attn_w"][L] for L in ablit.SITES["attn"]] + [s["spec"]["mlp_w"][L] for L in ablit.SITES["mlp"]]
    assert 2.13 < min(ws[:21]) and max(ws[:21]) < 2.21 and 0.88 < min(ws[21:]) and max(ws[21:]) < 1.50
    ablit.SPEC = None
    print("published spec (34 blocks, 49 sites, hashes): ok")


if __name__ == "__main__":
    test_resolve_and_load()
    test_projection_is_the_weight_edit()
    test_install()
    if os.environ.get("ABLIT_REAL_DIR"):
        test_published(os.environ["ABLIT_REAL_DIR"])
    print("all ok")
