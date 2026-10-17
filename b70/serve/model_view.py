"""Print the model directory the server should load: the checkpoint itself, or a symlink view with the DSA kpool tensors
added to the safetensors index.

turboderp/GLM-5.3-Flash-exl3 (3.05bpw branch, 332ab457) ships kpool_aux.safetensors (22 tensors:
layers.N.self_attn.indexer.index_kpool_compress_{ape,gate}), but model.safetensors.index.json does not list them, so
sglang's loader never reads that file. The campaign host added the file to the index as model-00016-of-00016.safetensors
(a byte-identical copy of kpool_aux.safetensors, checked 2026-10-17). /models is mounted read-only here, so instead of
writing into it this builds VIEW_DIR: one symlink per checkpoint file plus an index that maps those 22 tensors to
kpool_aux.safetensors. If the index already lists them, the checkpoint is used as is.

usage: python3 model_view.py SRC [VIEW_DIR]   (VIEW_DIR default: $GLM_B70_VIEW or /tmp/glm53-model)
"""
import json, os, struct, sys

AUX = "kpool_aux.safetensors"


def tensor_names(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return [k for k in json.loads(f.read(n)) if k != "__metadata__"]


def main():
    src = os.path.abspath(sys.argv[1])
    view = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("GLM_B70_VIEW", "/tmp/glm53-model")
    idx_path = os.path.join(src, "model.safetensors.index.json")
    aux = os.path.join(src, AUX)
    if not os.path.isfile(idx_path):
        sys.exit(f"model_view: {idx_path} not found (mount the checkpoint at {src})")
    idx = json.load(open(idx_path))
    wm = idx["weight_map"]
    if not os.path.isfile(aux):
        print(f"model_view: no {AUX} in {src}; using it as is", file=sys.stderr)
        print(src); return
    names = tensor_names(aux)
    missing = [k for k in names if k not in wm]
    if not missing:
        print(src); return
    os.makedirs(view, exist_ok=True)
    for name in os.listdir(src):
        if name == "model.safetensors.index.json" or name.startswith("."):
            continue
        dst = os.path.join(view, name)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.join(src, name), dst)
    for k in missing:
        wm[k] = AUX
    size = os.path.getsize(aux)
    if isinstance(idx.get("metadata"), dict) and "total_size" in idx["metadata"]:
        idx["metadata"]["total_size"] += size
    tmp = os.path.join(view, "model.safetensors.index.json.tmp")
    with open(tmp, "w") as f:
        json.dump(idx, f, indent=2)
    os.replace(tmp, os.path.join(view, "model.safetensors.index.json"))
    print(f"model_view: {view} = {src} + {len(missing)} kpool tensors from {AUX} in the index", file=sys.stderr)
    print(view)


if __name__ == "__main__":
    main()
