#!/bin/bash
# glm53-flash-b70-ftx image entrypoint: GLM-5.3-Flash EXL3 3.05 bpw on ONE Intel Arc Pro B70 32 GB, experts in host
# RAM with the FTX CPU expert tier (run G262-I0s800c1k).
#   (no args)        the measured server command (below), every knob overridable through GLM_B70_* env
#   check            CPU-only self check: libraries load, hooks import (no GPU, no weights)
#   anything else    the SGLang server argv (exec'd as PID 1); the checkpoint path in it is swapped for the model view
# The checkpoint is read from $EXL3_MODEL_PATH (default /models/turboderp-GLM-5.3-Flash-exl3-3.05bpw, read-only).
set -euo pipefail
if [ "${1:-}" = check ]; then exec python3 /opt/glm-b70/serve/selfcheck.py; fi

SRC=${EXL3_MODEL_PATH:-/models/turboderp-GLM-5.3-Flash-exl3-3.05bpw}
VIEW=$(python3 /opt/glm-b70/serve/model_view.py "$SRC")
export EXL3_MODEL_PATH=$VIEW
if [ "${FTX_CPUS:-auto}" = auto ]; then FTX_CPUS=$(python3 /opt/glm-b70/serve/ftx_cpus.py); fi
export FTX_CPUS
echo "glm-b70: model $SRC -> $VIEW, FTX_CPUS=$FTX_CPUS, EXL3_MOE_SLOTS=${EXL3_MOE_SLOTS:-}" >&2

if [ "$#" -eq 0 ]; then
  P=${GLM_B70_PORT:-30260}
  set -- python3 -m sglang.launch_server --model-path "$SRC" --quantization exl3 --trust-remote-code --device xpu \
    --host 0.0.0.0 --port "$P" --served-model-name "${GLM_B70_SERVED_NAME:-glm53-flash}" --pp-size 1 \
    --reasoning-parser glm45 --tool-call-parser glm47 --disable-shared-experts-fusion \
    --kv-cache-dtype fp8_e4m3 --dtype bfloat16 --context-length "${GLM_B70_CTX:-262144}" \
    --mem-fraction-static "${GLM_B70_MEMFRAC:-0.88}" --chunked-prefill-size "${GLM_B70_CHUNK:-1024}" \
    --max-total-tokens "${GLM_B70_MAXTOT:-270336}" --max-running-requests "${GLM_B70_MAXRUN:-4}" \
    --max-mamba-cache-size "${GLM_B70_MAMBA:-24}" --page-size 64 --cuda-graph-backend-decode full \
    --cuda-graph-bs-decode 1 2 4
fi
args=()
for a in "$@"; do
  if [ "$a" = "$SRC" ] || [ "$a" = "${SRC%/}" ]; then args+=("$VIEW"); else args+=("$a"); fi
done
exec "${args[@]}"
