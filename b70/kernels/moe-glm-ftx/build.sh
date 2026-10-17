#!/usr/bin/env bash
# Build the GLM-5.3-Flash grouped-MoE library for the B70 (loaded by exl3xpu through EXL3_MOE_LIB).
# Source: exl3_moe.n128.sycl = trellis-serve exl3_moe.sycl + the a9/n124 power-of-two splits + the GLM swiglu clamp
# (EXL3_ACT_LIMIT=10) + the FTX change: an expert id of -1 is a pick taken by the CPU tier, so the kernels skip it
# (no zero-copy read, no cache admission) and write a zero row. exl3_esimd.h / fa_esimd.h are identical to
# b70tier/kernels/moe-n128/. CPU only: needs the image's oneAPI icpx and torch XPU, no GPU. Device code is spir64
# (JIT on first launch; the server warms every row count at start).
# These are the flags that built the measured kern/_moe_glm_ftx.so (campaign build_ftx.sh, 2026-10-15 22:47).
#   OUT (default ./_moe_glm_ftx.so)
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -euo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-$PWD/_moe_glm_ftx.so}
T=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
icpx -fsycl -fsycl-targets=spir64 -O3 -fPIC -std=c++17 -shared -fsycl-device-code-split=per_kernel \
  -D_GLIBCXX_USE_CXX11_ABI=1 -I"$T/include" -I"$T/include/torch/csrc/api/include" \
  -ffast-math -DEXL3_ACT_LIMIT=10.0f -I . -x c++ exl3_moe.n128.sycl -x none -o "$OUT.tmp" \
  -L"$T/lib" -Wl,-rpath,"$T/lib" -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu
mv -f "$OUT.tmp" "$OUT"
ls -la "$OUT"
