#!/usr/bin/env bash
# One-command setup on a fresh GPU pod (phases 4-6). NOT RUN from this repo yet.
#
#   bash scripts/pod_setup.sh            # from the repo root
#
# Mirrors September's infra/bootstrap.sh: torch 2.14.0 from the cu130 index
# (needs NVIDIA driver >= 580), transformers 5.17.0, kernels 0.16.1 for native
# MXFP4. One 32 GB card is enough (September: 12.8 GiB loaded, 22.7 GiB peak
# on a 32k-token prompt).
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
VENV="${VENV:-$WORKSPACE/venv-wbm}"
export HF_HOME="${HF_HOME:-$WORKSPACE/hf}"
SEPT_DIR="${SEPT_DIR:-$WORKSPACE/linuxarena-monitor-readout}"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.12 "$VENV"
PY="$VENV/bin/python"
uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu130 torch==2.14.0
uv pip install --python "$PY" -e ".[hf,control-arena,dev]" kernels==0.16.1 accelerate==1.15.0

[ -d "$SEPT_DIR" ] || git clone https://github.com/thareqyusuf/linuxarena-monitor-readout "$SEPT_DIR"
git -C "$SEPT_DIR" checkout 788c25f3ab8ab6b00b272367d7f61796d26e82ce

"$PY" - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "transformers", transformers.__version__)
assert torch.cuda.is_available(), "no CUDA device"
print(torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
PY
"$VENV/bin/python" -m pytest -q

cat <<EOF

Setup done. Activate with:  source $VENV/bin/activate
Then (see README "GPU phases"):
  export HF_HOME=$HF_HOME PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  wbm validate-september --cache <checkpoints/local-monitor-activations> \\
      --september-repo $SEPT_DIR --n 300
EOF
