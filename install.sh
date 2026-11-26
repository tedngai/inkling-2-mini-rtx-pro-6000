#!/usr/bin/env bash
# Install the Inkling-2-capable vLLM nightly on Python 3.12, locally.
# No Docker and no system Python changes required.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p logs
export UV_PYTHON_INSTALL_DIR="$ROOT/.uv-python"

if [[ -x ./uv ]]; then
  UV=./uv
elif command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
else
  python3 -m pip install --user uv
  UV="$HOME/.local/bin/uv"
fi

# The cu130 nightly (dev156) has no native Inkling2Loop model; this cu132 build
# does. Its FA2 vision kernel is patched for the r580.159 driver below.
VLLM_VERSION="${VLLM_VERSION:-0.34.1rc1.dev159+ged686ccec}"

"$UV" python install 3.12
"$UV" venv --python 3.12 --clear .venv
"$UV" pip install --python .venv/bin/python --pre "vllm==$VLLM_VERSION" \
  --extra-index-url https://wheels.vllm.ai/nightly/cu132 \
  --extra-index-url https://download.pytorch.org/whl/cu132 \
  --index-strategy unsafe-best-match 2>&1 | tee logs/install.log

# r580.159 cannot JIT the bundled FA2 PTX on sm_120; FA3 has no sm_120 kernel.
.venv/bin/python patch_sm120_vision.py

.venv/bin/python -c 'import vllm, torch, transformers; print("vllm", vllm.__version__); print("torch", torch.__version__, "cuda", torch.version.cuda, "cap", torch.cuda.get_device_capability()); print("transformers", transformers.__version__)'
