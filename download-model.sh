#!/usr/bin/env bash
# Download the pinned native NVFP4/MXFP8 checkpoint (no requantization).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
MODEL="${MODEL:-thinkingmachines/Inkling-2-Mini}"
# Revision measured in the referenced DGX Spark recipe.
REVISION="${REVISION:-b85c3cfa615fc413cc9af3379f491b963f3b586a}"

.venv/bin/python - "$MODEL" "$REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download

model, revision = sys.argv[1], sys.argv[2]
path = snapshot_download(
    repo_id=model,
    revision=revision,
    allow_patterns=["*.json", "*.jinja", "*.model", "*.safetensors"],
)
print("checkpoint:", path)
PY
