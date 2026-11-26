#!/usr/bin/env bash
# Serve Inkling-2-Mini on one RTX Pro 6000 Blackwell.
# Adapted from the DGX Spark recipe: 128K window instead of 400K, native vLLM
# nightly instead of Docker, sm120 vision SDPA patch applied by install.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p logs

MODEL="${MODEL:-thinkingmachines/Inkling-2-Mini}"
PORT="${PORT:-8000}"
# Bind all interfaces so other machines on the LAN can reach the API.
# There is no authentication in this recipe: firewall or proxy accordingly.
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
KV_DTYPE="${KV_DTYPE:-fp8}"
BATCH_TOKENS="${BATCH_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "A server is already answering on port ${PORT}; not starting a second one." >&2
  exit 1
fi

exec .venv/bin/vllm serve "$MODEL" \
  --host "$HOST" --port "$PORT" \
  --served-model-name inkling-2-mini \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --kv-cache-dtype "$KV_DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$BATCH_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --enable-prefix-caching \
  --reasoning-parser inkling \
  --enable-auto-tool-choice \
  --tool-call-parser inkling \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  > "logs/vllm-${PORT}.log" 2>&1
