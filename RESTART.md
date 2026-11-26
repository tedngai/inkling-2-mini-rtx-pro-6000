# Restart / boot checklist

This host has no Docker. The runtime lives in a local venv and the checkpoint in
the Hugging Face cache.

```bash
cd ~/Documents/inkling-2-mini-rtx-repo

# If .venv or .uv-python was deleted:
./install.sh

# If ~/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini is missing:
./download-model.sh

# Start (binds 0.0.0.0:8000 by default, no auth):
./start.sh

# Check:
curl -s http://127.0.0.1:8000/health
python3 benchmarks/smoke-test.py

# Stop:
./stop.sh
```

Useful overrides:

| Variable | Default | Notes |
| --- | --- | --- |
| `PORT` | 8000 | also selects the log file `logs/vllm-${PORT}.log` |
| `HOST` | 0.0.0.0 | set to 127.0.0.1 to keep the API local-only |
| `MAX_MODEL_LEN` | 131072 | try 262144 only after checking the KV pool in the log |
| `GPU_MEM_UTIL` | 0.90 | leave headroom for driver/display use |
| `KV_DTYPE` | fp8 | BF16 roughly halves KV tokens |
| `BATCH_TOKENS` | 2048 | DSpark floor; larger values shrink the KV pool |
| `MAX_NUM_SEQS` | 8 | live admission cap |

If LAN clients cannot connect, check the host firewall, e.g.
`sudo ufw allow 8000/tcp`. Remember the API is unauthenticated.
