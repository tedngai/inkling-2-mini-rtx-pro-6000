# Inkling-2-Mini on 1× RTX Pro 6000 Blackwell

Run **Thinking Machines' Inkling-2-Mini** on a single NVIDIA RTX Pro 6000 Blackwell
(96 GiB VRAM, sm_120) without Docker. This is a port of
[`tedngai/inkling-2-mini-dgx-spark`](https://github.com/tedngai/inkling-2-mini-dgx-spark):
the native NVFP4/MXFP8 checkpoint is served through a vLLM nightly with FP8 KV,
DSpark speculative decoding, 8 admission slots and a 131,072-token window.

The API binds to `0.0.0.0:8000` and has **no authentication** — see the security note below.

## 1. Current status

| Item | Verified result or current limit |
| --- | --- |
| Model precision | `thinkingmachines/Inkling-2-Mini`: NVFP4 experts, MXFP8 last four layers. No local requantization. |
| Checkpoint revision | `b85c3cfa615fc413cc9af3379f491b963f3b586a` (the revision measured in the DGX Spark recipe). |
| Hardware | 1 × RTX Pro 6000 Blackwell DC-4-96Q, 96 GiB VRAM, compute capability 12.0. TP1. |
| Engine | vLLM `0.34.1rc1.dev159+ged686ccec` (cu132 nightly), Python 3.12, torch 2.14.0+cu132, transformers 5.24.1. |
| Driver / CUDA | NVIDIA 580.159.03 / CUDA 13.2 runtime. |
| API model ID | `inkling-2-mini` |
| Served context | **131,072 tokens** per request. Native window is 1,048,576; not enabled here. |
| KV capacity | **1,449,079 tokens** with FP8 KV at 0.90 utilization — **11.06×** the 128K window. |
| Admission slots | `max-num-seqs 8`; not separately load-tested at 8. |
| Speculative decoding | DSpark, 7 draft tokens, probabilistic draft sampling. |
| Speed | 320 completion tokens in 1.78 s (~180 tok/s) on a short prompt; see [`results/`](results/). |
| Features | Text, reasoning parser, tool-call parser, single-image vision (SDPA fallback), OpenAI-compatible API. |
| Smoke checks | `/health`, `/v1/models`, `19 + 23 → 42`, red-image → `Red` all pass. |
| Reliability | No one-hour soak, 400K-context test, multi-user trace or rollback test yet. |

## 2. Start the service

Requirements: one Blackwell GPU with ~24 GiB free VRAM for weights plus room for KV
(this recipe uses ~88 GiB of 96 GiB), 20 GB free disk for the checkpoint, and
network access to PyPI/Hugging Face. Docker is **not** required.

```bash
git clone https://github.com/tedngai/inkling-2-mini-rtx-pro-6000.git
cd inkling-2-mini-rtx-pro-6000
./install.sh           # once: local Python 3.12 + vLLM nightly + sm120 vision patch
./download-model.sh    # once: ~20 GB native NVFP4/MXFP8 weights
./start.sh             # background-style foreground process; logs to logs/vllm-8000.log
```

On later boots:

```bash
./start.sh
./stop.sh
```

The API binds to port **8000** on all interfaces. There is no authentication in this recipe.

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini",
       "messages":[{"role":"user","content":"What is 19 + 23?"}],
       "max_tokens":256,
       "reasoning_effort":"low"}'
```

From another machine on the LAN (replace the IP with this host's address):

```bash
curl -s http://198.51.100.12:8000/v1/models
```

> **Security:** anyone who can reach port 8000 can use the API. Keep the server on a
> trusted LAN, firewall the port, or put it behind an authenticating reverse proxy.

## 3. Configuration

| Setting | DGX Spark recipe | This recipe | Why |
| --- | --- | --- | --- |
| Runtime | Docker image `vllm/vllm-openai:inkling-2-mini` | native vLLM nightly in `.venv` | Docker is not installed on this host |
| Python | container runtime | local CPython 3.12.16 via `uv` | system Python 3.10 is too old for the Inkling-capable nightly |
| vLLM wheel | cu130 nightly pinned by image | `0.34.1rc1.dev159+ged686ccec`, cu132 nightly | cu130 nightly (`dev156`) lacks the native `Inkling2Loop` model; cu132 has it |
| GPU | GB10, 128 GB unified | RTX Pro 6000 Blackwell, 96 GiB, sm_120 | smaller VRAM pool |
| `max-model-len` | 409,600 | **131,072** | 400K FP8 KV consumed ~88 GiB on the Spark; on 96 GiB that leaves no headroom for weights/activations/CUDA graphs |
| `gpu-memory-utilization` | 0.90 | 0.90 | same fraction of a smaller pool |
| `kv-cache-dtype` | fp8 | fp8 | kept |
| `max-num-batched-tokens` | 2048 | 2048 | kept: DSpark floor and avoids hybrid-cache padding |
| `max-num-seqs` | 8 | 8 | kept |
| Prefix caching | on | on | kept |
| `inkling` reasoning + tool parsers | on | on | kept |
| DSpark | 7 tokens, probabilistic | 7 tokens, probabilistic | kept |
| Vision attention | bundled FA2/FA4 | FA4 on SM100; **SDPA on sm_120** | r580.159 cannot JIT the cu132 FA2 PTX; FA3 has no sm_120 kernel |
| Host binding | 0.0.0.0:8000 in container | 0.0.0.0:8000 | LAN-accessible by default |

The KV pool holds 1,449,079 tokens; 131,072 per request. Raising the window to
262,144 would still leave ~5.5 full sessions in the pool:
`MAX_MODEL_LEN=262144 ./start.sh`. This was not benchmarked.

### Compatibility patches applied by `install.sh`

- `patch_sm120_vision.py` swaps only the vision encoder's FA2 call for per-image
  PyTorch SDPA on compute capability 120. Decoder attention uses Inkling2's
  native sm120 kernels and is unchanged.
- vLLM also logs two non-fatal driver limitations on this host: the extensible
  KV cache is disabled (CUDA error 801, so KV is sized from profiling estimates),
  and the FlashInfer sampler falls back to the native sampler. Neither blocked serving.

## 4. Measured results

| Input | Concurrency | Output | Wall time | Throughput | Notes |
| --- | --- | --- | --- | --- | --- |
| 29 tokens (`19 + 23`) | 1 | 33 tokens (25 reasoning) | 1.95 s | 16.9 tok/s | cold/HTTP-dominated smoke request |
| ~short prose prompt | 1 | 320 tokens | 1.78 s | **179.9 tok/s** | `reasoning_effort=low`, `temperature=0` |
| 128×128 red PNG | 1 | `Red` | — | — | single-image vision via SDPA fallback |

KV pool: 1,449,079 tokens (65.78 GiB) at 0.90 utilization; ~88 GiB total GPU memory
in use while serving. Full per-cell files are in [`results/`](results/).

## 5. Reproduce the checks

```bash
python3 benchmarks/smoke-test.py     # health, models, arithmetic
```

The script expects the server at `http://127.0.0.1:8000`. To exercise the API from
another host, point any OpenAI client at `http://<host-ip>:8000/v1` with
`model="inkling-2-mini"`.

## 6. Provenance and limitations

| Component | Source |
| --- | --- |
| Model | [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), revision `b85c3cfa...` |
| Original recipe | [tedngai/inkling-2-mini-dgx-spark](https://github.com/tedngai/inkling-2-mini-dgx-spark) |
| vLLM runtime | [wheels.vllm.ai nightly](https://wheels.vllm.ai/nightly/cu132/), `0.34.1rc1.dev159+ged686ccec` |
| Reference docs | [vLLM recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini), [HF launch blog](https://huggingface.co/blog/inkling2mini) |
| Hardware | 1 × NVIDIA RTX Pro 6000 Blackwell DC-4-96Q (sm_120), driver 580.159.03 |

Weights are downloaded at runtime and retain their upstream Apache-2.0 license.
This repository contains no model weights. The published numbers are smoke-level
acceptance checks, not a quality score or a full throughput matrix; long-context
(≥256K) behavior, 8-way concurrency, tool-call round trips, 1M context and soak
tests remain open.
