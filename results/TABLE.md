# Measured cells — RTX Pro 6000 Blackwell

| Cell | Input | Concurrency | Output | Wall time | Throughput | Pass |
| --- | --- | --- | --- | --- | --- | --- |
| smoke-arithmetic | 29 tokens | 1 | 33 tokens (25 reasoning) | 1.95 s | 16.9 tok/s | yes |
| decode-short | short prose | 1 | 320 tokens | 1.78 s | 179.9 tok/s | yes |
| vision-red | 128×128 PNG | 1 | `Red` | — | — | yes |

KV pool: 1,449,079 tokens (65.78 GiB) at `gpu-memory-utilization=0.90`, FP8 KV.
Total GPU memory while serving: ~88 GiB / 96 GiB.

These are smoke-level acceptance measurements, not a quality score or a full
stress matrix. Long contexts, 8-way concurrency and soaks are not covered.
