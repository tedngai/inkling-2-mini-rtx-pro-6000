#!/usr/bin/env python3
"""Local compatibility patch for Inkling-2 vision attention on sm_120.

The bundled vllm_flash_attn FA2 kernels emit PTX that r580.159 cannot JIT, and
FA3 has no sm_120 kernel. The vision encoder only needs plain non-causal
var-length attention, so use PyTorch SDPA per image on compute capability 120.
Decoder attention is untouched (it uses Inkling2's dedicated sm120 kernels).
"""
from pathlib import Path
import sys

import vllm

path = Path(vllm.__file__).parent / "models/inkling2/nvidia/vision.py"
src = path.read_text()
if "_sm120_varlen_sdpa" in src:
    print("already patched:", path)
    sys.exit(0)

helper = '''

def _sm120_varlen_sdpa(q, k, v, cu_seqlens, softmax_scale):
    """Non-causal var-length attention via SDPA (one call per image)."""
    import torch.nn.functional as F

    out = []
    for i in range(len(cu_seqlens) - 1):
        start = int(cu_seqlens[i])
        end = int(cu_seqlens[i + 1])
        if end <= start:
            continue
        qi = q[start:end].transpose(0, 1).unsqueeze(0)
        ki = k[start:end].transpose(0, 1).unsqueeze(0)
        vi = v[start:end].transpose(0, 1).unsqueeze(0)
        yi = F.scaled_dot_product_attention(
            qi, ki, vi, scale=softmax_scale, is_causal=False
        )
        out.append(yi.squeeze(0).transpose(0, 1))
    return torch.cat(out, dim=0) if out else torch.empty_like(q)
'''

anchor = "class Inkling2VisionAttention(nn.Module):"
assert anchor in src, "vision.py layout changed"
src = src.replace(anchor, helper + "\n\n" + anchor, 1)

old = '''        else:
            from vllm.vllm_flash_attn import flash_attn_varlen_func

            version = 2 if current_platform.is_device_capability_family(120) else 3
            out = flash_attn_varlen_func(q, k, v, fa_version=version, **kwargs)
'''
new = '''        elif current_platform.is_device_capability_family(120):
            out = _sm120_varlen_sdpa(q, k, v, metadata.cu_seqlens, self.head_dim**-0.5)
        else:
            from vllm.vllm_flash_attn import flash_attn_varlen_func

            out = flash_attn_varlen_func(q, k, v, fa_version=3, **kwargs)
'''
assert old in src, "attention call site changed"
src = src.replace(old, new, 1)
path.write_text(src)
print("patched:", path)
