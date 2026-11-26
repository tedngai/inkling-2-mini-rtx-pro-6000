#!/usr/bin/env python3
"""Minimal acceptance checks against the adapted vLLM server."""
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("MODEL_ID", "inkling-2-mini")


def post(path, payload, timeout=600):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.read().decode()


def wait_ready(seconds=900):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            get("/health", timeout=5)
            return True
        except Exception:
            time.sleep(5)
    return False


def main():
    if not wait_ready():
        print("server never became healthy", file=sys.stderr)
        return 1

    models = json.loads(get("/v1/models"))
    ids = [m["id"] for m in models.get("data", [])]
    assert MODEL in ids, ids
    print(f"models: {ids}")

    prompt = "What is 19 + 23? Answer with just the number."
    started = time.perf_counter()
    out = post("/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "reasoning_effort": "low",
        "temperature": 0,
    })
    elapsed = time.perf_counter() - started
    choice = out["choices"][0]["message"]
    usage = out.get("usage", {})
    print(f"content: {choice.get('content')!r}")
    reasoning = choice.get("reasoning_content")
    if reasoning:
        print(f"reasoning ({len(reasoning)} chars): {reasoning[:200]!r}...")
    generated = usage.get("completion_tokens", 0)
    tps = generated / elapsed if elapsed else 0
    print(f"usage: {usage}")
    print(f"latency: {elapsed:.2f}s | {tps:.1f} output tok/s (incl. reasoning)")
    assert "42" in (choice.get("content") or ""), choice
    print("PASS: arithmetic + chat endpoint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
