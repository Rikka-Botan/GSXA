#!/usr/bin/env python
"""Encoder forward latency of GSXA and the four compared attention types.

Every model is the ``base`` configuration with random weights (speed does not depend
on their values) and differs only in ``attention_type``. Timed: the encoder forward
(``last_hidden_state``, no vocabulary projection) on full-length inputs, under
``torch.inference_mode``, the median of ``--iters`` forwards after ``--warmup``.

    python benchmarks/benchmark_inference.py                 # -> benchmarks/results/<device>.json
    python benchmarks/make_tables.py benchmarks/results/<device>.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gsxa import ATTENTION_TYPES, GSXAConfig, GSXAModel  # noqa: E402

SHAPES = ["1x512", "64x128", "32x512", "8x2048", "2x8192"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--types", nargs="+", default=list(ATTENTION_TYPES))
    p.add_argument("--modes", nargs="+", default=["eager", "compile"])
    p.add_argument("--shapes", nargs="+", default=SHAPES)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def time_forward(fn, ids, device, warmup, iters):
    for _ in range(warmup):
        fn(input_ids=ids)
    times = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn(input_ids=ids)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return statistics.median(times) * 1e3


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    env = {"device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
           "torch": torch.__version__, "cuda": torch.version.cuda, "dtype": args.dtype,
           "warmup": args.warmup, "iters": args.iters}
    print(env, flush=True)
    rows = []
    for kind in args.types:
        torch.manual_seed(0)
        model = GSXAModel(GSXAConfig(attention_type=kind)).to(device=device, dtype=dtype).eval()
        for mode in args.modes:
            fn = torch.compile(model) if mode == "compile" else model
            for shape in args.shapes:
                B, T = (int(x) for x in shape.split("x"))
                ids = torch.randint(5, 50000, (B, T), device=device)
                with torch.inference_mode():
                    ms = time_forward(fn, ids, device, args.warmup, args.iters)
                rows.append({"type": kind, "mode": mode, "shape": shape, "ms": ms,
                             "ktok_s": B * T / ms})
                print(f"{kind:13s} {mode:7s} {shape:>7s}  {ms:8.2f} ms  {B * T / ms:6.0f} k tok/s",
                      flush=True)
            if mode == "compile":
                torch._dynamo.reset()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out = Path(args.out or ROOT / "benchmarks" / "results" / f"{device.type}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"env": env, "rows": rows}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
