#!/usr/bin/env python
"""Encoder forward latency of GSXA and the four compared attention types.

Every model is the ``base`` configuration with random weights (speed does not depend
on their values) and differs only in ``attention_type``. Timed: the encoder forward
(``last_hidden_state``, no vocabulary projection) on full-length inputs under
``torch.inference_mode``.

All models are built (and compiled) first. Each of ``--rounds`` rounds then times
every model -- the median of ``--iters`` forwards -- in an order rotated from round to
round, so drift in GPU clocks or temperature falls on all models alike. A model's
latency is the mean over rounds with a 95% Student-t interval; its change relative to
GQA is computed per round (paired) and summarised the same way.

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

# two-sided 95% Student-t critical values by degrees of freedom
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
       9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
       19: 2.093, 24: 2.064, 29: 2.045}


def ci95(xs):
    """(mean, 95% half-width) of ``xs``."""
    n = len(xs)
    if n < 2:
        return xs[0], float("nan")
    t = T95.get(n - 1) or T95[max(k for k in T95 if k <= n - 1)]
    return statistics.mean(xs), t * statistics.stdev(xs) / n ** 0.5


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--types", nargs="+", default=list(ATTENTION_TYPES))
    p.add_argument("--mode", default="compile", choices=["compile", "eager"])
    p.add_argument("--shape", default="32x1024", help="batch x length")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def median_ms(fn, ids, device, iters):
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
    B, T = (int(x) for x in args.shape.split("x"))
    env = {"device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
           "torch": torch.__version__, "cuda": torch.version.cuda, "dtype": args.dtype,
           "mode": args.mode, "shape": args.shape, "rounds": args.rounds,
           "warmup": args.warmup, "iters": args.iters}
    print(env, flush=True)
    ids = torch.randint(5, 50000, (B, T), device=device)

    fns = {}
    with torch.inference_mode():
        for kind in args.types:
            torch.manual_seed(0)
            model = GSXAModel(GSXAConfig(attention_type=kind)).to(device=device, dtype=dtype)
            fn = torch.compile(model.eval()) if args.mode == "compile" else model.eval()
            for _ in range(args.warmup):          # compiles on the first call
                fn(input_ids=ids)
            fns[kind] = fn
        rounds = {kind: [] for kind in args.types}
        for r in range(args.rounds):
            order = args.types[r % len(args.types):] + args.types[:r % len(args.types)]
            for kind in order:
                rounds[kind].append(median_ms(fns[kind], ids, device, args.iters))
            print(f"round {r + 1}/{args.rounds}: "
                  + "  ".join(f"{k} {rounds[k][-1]:.2f}" for k in args.types), flush=True)

    rows = []
    for kind in args.types:
        mean, half = ci95(rounds[kind])
        row = {"type": kind, "round_ms": rounds[kind], "ms": mean, "ms_ci95": half}
        if "gqa" in rounds and kind != "gqa":
            rel = [100.0 * (a / b - 1.0) for a, b in zip(rounds[kind], rounds["gqa"])]
            row["vs_gqa_pct"], row["vs_gqa_pct_ci95"] = ci95(rel)
        rows.append(row)
    out = Path(args.out or ROOT / "benchmarks" / "results" / f"{device.type}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"env": env, "rows": rows}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
