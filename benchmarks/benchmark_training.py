#!/usr/bin/env python
"""Training-step cost of GSXA and the four compared attention types: forward and
backward latency, and peak GPU memory in each pass.

Every model is the ``base`` configuration with random weights and differs only in
``attention_type``. One step is the encoder forward with gradients enabled
(``last_hidden_state``, no vocabulary projection) followed by the backward from a fixed
random output gradient. No optimizer step and no MLM head: they cost the same for
every attention type.

Precision (``--precision``):

* ``autocast`` (default) -- fp32 weights and gradients, bf16 compute under
  ``torch.autocast``, as in mixed-precision training.
* ``bf16`` -- weights, gradients and compute all in bf16.

Latency. All models are built and compiled first; only the model being timed sits on
the GPU, the others wait in host memory (five models' activations do not fit a 16 GB
GPU at once, and an oversubscribed GPU pages to host memory, which can triple the step
time). Each of ``--rounds`` rounds then moves every model to the GPU, runs
``--visit_warmup`` untimed steps and times the median of ``--iters`` steps, in an
order rotated from round to round. The forward and the backward are timed
separately (the GPU is synchronised between them); "step" is their sum per step. The
summary is the mean over rounds with a 95% Student-t interval; the change relative to
GQA is computed per round (paired).

Memory. Each model is measured alone in a fresh subprocess, after a warm-up, from
``torch.cuda.max_memory_allocated`` / ``max_memory_reserved``:

* forward peak -- the peak during the forward (with gradients enabled),
* backward peak -- the peak during the following backward,
* both include the weights, the parameter gradients (allocated during the backward;
  freed with ``set_to_none`` before every step), the saved activations, the input ids
  and the output gradient; no optimizer state.

    python benchmarks/benchmark_training.py              # -> benchmarks/results/<name>.json
    python benchmarks/make_tables.py benchmarks/results/<name>.json
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_inference import ci95  # noqa: E402
from gsxa import ATTENTION_TYPES, GSXAConfig, GSXAModel  # noqa: E402

MiB = 2 ** 20


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--types", nargs="+", default=list(ATTENTION_TYPES))
    p.add_argument("--mode", default="compile", choices=["compile", "eager"])
    p.add_argument("--shape", default="32x1024", help="batch x length")
    p.add_argument("--precision", default="autocast", choices=["autocast", "bf16"])
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--visit_warmup", type=int, default=3)
    p.add_argument("--skip_memory", action="store_true")
    p.add_argument("--skip_latency", action="store_true")
    p.add_argument("--memory_only", default=None, help=argparse.SUPPRESS)  # subprocess
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


class Step:
    """One model with its compiled forward, inputs and output gradient."""

    def __init__(self, kind, args, device):
        B, T = (int(x) for x in args.shape.split("x"))
        torch.manual_seed(0)
        model = GSXAModel(GSXAConfig(attention_type=kind)).to(device)
        if args.precision == "bf16":
            model = model.to(torch.bfloat16)
        self.model = model.train()
        self.fn = torch.compile(self.model) if args.mode == "compile" else self.model
        self.autocast = args.precision == "autocast"
        self.ids = torch.randint(5, 50000, (B, T), device=device)
        self.grad = None

    def to(self, device):
        self.model.to(device)  # in place: the compiled graphs keep their parameters

    def zero_grad(self):
        self.model.zero_grad(set_to_none=True)

    def forward(self):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
            y = self.fn(input_ids=self.ids).last_hidden_state
        if self.grad is None:
            g = torch.Generator(device=y.device).manual_seed(1)
            self.grad = torch.randn(y.shape, dtype=y.dtype, device=y.device, generator=g)
        return y

    def backward(self, y):
        y.backward(self.grad)

    def warmup(self, n):
        for _ in range(n):  # compiles the forward and the backward on the first call
            self.zero_grad()
            self.backward(self.forward())
        self.zero_grad()


def time_step(step, iters):
    """Median forward / backward / step milliseconds over ``iters`` steps."""
    fwd, bwd, tot = [], [], []
    for _ in range(iters):
        step.zero_grad()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = step.forward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        step.backward(y)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        del y
        fwd.append((t1 - t0) * 1e3)
        bwd.append((t2 - t1) * 1e3)
        tot.append((t2 - t0) * 1e3)
    step.zero_grad()
    return statistics.median(fwd), statistics.median(bwd), statistics.median(tot)


def measure_memory(kind, args, device):
    """Peak memory (MiB) of one step of ``kind``; run in a fresh process."""
    step = Step(kind, args, device)
    weights = torch.cuda.memory_allocated(device)
    step.warmup(args.warmup)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    y = step.forward()
    torch.cuda.synchronize()
    fwd_alloc = torch.cuda.max_memory_allocated(device)
    fwd_res = torch.cuda.max_memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    step.backward(y)
    torch.cuda.synchronize()
    bwd_alloc = torch.cuda.max_memory_allocated(device)
    bwd_res = torch.cuda.max_memory_reserved(device)
    return {"weights_mib": weights / MiB,
            "fwd_alloc_mib": fwd_alloc / MiB, "bwd_alloc_mib": bwd_alloc / MiB,
            "fwd_reserved_mib": fwd_res / MiB, "bwd_reserved_mib": bwd_res / MiB,
            "peak_alloc_mib": max(fwd_alloc, bwd_alloc) / MiB,
            "peak_reserved_mib": max(fwd_res, bwd_res) / MiB}


def memory_in_subprocess(kind, args):
    cmd = [sys.executable, __file__, "--memory_only", kind, "--mode", args.mode,
           "--shape", args.shape, "--precision", args.precision,
           "--warmup", str(args.warmup)]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    line = next(x for x in out.splitlines() if x.startswith("MEMORY_JSON "))
    return json.loads(line[len("MEMORY_JSON "):])


def summarise(kind, rounds, ref):
    mean, half = ci95(rounds[kind])
    row = {"round_ms": rounds[kind], "ms": mean, "ms_ci95": half}
    if ref in rounds and kind != ref:
        rel = [100.0 * (a / b - 1.0) for a, b in zip(rounds[kind], rounds[ref])]
        row["vs_gqa_pct"], row["vs_gqa_pct_ci95"] = ci95(rel)
    return row


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("benchmark_training.py needs a CUDA GPU")
    device = torch.device("cuda")
    # every compiled model re-traces the same forward code objects
    torch._dynamo.config.cache_size_limit = 64
    if hasattr(torch._dynamo.config, "recompile_limit"):
        torch._dynamo.config.recompile_limit = 64

    if args.memory_only:
        print("MEMORY_JSON " + json.dumps(measure_memory(args.memory_only, args, device)),
              flush=True)
        return 0

    env = {"benchmark": "training", "device": torch.cuda.get_device_name(device),
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "precision": args.precision, "mode": args.mode, "shape": args.shape,
           "rounds": args.rounds, "warmup": args.warmup, "iters": args.iters,
           "visit_warmup": args.visit_warmup}
    print(env, flush=True)
    rows = {kind: {"type": kind} for kind in args.types}

    if not args.skip_memory:
        for kind in args.types:
            rows[kind]["memory"] = mem = memory_in_subprocess(kind, args)
            print(f"memory {kind}: fwd {mem['fwd_alloc_mib']:.0f} MiB  "
                  f"bwd {mem['bwd_alloc_mib']:.0f} MiB  "
                  f"(reserved {mem['peak_reserved_mib']:.0f} MiB)", flush=True)

    if not args.skip_latency:
        steps = {}
        for kind in args.types:
            steps[kind] = Step(kind, args, device)
            steps[kind].warmup(args.warmup)
            steps[kind].to("cpu")
            torch.cuda.empty_cache()
        parts = ("fwd", "bwd", "step")
        rounds = {p: {kind: [] for kind in args.types} for p in parts}
        for r in range(args.rounds):
            order = args.types[r % len(args.types):] + args.types[:r % len(args.types)]
            for kind in order:
                steps[kind].to(device)
                steps[kind].warmup(args.visit_warmup)
                for p, ms in zip(parts, time_step(steps[kind], args.iters)):
                    rounds[p][kind].append(ms)
                steps[kind].to("cpu")
            print(f"round {r + 1}/{args.rounds}: " + "  ".join(
                f"{k} {rounds['fwd'][k][-1]:.1f}+{rounds['bwd'][k][-1]:.1f}"
                for k in args.types), flush=True)
        for kind in args.types:
            for p in parts:
                rows[kind][p] = summarise(kind, rounds[p], "gqa")

    name = f"train_{args.precision}_{args.mode}_{args.shape}"
    out = Path(args.out or ROOT / "benchmarks" / "results" / f"{name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"env": env, "rows": list(rows.values())}, indent=2),
                   encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
