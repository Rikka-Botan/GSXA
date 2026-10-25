#!/usr/bin/env python
"""GSXA quickstart with the local package: fill-mask, embeddings, a classifier.

    python examples/quickstart.py --model <GSXA checkpoint dir>     # e.g. a converted one
    python examples/quickstart.py --random                         # untrained, API demo
    python examples/quickstart.py --random --attention_type xsa --device cuda --compile

A checkpoint directory comes from ``scripts/convert_nexterabert_v2.py`` (or any
``GSXAForMaskedLM.save_pretrained``). ``--random`` builds an untrained ``base`` model
instead, which shows the API but predicts noise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoTokenizer  # noqa: E402

from gsxa import (  # noqa: E402
    ATTENTION_TYPES,
    GSXAConfig,
    GSXAForMaskedLM,
    GSXAForSequenceClassification,
)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="GSXA checkpoint directory")
    p.add_argument("--random", action="store_true", help="untrained base model (API demo)")
    p.add_argument("--attention_type", default="gsxa", choices=ATTENTION_TYPES,
                   help="with --random: the attention to build")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compile", action="store_true", help="torch.compile the masked LM")
    args = p.parse_args(argv)
    if (args.model is None) == (not args.random):
        p.error("pass exactly one of --model / --random")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    # GSXA uses ModernBERT's tokenizer (same vocabulary and special tokens)
    tok_src = args.model if args.model and (Path(args.model) / "tokenizer.json").exists() \
        else "answerdotai/ModernBERT-base"
    tok = AutoTokenizer.from_pretrained(tok_src)
    if args.random:
        print(f"[quickstart] --random ({args.attention_type}): untrained weights, "
              f"the predictions below are noise")
        mlm = GSXAForMaskedLM(GSXAConfig(attention_type=args.attention_type))
    else:
        mlm = GSXAForMaskedLM.from_pretrained(args.model)
    mlm = mlm.to(device=device, dtype=dtype).eval()
    run = torch.compile(mlm) if args.compile else mlm
    print(f"attention_type={mlm.config.attention_type}, "
          f"{sum(p.numel() for p in mlm.parameters()):,} parameters")

    # 1. fill-mask
    text = f"The capital of France is {tok.mask_token}."
    enc = tok(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        logits = run(**enc).logits
    pos = (enc["input_ids"][0] == tok.mask_token_id).nonzero()[0, 0]
    top = logits[0, pos].float().softmax(-1).topk(5)
    print(text)
    for prob, idx in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {tok.decode([idx]).strip()!r:>12}  {prob:.3f}")

    # 2. sentence embeddings: masked mean of the encoder output
    sentences = ["A man is playing a guitar.",
                 "Someone is playing an instrument.",
                 "The stock market fell sharply today."]
    enc = tok(sentences, padding=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        h = mlm.model(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
    emb = torch.nn.functional.normalize(((h * m).sum(1) / m.sum(1)).float(), dim=-1)
    print("cosine similarity:\n", (emb @ emb.T).cpu().numpy().round(3))

    # 3. a sequence-classification head on the same encoder (fine-tune before use)
    clf = GSXAForSequenceClassification(GSXAConfig(**{**mlm.config.to_dict(), "num_labels": 2}))
    clf.model.load_state_dict(mlm.model.state_dict())
    clf = clf.to(device=device, dtype=dtype).eval()
    with torch.inference_mode():
        print("classifier logits (untrained head):", clf(**enc).logits.float().cpu().tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
