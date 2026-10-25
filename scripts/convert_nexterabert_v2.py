#!/usr/bin/env python
"""Convert a NexteraBERTv2 backbone of the attention ablation into a GSXA checkpoint.

The models in the README were trained with the NexteraBERTv2 pretraining pipeline
(not part of this repository), configured as a plain stack of attention layers. Its
export is a directory with ``config.json``, ``model.safetensors`` (or
``pytorch_model.bin``) -- the encoder -- and ``mlm_head.safetensors`` -- the masked-LM
head. This script maps that directory onto ``GSXAForMaskedLM`` and saves it with
``save_pretrained`` (plus the modeling code, so it loads with
``trust_remote_code=True``):

    python scripts/convert_nexterabert_v2.py --src <backbone dir> --out <GSXA dir>

Only that configuration converts: every layer an ``attn`` layer, no q/k
normalisation, no MUDD connections, the dense squared-ReLU MLP, and an
(``ssmax_mode``, ``attn_gate``) pair that is one of the five attention types.
Anything else is refused rather than converted into a different model.

The mapping is exact (``tests/`` cannot check it without the training code; it was
verified against NexteraBERTv2 to fp64 round-off on all five attention types).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gsxa import GSXAConfig, GSXAForMaskedLM  # noqa: E402

# (ssmax_mode, attn_gate) of NexteraBERTv2 -> GSXA attention_type
ATTENTION = {
    ("sssmax", "exclusive"): "gsxa",
    ("sssmax", "head"): "gated",
    ("sssmax", "xsa"): "xsa",
    ("sssmax", "none"): "scalable_gqa",
    ("none", "none"): "gqa",
}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def load_state(path: Path) -> dict:
    if (path / "model.safetensors").exists():
        from safetensors.torch import load_file
        return load_file(str(path / "model.safetensors"))
    if (path / "pytorch_model.bin").exists():
        return torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"{path}: no model.safetensors or pytorch_model.bin")


def gsxa_config(src: dict) -> GSXAConfig:
    """The GSXA config of a NexteraBERTv2 ``config.json``, or a ValueError saying why
    it is not one of the ablation's models."""
    n_layer = int(src["n_layer"])
    blocks = list(src.get("block_types") or [])
    problems = []
    if blocks != ["attn"] * n_layer:
        problems.append(f"block_types must be {n_layer} x 'attn', got {blocks}")
    if src.get("attn_qk_norm", "dyt") != "none":
        problems.append(f"attn_qk_norm must be 'none', got {src.get('attn_qk_norm', 'dyt')!r}")
    if int(src.get("mudd_dim", 0)) > 0 and src.get("mudd_layers"):
        problems.append("MUDD connections are not part of GSXA (mudd_dim must be 0)")
    if src.get("mlp_class", "mlp") != "mlp":
        problems.append(f"mlp_class must be 'mlp', got {src.get('mlp_class')!r}")
    if int(src.get("rope_ntk_train_len", 0) or 0) > 0:
        problems.append("rope_ntk_train_len > 0 (dynamic NTK RoPE) is not implemented here")
    key = (src.get("ssmax_mode", "sssmax"), src.get("attn_gate", "exclusive"))
    if key not in ATTENTION:
        problems.append(f"(ssmax_mode, attn_gate) = {key} is not one of {sorted(ATTENTION)}")
    if problems:
        raise ValueError("not a GSXA-ablation model:\n  " + "\n  ".join(problems))
    special = {k: src[k] for k in ("pad_token_id", "cls_token_id", "sep_token_id",
                                   "mask_token_id") if src.get(k) is not None}
    if "cls_token_id" in special:
        special["bos_token_id"] = special["cls_token_id"]
    if "sep_token_id" in special:
        special["eos_token_id"] = special["sep_token_id"]
    return GSXAConfig(
        vocab_size=int(src["vocab_size"]),
        hidden_size=int(src["n_embd"]),
        num_hidden_layers=n_layer,
        num_attention_heads=int(src["n_head"]),
        num_key_value_heads=int(src["n_kv_head"]),
        intermediate_size=int(src["n_inter"]),
        attention_type=ATTENTION[key],
        # an export without the key predates sigmoid gates (2026-10-26): its g1 is linear
        gate_activation=src.get("gate_activation", "linear"),
        sf_s=float(src.get("ssmax_s", 0.43)),
        sf_b=float(src.get("ssmax_b", 0.1)),
        sf_base=float(src.get("ssmax_base", 0.1)),
        rope_theta=float(src.get("rope_base", 100000)),
        tie_word_embeddings=bool(src.get("tie_word_embeddings", True)),
        **special,
    )


def convert_state(enc: dict, head: dict, config: GSXAConfig) -> dict:
    """NexteraBERTv2 encoder + MLM-head tensors -> a ``GSXAForMaskedLM`` state_dict."""
    out = {"model.embeddings.weight": enc["embedding.weight"],
           "model.final_norm.weight": enc["out_norm.weight"],
           "model.final_norm.bias": enc["out_norm.bias"]}
    for i in range(config.num_hidden_layers):
        s, d = f"blocks.{i}.", f"model.layers.{i}."
        out[d + "attn_norm.weight"] = enc[s + "attn_norm.weight"]
        out[d + "mlp_norm.weight"] = enc[s + "mlp_norm.weight"]
        out[d + "attn.Wqkv.weight"] = torch.cat(
            [enc[s + "attn.c_q.weight"], enc[s + "attn.c_k.weight"],
             enc[s + "attn.c_v.weight"]], dim=0)
        out[d + "attn.Wo.weight"] = enc[s + "attn.c_proj.weight"]
        if s + "attn.c_gate.weight" in enc:
            out[d + "attn.Wg.weight"] = enc[s + "attn.c_gate.weight"]
        if s + "attn.scalable_factor.s" in enc:
            out[d + "attn.scalable_factor.s"] = enc[s + "attn.scalable_factor.s"]
            out[d + "attn.scalable_factor.b"] = enc[s + "attn.scalable_factor.b"]
        out[d + "mlp.Wi.weight"] = enc[s + "mlp.c_fc.weight"]
        out[d + "mlp.Wo.weight"] = enc[s + "mlp.c_proj.weight"]
    out["head.dense.weight"] = head["head.dense.weight"]
    out["head.norm.weight"] = head["head.norm.weight"]
    out["decoder.bias"] = head["lm_head.bias"]
    out["decoder.weight"] = (out["model.embeddings.weight"] if config.tie_word_embeddings
                             else head["lm_head.weight"])
    return out


def convert(src: str | Path, out: str | Path, dtype: str = "float32") -> GSXAForMaskedLM:
    src, out = Path(src), Path(out)
    config = gsxa_config(json.loads((src / "config.json").read_text(encoding="utf-8")))
    head_file = src / "mlm_head.safetensors"
    if not head_file.exists():
        raise FileNotFoundError(f"{head_file} is missing: without the MLM head the "
                                f"masked-LM predictions would be noise")
    from safetensors.torch import load_file
    state = convert_state(load_state(src), load_file(str(head_file)), config)

    model = GSXAForMaskedLM(config).eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if k != "decoder.weight"]       # tied
    if missing or unexpected:
        raise RuntimeError(f"state mismatch: missing {missing}, unexpected {unexpected}")
    model = model.to(getattr(torch, dtype))
    model.save_pretrained(out)
    for name in TOKENIZER_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
    return model


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="exported NexteraBERTv2 backbone directory")
    p.add_argument("--out", required=True, help="output GSXA directory")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--tokenizer", default="answerdotai/ModernBERT-base",
                   help="saved next to the weights when --src carries no tokenizer "
                        "('' to skip)")
    args = p.parse_args(argv)
    model = convert(args.src, args.out, args.dtype)
    out = Path(args.out)
    if args.tokenizer and not (out / "tokenizer.json").exists():
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(args.tokenizer).save_pretrained(out)
    n = sum(p.numel() for p in model.parameters())
    print(f"wrote {out} ({model.config.attention_type}, {n:,} parameters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
