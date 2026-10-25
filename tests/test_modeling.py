"""CPU tests for GSXA.

Run with:  python -m pytest -q tests
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from transformers import AutoModel, AutoModelForMaskedLM  # noqa: E402

from gsxa import (  # noqa: E402
    ATTENTION_TYPES,
    GSXAConfig,
    GSXAForMaskedLM,
    GSXAForSequenceClassification,
    GSXAModel,
)
from gsxa.modeling_gsxa import GSXAAttention, apply_rotary  # noqa: E402


def tiny(**kw):
    base = dict(vocab_size=128, hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
                num_key_value_heads=2, intermediate_size=128, pad_token_id=0, bos_token_id=1,
                eos_token_id=2, cls_token_id=1, sep_token_id=2, mask_token_id=3)
    base.update(kw)
    return GSXAConfig(**base)


def batch(B=2, T=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, 128, (B, T), generator=g)


# --- configuration -----------------------------------------------------------------

def test_default_is_the_ablation_model():
    cfg = GSXAConfig()
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads,
            cfg.num_key_value_heads, cfg.intermediate_size) == (12, 768, 12, 4, 2304)
    assert cfg.attention_type == "gsxa" and cfg.vocab_size == 50368
    assert (cfg.sf_s, cfg.sf_b, cfg.sf_base, cfg.rope_theta) == (0.43, 0.1, 0.1, 100000.0)
    sizes = {}
    for kind in ATTENTION_TYPES:
        with torch.device("meta"):
            model = GSXAForMaskedLM(GSXAConfig(attention_type=kind))
        sizes[kind] = sum(p.numel() for p in model.parameters())
    assert model.decoder.weight is model.model.embeddings.weight
    assert sizes["gsxa"] == 100_906_456
    # W_G: 2H x d (GSXA) or H x d (Gated Attention) per layer; lambda: 2 scalars per layer
    per_head_gate = 12 * 768 * 12
    assert sizes["gsxa"] - sizes["gated"] == per_head_gate
    assert sizes["gated"] - sizes["xsa"] == per_head_gate
    assert sizes["xsa"] == sizes["scalable_gqa"] == sizes["gqa"] + 12 * 2


def test_invalid_configs_are_rejected():
    for bad in (dict(num_key_value_heads=3), dict(attention_type="sigmoid"),
                dict(sf_s=0.0), dict(sf_base=-1.0), dict(classifier_pooling="max")):
        with pytest.raises(ValueError):
            tiny(**bad)


def test_each_type_registers_only_what_it_uses():
    widths = {}
    for kind in ATTENTION_TYPES:
        attn = GSXAAttention(tiny(attention_type=kind))
        widths[kind] = None if attn.Wg is None else attn.Wg.out_features
        assert (attn.scalable_factor is None) == (kind == "gqa")
    assert widths == {"gsxa": 8, "gated": 4, "xsa": None, "scalable_gqa": None, "gqa": None}


# --- the attention -------------------------------------------------------------------

def per_head_reference(attn, o, v):
    """The README's equations, one query head at a time (no reshapes shared with the
    implementation): kappa(h) = h // G, vhat the unit own value."""
    B, T, C = o.shape
    H, D, G = attn.n_head, attn.head_dim, attn.n_group
    g = attn.Wg(o) if attn.Wg is not None else None
    out = torch.empty_like(o)
    for h in range(H):
        oh = o[..., h * D:(h + 1) * D]
        vhat = F.normalize(v[:, :, h // G], dim=-1)
        dot = (oh * vhat).sum(-1, keepdim=True)
        if attn.kind == "gsxa":
            g1, g2 = g[..., h:h + 1], g[..., H + h:H + h + 1]
            z = g1 * (oh - torch.tanh(g2) * dot * vhat)
        elif attn.kind == "gated":
            z = g[..., h:h + 1] * oh
        elif attn.kind == "xsa":
            z = oh - dot * vhat
        else:
            z = oh
        out[..., h * D:(h + 1) * D] = z
    return out


@pytest.mark.parametrize("kind", ATTENTION_TYPES)
def test_output_transform_matches_the_equations(kind):
    torch.manual_seed(0)
    attn = GSXAAttention(tiny(attention_type=kind))
    if attn.Wg is not None:
        torch.nn.init.normal_(attn.Wg.weight, std=0.5)
    o, v = torch.randn(2, 5, 64), torch.randn(2, 5, 2, 16)
    assert torch.allclose(attn.output_transform(o, v), per_head_reference(attn, o, v),
                          atol=1e-6)


def test_xsa_output_is_orthogonal_to_the_own_value():
    torch.manual_seed(0)
    attn = GSXAAttention(tiny(attention_type="xsa"))
    o, v = torch.randn(2, 5, 64), torch.randn(2, 5, 2, 16)
    z = attn.output_transform(o, v).view(2, 5, 2, 2, 16)
    vhat = F.normalize(v, dim=-1).unsqueeze(3)
    assert (z * vhat).sum(-1).abs().max() < 1e-5


def test_gsxa_reduces_to_gated_attention_when_g2_is_zero():
    torch.manual_seed(0)
    gsxa = GSXAAttention(tiny(attention_type="gsxa"))
    gated = GSXAAttention(tiny(attention_type="gated"))
    with torch.no_grad():
        gsxa.Wg.weight[4:].zero_()                      # g2 = 0 -> tanh(g2) = 0
        gated.Wg.weight.copy_(gsxa.Wg.weight[:4])       # the same g1
    o, v = torch.randn(2, 5, 64), torch.randn(2, 5, 2, 16)
    assert torch.allclose(gsxa.output_transform(o, v), gated.output_transform(o, v), atol=1e-6)


def test_scalable_factor_formula_and_its_floor():
    attn = GSXAAttention(tiny(attention_type="scalable_gqa"))
    sf = attn.scalable_factor
    n = torch.tensor([1, 8, 1024])
    want = 0.43 * torch.log(n.float()) + 0.1 + 0.1
    assert torch.allclose(sf(n).flatten(), want)
    with torch.no_grad():
        sf.b.fill_(-3.0)                                # relu(b) = 0: the floor beta remains
    assert torch.allclose(sf(torch.tensor([1])).flatten(), torch.tensor([0.1]))


def test_rotary_convention_and_relative_positions():
    D = 8
    x = torch.randn(1, 6, 1, D)
    inv = 1.0 / (100000.0 ** (torch.arange(0, D, 2).float() / D))
    f = torch.outer(torch.arange(6).float(), inv)
    y = apply_rotary(x, f.cos(), f.sin())
    x1, x2 = x[..., :D // 2], x[..., D // 2:]
    c, s = f.cos()[None, :, None], f.sin()[None, :, None]
    assert torch.allclose(y, torch.cat((x1 * c + x2 * s, x2 * c - x1 * s), dim=-1), atol=1e-6)
    # the same vector at every position: the score depends on the offset only
    q = apply_rotary(x[:, :1].expand(1, 6, 1, D), f.cos(), f.sin())[0, :, 0]
    k = apply_rotary(x[:, 1:2].expand(1, 6, 1, D), f.cos(), f.sin())[0, :, 0]
    assert torch.allclose(q[0] @ k[2], q[3] @ k[5], atol=1e-5)


# --- the model -----------------------------------------------------------------------

@pytest.mark.parametrize("kind", ATTENTION_TYPES)
def test_every_type_runs_and_reaches_every_parameter(kind):
    torch.manual_seed(0)
    model = GSXAForMaskedLM(tiny(attention_type=kind))
    ids = batch()
    mask = torch.ones_like(ids)
    mask[1, 7:] = 0
    labels = torch.full_like(ids, -100)
    labels[:, 3] = ids[:, 3]
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    assert out.logits.shape == (2, 12, 128) and torch.isfinite(out.loss)
    out.loss.backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None]
    assert dead == [], dead


@pytest.mark.parametrize("kind", ["gsxa", "scalable_gqa"])
def test_padding_never_reaches_the_real_tokens(kind):
    """A padded row equals the same sequence alone -- including the scalable factor,
    whose n counts the non-padding keys only."""
    torch.manual_seed(0)
    model = GSXAForMaskedLM(tiny(attention_type=kind)).double().eval()
    ids = batch()
    mask = torch.ones_like(ids)
    mask[1, 7:] = 0
    with torch.no_grad():
        padded = model(input_ids=ids, attention_mask=mask).logits
        alone = model(input_ids=ids[1:, :7]).logits
    assert torch.allclose(padded[1, :7], alone[0], atol=1e-10)


def test_bf16_inference():
    model = GSXAForMaskedLM(tiny()).to(torch.bfloat16).eval()
    with torch.inference_mode():
        logits = model(input_ids=batch()).logits
    assert logits.dtype == torch.bfloat16 and torch.isfinite(logits.float()).all()


def test_sequence_classification():
    model = GSXAForSequenceClassification(tiny(num_labels=3))
    ids = batch()
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=torch.tensor([0, 2]))
    assert out.logits.shape == (2, 3) and torch.isfinite(out.loss)


def test_save_and_load_round_trip(tmp_path):
    torch.manual_seed(0)
    model = GSXAForMaskedLM(tiny()).eval()
    with torch.no_grad():
        for layer in model.model.layers:
            layer.attn.scalable_factor.s.fill_(0.7)
    model.save_pretrained(tmp_path)
    assert (tmp_path / "modeling_gsxa.py").exists()      # trust_remote_code loads it
    ids = batch()
    with torch.no_grad():
        ref = model(input_ids=ids).logits
        for cls in (GSXAForMaskedLM, AutoModelForMaskedLM):
            again = cls.from_pretrained(tmp_path).eval()
            assert torch.equal(again(input_ids=ids).logits, ref)
        enc = AutoModel.from_pretrained(tmp_path).eval()
        assert isinstance(enc, GSXAModel)
        assert torch.equal(enc(input_ids=ids).last_hidden_state,
                           model.model(input_ids=ids).last_hidden_state)


# --- the converter --------------------------------------------------------------------

def to_nexterabert_v2(model):
    """A GSXA masked LM written back in the NexteraBERTv2 export layout (the inverse of
    the converter's mapping), with the config.json keys the converter reads."""
    cfg, sd = model.config, model.state_dict()
    H, Hkv, D = cfg.num_attention_heads, cfg.num_key_value_heads, \
        cfg.hidden_size // cfg.num_attention_heads
    enc = {"embedding.weight": sd["model.embeddings.weight"],
           "out_norm.weight": sd["model.final_norm.weight"],
           "out_norm.bias": sd["model.final_norm.bias"]}
    for i in range(cfg.num_hidden_layers):
        s, d = f"blocks.{i}.", f"model.layers.{i}."
        q, k, v = sd[d + "attn.Wqkv.weight"].split([H * D, Hkv * D, Hkv * D])
        enc.update({s + "attn_norm.weight": sd[d + "attn_norm.weight"],
                    s + "mlp_norm.weight": sd[d + "mlp_norm.weight"],
                    s + "attn.c_q.weight": q, s + "attn.c_k.weight": k,
                    s + "attn.c_v.weight": v, s + "attn.c_proj.weight": sd[d + "attn.Wo.weight"],
                    s + "mlp.c_fc.weight": sd[d + "mlp.Wi.weight"],
                    s + "mlp.c_proj.weight": sd[d + "mlp.Wo.weight"]})
        if d + "attn.Wg.weight" in sd:
            enc[s + "attn.c_gate.weight"] = sd[d + "attn.Wg.weight"]
        if d + "attn.scalable_factor.s" in sd:
            enc[s + "attn.scalable_factor.s"] = sd[d + "attn.scalable_factor.s"]
            enc[s + "attn.scalable_factor.b"] = sd[d + "attn.scalable_factor.b"]
    head = {"head.dense.weight": sd["head.dense.weight"],
            "head.norm.weight": sd["head.norm.weight"], "lm_head.bias": sd["decoder.bias"]}
    ssmax, gate = {"gsxa": ("sssmax", "exclusive"), "gated": ("sssmax", "head"),
                   "xsa": ("sssmax", "xsa"), "scalable_gqa": ("sssmax", "none"),
                   "gqa": ("none", "none")}[cfg.attention_type]
    config = dict(n_layer=cfg.num_hidden_layers, n_head=H, n_kv_head=Hkv,
                  n_embd=cfg.hidden_size, n_inter=cfg.intermediate_size,
                  vocab_size=cfg.vocab_size, block_types=["attn"] * cfg.num_hidden_layers,
                  ssmax_mode=ssmax, attn_gate=gate, attn_qk_norm="none", mudd_dim=0,
                  mudd_layers=[], mlp_class="mlp", rope_base=cfg.rope_theta,
                  rope_ntk_train_len=0, tie_word_embeddings=True, pad_token_id=0,
                  cls_token_id=1, sep_token_id=2, mask_token_id=3)
    return enc, head, config


@pytest.mark.parametrize("kind", ATTENTION_TYPES)
def test_converter_round_trip(tmp_path, kind):
    import json

    from safetensors.torch import save_file

    from convert_nexterabert_v2 import convert

    torch.manual_seed(0)
    model = GSXAForMaskedLM(tiny(attention_type=kind)).eval()
    enc, head, config = to_nexterabert_v2(model)
    src = tmp_path / "v2"
    src.mkdir()
    save_file({k: v.contiguous() for k, v in enc.items()}, str(src / "model.safetensors"))
    save_file({k: v.contiguous() for k, v in head.items()}, str(src / "mlm_head.safetensors"))
    (src / "config.json").write_text(json.dumps(config), encoding="utf-8")
    converted = convert(src, tmp_path / "gsxa").eval()
    assert converted.config.attention_type == kind
    ids = batch()
    with torch.no_grad():
        assert torch.equal(converted(input_ids=ids).logits, model(input_ids=ids).logits)


def test_converter_refuses_other_architectures():
    from convert_nexterabert_v2 import gsxa_config

    good = dict(n_layer=2, n_head=4, n_kv_head=2, n_embd=64, n_inter=128, vocab_size=128,
                block_types=["attn", "attn"], ssmax_mode="sssmax", attn_gate="exclusive",
                attn_qk_norm="none", mudd_dim=0, mudd_layers=[])
    assert gsxa_config(good).attention_type == "gsxa"
    for bad in (dict(block_types=["lily", "attn"]), dict(attn_qk_norm="dyt"),
                dict(attn_gate="sigmoid"), dict(ssmax_mode="ssmax"),
                dict(mudd_dim=64, mudd_layers=[1]), dict(rope_ntk_train_len=1024)):
        with pytest.raises(ValueError):
            gsxa_config({**good, **bad})
