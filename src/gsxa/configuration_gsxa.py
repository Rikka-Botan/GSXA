"""GSXA configuration."""

from __future__ import annotations

try:
    from transformers import PretrainedConfig
except ImportError:  # pragma: no cover - newer alias
    from transformers import PreTrainedConfig as PretrainedConfig


# The attention of every layer (``attention_type``). GSXA is the proposed method; the
# other four are the models it is compared with in the README, and differ from it
# only in the attention's length scaling and output transform:
#
#   name            scalable factor   XSA projection        per-head gate
#   "gsxa"          yes               learned, tanh(g2)     g1      (proposed)
#   "gated"         yes               --                    g1      "Gated Attention"
#   "xsa"           yes               full, parameter-free  --      "XSA"
#   "scalable_gqa"  yes               --                    --      "Scalable GQA"
#   "gqa"           --                --                    --      "GQA" (RoPE + GQA)
ATTENTION_TYPES = ("gsxa", "gated", "xsa", "scalable_gqa", "gqa")
CLASSIFIER_POOLINGS = ("mean", "cls")

# Written into every saved config.json, so a checkpoint saved from any one head loads
# through all of them with ``trust_remote_code=True`` (``save_pretrained`` alone only
# records the class that was saved).
AUTO_MAP = {
    "AutoConfig": "configuration_gsxa.GSXAConfig",
    "AutoModel": "modeling_gsxa.GSXAModel",
    "AutoModelForMaskedLM": "modeling_gsxa.GSXAForMaskedLM",
    "AutoModelForSequenceClassification": "modeling_gsxa.GSXAForSequenceClassification",
}

# Named sizes (overrides on top of the defaults, which are "base").
#   base  12 layers x 768, 12 / 4 heads, MLP 2304   100.9 M (the size of the ablation)
#   tiny   8 layers x 384,  6 / 2 heads, MLP 1152    29.8 M (same ratios: MLP 3x,
#                                                             GQA 3:1, head dim 64)
# (masked-LM parameter counts with GSXA attention; the vocabulary is 50,368)
PRESETS = {
    "base": {},
    "tiny": dict(hidden_size=384, num_hidden_layers=8, num_attention_heads=6,
                 num_key_value_heads=2, intermediate_size=1152),
}


class GSXAConfig(PretrainedConfig):
    """Configuration of a GSXA encoder.

    The defaults are the model of the README's ablation: a BERT-style pre-norm
    encoder of 12 layers at hidden size 768 -- grouped-query attention (12 query /
    4 KV heads) with rotary positions and a squared-ReLU MLP -- whose attention is
    GSXA (Gated Scalable Exclusive Self Attention). Vocabulary and special-token ids
    are ModernBERT's.

    Attention (``attention_type``, see ``ATTENTION_TYPES``) selects GSXA or one of
    the compared variants; each registers only the parameters it uses.

    Scalable factor (``sf_*``, every type except ``"gqa"``): the queries are scaled
    by ``s * log(n) + sf_base + relu(b)``, ``n`` the number of non-padding keys, with
    a learnable ``s`` and ``b`` per layer initialised to ``sf_s`` and ``sf_b``.
    """

    model_type = "gsxa"

    def __init__(
        self,
        vocab_size: int = 50368,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,
        intermediate_size: int = 2304,
        attention_type: str = "gsxa",
        sf_s: float = 0.43,
        sf_b: float = 0.1,
        sf_base: float = 0.1,
        max_position_embeddings: int = 8192,
        rope_theta: float = 100000.0,
        norm_eps: float = 1e-8,
        final_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        classifier_pooling: str = "mean",
        classifier_dropout: float = 0.0,
        pad_token_id: int = 50283,
        bos_token_id: int = 50281,
        eos_token_id: int = 50282,
        cls_token_id: int = 50281,
        sep_token_id: int = 50282,
        mask_token_id: int = 50284,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        if hidden_size % num_attention_heads:
            raise ValueError(f"hidden_size {hidden_size} is not a multiple of "
                             f"num_attention_heads {num_attention_heads}")
        if num_attention_heads % num_key_value_heads:
            raise ValueError(f"num_attention_heads {num_attention_heads} is not a multiple "
                             f"of num_key_value_heads {num_key_value_heads}")
        if (hidden_size // num_attention_heads) % 2:
            raise ValueError("the head dimension must be even for rotary embeddings")
        if attention_type not in ATTENTION_TYPES:
            raise ValueError(f"attention_type must be one of {ATTENTION_TYPES}, "
                             f"got {attention_type!r}")
        if sf_s <= 0 or sf_base <= 0:
            raise ValueError("sf_s and sf_base must be positive")
        if classifier_pooling not in CLASSIFIER_POOLINGS:
            raise ValueError(f"classifier_pooling must be one of {CLASSIFIER_POOLINGS}, "
                             f"got {classifier_pooling!r}")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.attention_type = attention_type
        self.sf_s = float(sf_s)
        self.sf_b = float(sf_b)
        self.sf_base = float(sf_base)
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = float(rope_theta)
        self.norm_eps = norm_eps
        self.final_norm_eps = final_norm_eps
        self.initializer_range = initializer_range
        self.classifier_pooling = classifier_pooling
        self.classifier_dropout = classifier_dropout
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.mask_token_id = mask_token_id
        kwargs["auto_map"] = {**AUTO_MAP, **(kwargs.get("auto_map") or {})}
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "GSXAConfig":
        """A named size (``PRESETS``: "base", "tiny") with optional overrides."""
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}; choose one of {tuple(PRESETS)}")
        return cls(**{**PRESETS[name], **overrides})


__all__ = ["GSXAConfig", "ATTENTION_TYPES", "CLASSIFIER_POOLINGS", "PRESETS", "AUTO_MAP"]
