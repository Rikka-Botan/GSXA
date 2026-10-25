"""GSXA: Gated Scalable Exclusive Self Attention, in a BERT-style encoder.

Encoder layer (pre-norm)::

    x = x + Attention(RMSNorm(x))        # GQA + RoPE, then the output transform below
    x = x + MLP(RMSNorm(x))              # W_o relu(W_i x)^2, no bias

GSXA attention, per query head ``h`` (KV head ``kappa(h) = h // (H / H_kv)``)::

    lambda   = s log(n) + beta + relu(b)                    scalable factor, n = #keys
    o^h      = softmax(lambda q^h K^T / sqrt(D)) V          rotary q, k; GQA K, V
    g1, g2   = split(W_G [o^1; ...; o^H])                   one scalar each, per head
    vhat     = v_t^{kappa(h)} / ||v_t^{kappa(h)}||          the token's OWN value
    z^h      = g1^h (o^h - tanh(g2^h) <o^h, vhat> vhat)
    y        = W_O [z^1; ...; z^H]

``attention_type`` selects GSXA or one of the variants it is compared with: "gated"
(``z = g1 o``), "xsa" (``z = o - <o, vhat> vhat``), "scalable_gqa" (``z = o``) and
"gqa" (``z = o`` and ``lambda = 1``). See the README for the full definitions.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
)

try:  # transformers >= 5: init helpers that skip tensors a checkpoint filled
    from transformers import initialization as init
except ImportError:  # pragma: no cover
    from torch.nn import init

from .configuration_gsxa import GSXAConfig


# ---------------------------------------------------------------------------
# Rotary position embedding
# ---------------------------------------------------------------------------

class GSXARotaryEmbedding(nn.Module):
    """cos / sin tables of the rotary embedding, computed on demand.

    Deliberately holds no buffer: transformers >= 5 builds a model on the meta
    device and leaves computed non-persistent buffers zero after loading. The
    tables are memoised per (length, device) in eager mode and recomputed inside a
    compiled graph, where inductor folds them. They stay fp32; ``apply_rotary``
    casts them to the activations' dtype.
    """

    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self._cache: dict = {}

    def forward(self, seq_len: int, device):
        compiling = torch.compiler.is_compiling()
        key = (seq_len, str(device))
        if not compiling and key in self._cache:
            return self._cache[key]
        # Built outside inference mode so a cached table can later be saved for
        # backward (fine-tuning after an inference_mode call).
        with torch.inference_mode(False), torch.no_grad():
            inv_freq = 1.0 / (self.theta ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)
                / self.head_dim))
            t = torch.arange(seq_len, dtype=torch.float32, device=device)
            freqs = torch.outer(t, inv_freq)                     # (T, D / 2)
            cos, sin = freqs.cos(), freqs.sin()
        if not compiling:
            if len(self._cache) > 32:
                self._cache.clear()
            self._cache[key] = (cos, sin)
        return cos, sin


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the pairs ``(x_i, x_{i + D/2})`` of ``x`` (B, T, H, D) by ``-t omega_i``.

    This is the convention the checkpoints were trained with:
    ``(x1, x2) -> (x1 cos + x2 sin, -x1 sin + x2 cos)``. It is the usual
    rotate-half RoPE with the angle negated -- still a relative-position encoding
    (the score depends on ``t_k - t_q`` only), but not interchangeable with the
    other sign for trained weights.
    """
    cos = cos.to(x.dtype)[None, :, None, :]
    sin = sin.to(x.dtype)[None, :, None, :]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos + x2 * sin, x2 * cos - x1 * sin), dim=-1)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class GSXARMSNorm(nn.RMSNorm):
    """``nn.RMSNorm`` whose gain follows the input's dtype.

    Under autocast the weights stay fp32 while the activations are bf16;
    ``rms_norm`` is not on autocast's fp32 list, so a plain ``nn.RMSNorm`` then warns
    and falls back to an unfused implementation. Casting the gain keeps the fused
    kernel (which accumulates in fp32 either way).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = None if self.weight is None else self.weight.to(x.dtype)
        return F.rms_norm(x, self.normalized_shape, weight, self.eps)


class GSXAScalableFactor(nn.Module):
    """The scalable factor ``lambda(n) = s log(n) + beta + relu(b)`` (one per layer).

    ``n`` is the number of keys a query attends to -- the non-padding tokens of its
    sequence. ``s`` and ``b`` are learnable scalars (initialised to ``config.sf_s`` /
    ``config.sf_b``; ``b > 0`` so ``relu(b)`` has a gradient at the start), ``beta``
    a fixed positive floor (``config.sf_base``). Because ``s log n >= 0`` and
    ``relu(b) >= 0``, ``lambda >= beta > 0`` at every length: the query scale can
    neither vanish on a single key nor change sign. This is the scaling of
    Scalable-Softmax (Nakanishi, 2025) with the offset added for stability.
    """

    def __init__(self, config: GSXAConfig):
        super().__init__()
        self.s = nn.Parameter(torch.tensor(config.sf_s, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(config.sf_b, dtype=torch.float32))
        self.base = config.sf_base

    def forward(self, lengths: torch.Tensor) -> torch.Tensor:
        """``lengths`` (B,) -> the (B, 1, 1, 1) fp32 scale of each sequence's queries."""
        n = lengths.to(torch.float32).clamp_min(1.0)
        scale = (self.s.to(torch.float32) * torch.log(n) + self.base
                 + F.relu(self.b.to(torch.float32)))
        return scale.reshape(-1, 1, 1, 1)

    def extra_repr(self) -> str:
        return f"base={self.base}"


def exclusive_projection(o: torch.Tensor, v: torch.Tensor, strength=None) -> torch.Tensor:
    """``o - strength <o, vhat> vhat`` per head, ``vhat`` the unit own value.

    ``o`` (B, T, H_kv, G, D): the attention output, query head ``h`` at
    ``[h // G, h % G]``; ``v`` (B, T, H_kv, D): each token's own value; ``strength``
    (B, T, H_kv, G) or ``None`` for 1 (the full projection, as in XSA).
    """
    vhat = F.normalize(v, dim=-1).to(o.dtype).unsqueeze(3)          # (B, T, H_kv, 1, D)
    proj = (o * vhat).sum(-1, keepdim=True) * vhat
    if strength is not None:
        proj = strength.unsqueeze(-1) * proj
    return o - proj


class GSXAAttention(nn.Module):
    """Bidirectional GQA with rotary positions and the output transform of
    ``config.attention_type`` (GSXA by default; see the module docstring)."""

    def __init__(self, config: GSXAConfig):
        super().__init__()
        self.kind = config.attention_type
        self.n_head = config.num_attention_heads
        self.n_kv_head = config.num_key_value_heads
        self.n_group = self.n_head // self.n_kv_head
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.Wqkv = nn.Linear(config.hidden_size,
                              (self.n_head + 2 * self.n_kv_head) * self.head_dim, bias=False)
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # Gate logits from the attention output: g1 and g2 per head (GSXA) or g1 alone
        # (Gated Attention). XSA's projection and the plain variants have none.
        gate_out = {"gsxa": 2 * self.n_head, "gated": self.n_head}.get(self.kind)
        self.Wg = (nn.Linear(config.hidden_size, gate_out, bias=False)
                   if gate_out is not None else None)
        self.scalable_factor = GSXAScalableFactor(config) if self.kind != "gqa" else None

    def extra_repr(self) -> str:
        return f"attention_type={self.kind!r}"

    def output_transform(self, o: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """``o`` (B, T, H * D) attention output, ``v`` (B, T, H_kv, D) own values ->
        the transformed (B, T, H * D) output, before ``W_O``."""
        if self.kind in ("gqa", "scalable_gqa"):
            return o
        B, T, C = o.shape
        oh = o.view(B, T, self.n_kv_head, self.n_group, self.head_dim)
        if self.kind == "xsa":
            return exclusive_projection(oh, v).reshape(B, T, C)
        g = self.Wg(o)
        if self.kind == "gated":
            g1 = g.view(B, T, self.n_kv_head, self.n_group)
            return (g1.unsqueeze(-1) * oh).reshape(B, T, C)
        g1, g2 = (x.view(B, T, self.n_kv_head, self.n_group) for x in g.chunk(2, dim=-1))
        z = exclusive_projection(oh, v, torch.tanh(g2))
        return (g1.unsqueeze(-1) * z).reshape(B, T, C)

    def forward(self, x, cos, sin, attn_mask=None, lengths=None):
        B, T, C = x.shape
        qkv = self.Wqkv(x).view(B, T, self.n_head + 2 * self.n_kv_head, self.head_dim)
        q, k, v = qkv.split([self.n_head, self.n_kv_head, self.n_kv_head], dim=2)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        v_own = v
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.scalable_factor is not None:
            if lengths is None:
                lengths = torch.full((B,), T, device=x.device, dtype=torch.long)
            # cast the (B,1,1,1) fp32 scale down first: q is never promoted to fp32
            q = q * self.scalable_factor(lengths).to(q.dtype)
        if self.n_group > 1:
            k = k.repeat_interleave(self.n_group, dim=1)
            v = v.repeat_interleave(self.n_group, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # scale 1/sqrt(D)
        o = o.transpose(1, 2).reshape(B, T, C)
        return self.Wo(self.output_transform(o, v_own))


class GSXAMLP(nn.Module):
    """``W_o relu(W_i x)^2`` (squared ReLU, no gate, no bias)."""

    def __init__(self, config: GSXAConfig):
        super().__init__()
        self.Wi = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.Wo(F.relu(self.Wi(x)).square())


class GSXALayer(nn.Module):
    def __init__(self, config: GSXAConfig):
        super().__init__()
        self.attn_norm = GSXARMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = GSXAAttention(config)
        self.mlp_norm = GSXARMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = GSXAMLP(config)

    def forward(self, x, cos, sin, attn_mask=None, lengths=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, attn_mask, lengths)
        return x + self.mlp(self.mlp_norm(x))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class GSXAPreTrainedModel(PreTrainedModel):
    config_class = GSXAConfig
    base_model_prefix = "model"
    _no_split_modules = ["GSXALayer"]
    _supports_sdpa = True

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        out_std = std / math.sqrt(2.0 * self.config.num_hidden_layers)
        if isinstance(module, GSXAAttention):
            init.normal_(module.Wqkv.weight, mean=0.0, std=std)
            init.normal_(module.Wo.weight, mean=0.0, std=out_std)
            if module.Wg is not None:
                init.normal_(module.Wg.weight, mean=0.0, std=std)
        elif isinstance(module, GSXAScalableFactor):
            init.constant_(module.s, self.config.sf_s)
            init.constant_(module.b, self.config.sf_b)
        elif isinstance(module, GSXAMLP):
            init.normal_(module.Wi.weight, mean=0.0, std=std)
            init.normal_(module.Wo.weight, mean=0.0, std=out_std)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
            if module.weight is not None:
                init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            # prediction head, decoder bias, classifier (the tied decoder weight is
            # the embedding and is filled through it)
            if module is not getattr(self, "decoder", None):
                init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)


class GSXAModel(GSXAPreTrainedModel):
    """The encoder: token embedding -> layers -> final LayerNorm."""

    def __init__(self, config: GSXAConfig):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([GSXALayer(config) for _ in range(config.num_hidden_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.final_norm_eps)
        self.rotary = GSXARotaryEmbedding(
            config.hidden_size // config.num_attention_heads, config.rope_theta)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    @staticmethod
    def _attention_mask(attention_mask):
        """(B, T) 0/1 key mask -> SDPA boolean mask, or None when nothing is padded
        (which lets SDPA take its flash kernel)."""
        if attention_mask is None:
            return None
        if not torch.compiler.is_compiling() and bool(attention_mask.all()):
            return None
        return attention_mask.bool()[:, None, None, :]

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        """``output_hidden_states`` returns the embedding and the residual stream after
        every layer."""
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        output_hidden_states = (output_hidden_states if output_hidden_states is not None
                                else self.config.output_hidden_states)
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids / inputs_embeds")
        x = self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        cos, sin = self.rotary(x.shape[1], x.device)
        mask = self._attention_mask(attention_mask)
        # n of the scalable factor: the non-padding tokens of each sequence
        lengths = None if attention_mask is None else attention_mask.sum(dim=-1)
        states = [x] if output_hidden_states else None
        for layer in self.layers:
            x = layer(x, cos, sin, mask, lengths)
            if states is not None:
                states.append(x)
        x = self.final_norm(x)
        hidden_states = tuple(states) if states is not None else None
        if not return_dict:
            return tuple(v for v in (x, hidden_states) if v is not None)
        return BaseModelOutput(last_hidden_state=x, hidden_states=hidden_states)


class GSXAPredictionHead(nn.Module):
    """dense (no bias) -> GELU -> LayerNorm (no bias), as ModernBERT's."""

    def __init__(self, config: GSXAConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.final_norm_eps, bias=False)

    def forward(self, x):
        return self.norm(F.gelu(self.dense(x)))


class GSXAForMaskedLM(GSXAPreTrainedModel):
    _tied_weights_keys = {"decoder.weight": "model.embeddings.weight"}

    def __init__(self, config: GSXAConfig):
        super().__init__(config)
        self.model = GSXAModel(config)
        self.head = GSXAPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        self.post_init()

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, new_embeddings):
        self.decoder = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         inputs_embeds=inputs_embeds,
                         output_hidden_states=output_hidden_states, return_dict=True)
        logits = self.decoder(self.head(out.last_hidden_state))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1),
                                   ignore_index=-100)
        if not return_dict:
            rest = (logits,) + ((out.hidden_states,) if out.hidden_states is not None else ())
            return ((loss,) + rest) if loss is not None else rest
        return MaskedLMOutput(loss=loss, logits=logits, hidden_states=out.hidden_states)


class GSXAForSequenceClassification(GSXAPreTrainedModel):
    """Pooled (masked mean, or the first token) -> prediction head -> classifier."""

    def __init__(self, config: GSXAConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = GSXAModel(config)
        self.head = GSXAPredictionHead(config)
        self.drop = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         inputs_embeds=inputs_embeds,
                         output_hidden_states=output_hidden_states, return_dict=True)
        h = out.last_hidden_state
        if self.config.classifier_pooling == "cls":
            pooled = h[:, 0]
        elif attention_mask is None:
            pooled = h.mean(dim=1)
        else:
            m = attention_mask.to(h.dtype).unsqueeze(-1)
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(self.drop(self.head(pooled)))

        loss = None
        if labels is not None:
            problem = self.config.problem_type
            if problem is None:
                if self.num_labels == 1:
                    problem = "regression"
                elif labels.dtype in (torch.long, torch.int):
                    problem = "single_label_classification"
                else:
                    problem = "multi_label_classification"
                self.config.problem_type = problem
            if problem == "regression":
                loss = F.mse_loss(logits.squeeze(-1).float(), labels.float().view(-1)) \
                    if self.num_labels == 1 else F.mse_loss(logits.float(), labels.float())
            elif problem == "single_label_classification":
                loss = F.cross_entropy(logits.float().view(-1, self.num_labels), labels.view(-1))
            else:
                loss = F.binary_cross_entropy_with_logits(logits.float(), labels.float())
        if not return_dict:
            rest = (logits,) + ((out.hidden_states,) if out.hidden_states is not None else ())
            return ((loss,) + rest) if loss is not None else rest
        return SequenceClassifierOutput(loss=loss, logits=logits,
                                        hidden_states=out.hidden_states)


__all__ = [
    "GSXAPreTrainedModel",
    "GSXAModel",
    "GSXAForMaskedLM",
    "GSXAForSequenceClassification",
]
