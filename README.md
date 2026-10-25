# GSXA: Gated Scalable Exclusive Self Attention

**GSXA** is a drop-in replacement for softmax self-attention that aims at higher
accuracy. It adds three components to scaled dot-product attention:

* a **scalable factor**: a learned, length-dependent query scale;
* an **exclusive projection**: removes, to a learned degree, the part of each head's
  output that points along the token's own value;
* a **per-head gate**.

This repository implements GSXA (PyTorch + 🤗 `transformers`) together with the four
attentions it is compared with, and reports a controlled comparison in a masked-LM
encoder.

<p align="center">
  <img src="docs/figures/fig1_bars_summary.png" alt="GSXA reduces pseudo-perplexity by 11.3 % and raises MLM accuracy by 1.27 points over GQA, more than Gated Attention, XSA or Scalable GQA" width="60%">
</p>

**Figure 1.** Gain over plain GQA, averaged over sequence lengths 64–4,096, for
~100 M-parameter encoders trained identically on 1 B tokens: masked-LM
pseudo-perplexity reduction (top) and MLM-accuracy gain (bottom).

## Method

For an input $X \in \mathbb{R}^{T \times d}$ with $H$ query heads of width $D$
sharing $H_{kv}$ KV heads (query head $h$ reads KV head $\kappa(h)$), and $n$
non-padding tokens:

**① Scalable factor.** Each layer learns $s$ and $b$, with a fixed floor
$\beta = 0.1$, and scales the queries:

$$
\lambda(n) = s \log n + \beta + \mathrm{ReLU}(b), \qquad
o_t^h = \sum_j \mathrm{softmax}_j\!\left(\frac{\lambda(n)\,\langle \tilde q_t^h, \tilde k_j^{\kappa(h)} \rangle}{\sqrt{D}}\right) v_j^{\kappa(h)}
$$

($\tilde q, \tilde k$: rotary-embedded). The factor keeps attention sharp as $n$ grows,
as in Scalable-Softmax, and the floor keeps $\lambda \ge \beta > 0$ at every length.

**② Exclusive projection and gate.** With
$\hat v_t^h = v_t^{\kappa(h)} / \lVert v_t^{\kappa(h)} \rVert$ the direction of the
token's own value and $[g_{1,t}; g_{2,t}] = W_G\, [o_t^1; \dots; o_t^H] \in \mathbb{R}^{2H}$:

$$
z_t^h = g_{1,t}^h \Big( o_t^h - \tanh\!\big(g_{2,t}^h\big)\,
\big\langle o_t^h, \hat v_t^h \big\rangle\, \hat v_t^h \Big),
\qquad y_t = W_O\, [z_t^1; \dots; z_t^H].
$$

Writing $o = o_\perp + o_\parallel$ (orthogonal to and along $\hat v$), this is
$z = g_1 \big(o_\perp + (1 - \tanh g_2)\, o_\parallel\big)$. Each head always passes
on what the context adds ($o_\perp$). It learns per token how much of the token's own
information to keep: none ($\tanh g_2 \to 1$, as in XSA), all ($g_2 = 0$), or more.
$g_1$ then scales the head.

**Cost.** $W_G \in \mathbb{R}^{2H \times d}$ plus two scalars per layer: +0.22 %
parameters at the size below.

The implementation is `GSXAAttention` in
[`src/gsxa/modeling_gsxa.py`](src/gsxa/modeling_gsxa.py).

## Compared methods

All five share everything except ① and ②; select one with
`GSXAConfig(attention_type=...)`.

| Name | `attention_type` | ① query scale | ② head output $z^h$ |
|:--|:--|:--:|:--|
| **GSXA** | `gsxa` | $\lambda(n)$ | $g_1^h\,(o^h - \tanh(g_2^h) \langle o^h, \hat v^h \rangle \hat v^h)$ |
| Gated Attention | `gated` | $\lambda(n)$ | $g_1^h\, o^h$ |
| XSA | `xsa` | $\lambda(n)$ | $o^h - \langle o^h, \hat v^h \rangle \hat v^h$ |
| Scalable GQA | `scalable_gqa` | $\lambda(n)$ | $o^h$ |
| GQA | `gqa` | $1$ | $o^h$ |

* **GQA**: grouped-query attention with RoPE (Ainslie et al., 2023; Su et al., 2021).
* **Scalable GQA**: GQA with the scalable factor, Scalable-Softmax (Nakanishi, 2025)
  plus the floor $\beta + \mathrm{ReLU}(b)$.
* **XSA**: Exclusive Self Attention (Zhai, 2026), in its parameter-free form.
* **Gated Attention**: GSXA's gate alone. It is linear, per head and computed from the
  attention output. It is not the sigmoid gate on the layer input of Qiu et al. (2025).

## Results

**Setup.** All five models are 12-layer encoders: width 768, 12 query / 4 KV heads,
RoPE θ = 100,000, a ReLU² MLP of 2,304, 100.7–100.9 M parameters. Each was trained
once, on 1 B tokens of English web and code at sequence length 1,024. The recipe and
data order are identical: masked LM with 30 % masking, batch 512, AdamW at learning
rate 6 × 10⁻⁴ with cosine decay, seed 1234. Evaluation uses held-out text: for each
length, 20 measurements of 65,536 tokens with 15 % masking, reported as mean ± 95 % CI.
The metrics are pseudo-perplexity (PPPL, Salazar et al., 2020) and MLM accuracy. Raw
scores are in [`results/attention_ablation/`](results/attention_ablation).

**Table 1.** Masked-LM quality by sequence length († = beyond the training length;
bold = best in the column).

*Pseudo-perplexity ↓*

| Model | 64 | 128 | 256 | 512 | 1,024 | 2,048† | 4,096† |
|:--|--:|--:|--:|--:|--:|--:|--:|
| GSXA | **18.53 ± 0.35** | **13.51 ± 0.39** | **11.25 ± 0.29** | **10.18 ± 0.28** | **9.67 ± 0.30** | **9.82 ± 0.40** | **10.69 ± 0.49** |
| Gated Attention | 18.72 ± 0.36 | 13.72 ± 0.39 | 11.52 ± 0.30 | 10.42 ± 0.30 | 9.89 ± 0.30 | 9.99 ± 0.39 | 10.98 ± 0.49 |
| XSA | 18.82 ± 0.33 | 13.64 ± 0.38 | 11.43 ± 0.29 | 10.36 ± 0.27 | 9.97 ± 0.31 | 10.25 ± 0.42 | 11.20 ± 0.54 |
| Scalable GQA | 19.41 ± 0.40 | 13.86 ± 0.38 | 11.53 ± 0.31 | 10.52 ± 0.28 | 10.20 ± 0.31 | 10.64 ± 0.42 | 11.50 ± 0.60 |
| GQA | 21.42 ± 0.41 | 14.64 ± 0.42 | 11.95 ± 0.30 | 10.78 ± 0.29 | 10.39 ± 0.32 | 11.15 ± 0.45 | 14.42 ± 0.97 |

*MLM accuracy (%) ↑*

| Model | 64 | 128 | 256 | 512 | 1,024 | 2,048† | 4,096† |
|:--|--:|--:|--:|--:|--:|--:|--:|
| GSXA | **47.87 ± 0.34** | 52.20 ± 0.39 | **55.04 ± 0.33** | **56.45 ± 0.36** | **57.37 ± 0.43** | **57.33 ± 0.49** | **56.23 ± 0.66** |
| Gated Attention | 47.80 ± 0.32 | 52.05 ± 0.39 | 54.69 ± 0.39 | 56.22 ± 0.39 | 56.98 ± 0.47 | 57.08 ± 0.52 | 55.88 ± 0.69 |
| XSA | 47.84 ± 0.27 | **52.22 ± 0.41** | 54.88 ± 0.30 | 56.38 ± 0.34 | 56.98 ± 0.46 | 56.76 ± 0.51 | 55.74 ± 0.67 |
| Scalable GQA | 47.48 ± 0.33 | 52.01 ± 0.38 | 54.77 ± 0.33 | 56.21 ± 0.38 | 56.72 ± 0.44 | 56.45 ± 0.49 | 55.58 ± 0.72 |
| GQA | 46.93 ± 0.30 | 51.70 ± 0.37 | 54.45 ± 0.32 | 55.85 ± 0.32 | 56.43 ± 0.43 | 55.58 ± 0.49 | 52.69 ± 0.81 |

<p align="center">
  <img src="docs/figures/fig2_bars_by_length.png" alt="Pseudo-perplexity reduction and MLM accuracy gain over GQA at each length from 64 to 4,096" width="100%">
</p>

**Figure 2.** Gain over GQA at each length. The whiskers are 95 % CIs that treat the
two models as independent; they are conservative, since all models share the
evaluation windows and masks.

* GSXA has the lowest pseudo-perplexity at every length and the highest MLM accuracy
  at all but one (128).
* Against GQA it reduces pseudo-perplexity by 7.0 % at the training length. The
  intervals do not overlap there. Beyond the training length the reduction is
  11.9 % at 2,048 and 25.9 % at 4,096.
* Against the other Scalable-GQA variants its margin at any single length is within
  the intervals. What sets it apart is that it leads consistently across all lengths.
  The comparison rests on one training run per model.

## Speed

Encoder forward pass with `torch.compile`, 32 × 1,024 tokens, bf16, one RTX 5080.
The models are timed in 10 interleaved rounds, each the median of 50 forwards. The
table gives the mean ± 95 % CI over the rounds. The change vs GQA is paired within
each round.

| | GSXA | Gated Attention | XSA | Scalable GQA | GQA |
|:--|--:|--:|--:|--:|--:|
| latency (ms) | 82.14 ± 0.09 | 81.14 ± 0.08 | 81.70 ± 0.08 | 80.29 ± 0.08 | 80.22 ± 0.07 |
| vs GQA | +2.39 ± 0.11 % | +1.14 ± 0.11 % | +1.84 ± 0.11 % | +0.08 ± 0.11 % | — |

To reproduce: `python benchmarks/benchmark_inference.py`, then
`python benchmarks/make_tables.py <result json>`.

## Usage

```bash
pip install -e .            # torch >= 2.4, transformers >= 5.0
```

```python
import torch
from gsxa import GSXAConfig, GSXAForMaskedLM

model = GSXAForMaskedLM(GSXAConfig(attention_type="gsxa"))      # 12 x 768, untrained
model = GSXAForMaskedLM.from_pretrained("path/to/checkpoint", dtype=torch.bfloat16)
```

* `GSXAModel`, `GSXAForMaskedLM` and `GSXAForSequenceClassification` are standard
  `transformers` models. A saved checkpoint also loads without this package, through
  `AutoModel*.from_pretrained(path, trust_remote_code=True)`.
* The tokenizer is ModernBERT's: `answerdotai/ModernBERT-base`.
* [`scripts/convert_nexterabert_v2.py`](scripts/convert_nexterabert_v2.py) converts
  the ablation checkpoints, which were trained with a separate pipeline (NexteraBERTv2).
  The conversion is exact.
* [`examples/quickstart.py`](examples/quickstart.py) shows fill-mask, embeddings and
  classification.
* `python -m pytest -q tests` runs the CPU tests.
* `python results/make_figures.py --tables` redraws the figures and tables.

## References

* S. Zhai. *Exclusive Self Attention.* arXiv:2603.09078, 2026.
* Z. Qiu et al. *Gated Attention for Large Language Models: Non-linearity, Sparsity,
  and Attention-Sink-Free.* arXiv:2505.06708, 2025.
* K. M. Nakanishi. *Scalable-Softmax Is Superior for Attention.* arXiv:2501.19399, 2025.
* J. Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from
  Multi-Head Checkpoints.* arXiv:2305.13245, 2023.
* J. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.*
  arXiv:2104.09864, 2021.
* J. Salazar et al. *Masked Language Model Scoring.* ACL 2020.

## Citation

```bibtex
@misc{gsxa,
  title        = {{GSXA}: Gated Scalable Exclusive Self Attention},
  author       = {Rikka Botan},
  year         = {2026},
  howpublished = {\url{https://github.com/Rikka-Botan/GSXA}}
}
```

## License

MIT — see [LICENSE](LICENSE).
