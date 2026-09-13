"""Absorbing-mask OCR: explicit forward kernel, discrete ELBO and reverse kernel.

See thesis/DIFFUSION.md for the derivation and departures from MDiff4STR.
This is an independently implemented mask-only baseline, not an OpenOCR port.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .tokenizer import CharacterTokenizer as Tokens


@dataclass
class ModelConfig:
    vocab_size: int
    capacity: int
    d_model: int = 192
    heads: int = 6
    encoder_layers: int = 2
    decoder_layers: int = 3
    patch_size: int = 8
    dropout: float = 0.1
    diffusion_steps: int = 256
    encoder_kind: str = "patch"
    vit_config: dict | None = None
    decoder_kind: str = "diffusion"
    eos_target: str = "repeat"
    full_mask_probability: float = 0.0
    text_policy: str = "literal"
    excluded_token_ids: tuple = ()

    def __post_init__(self):
        if self.vocab_size < 4 or self.capacity < 1 or self.heads < 1:
            raise ValueError("Invalid vocabulary, capacity or attention heads")
        if self.d_model < 4 or self.d_model % self.heads or self.d_model % 4:
            raise ValueError("d_model must be divisible by heads and by four")
        if min(self.encoder_layers, self.decoder_layers, self.patch_size, self.diffusion_steps) < 1:
            raise ValueError("Layers, patch size and diffusion steps must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.encoder_kind not in ("patch", "vit"):
            raise ValueError("Unknown visual encoder")
        if self.encoder_kind == "vit" and not self.vit_config:
            raise ValueError("ViT encoder requires saved configuration")
        if self.decoder_kind not in ("diffusion", "autoregressive"):
            raise ValueError("Unknown decoder kind")
        if self.eos_target not in ("repeat", "single"):
            raise ValueError("Unknown EOS target policy")
        if not 0 <= self.full_mask_probability <= 1:
            raise ValueError("Full-mask probability must be in [0, 1]")
        if self.decoder_kind != "diffusion" and self.full_mask_probability:
            raise ValueError("Full-mask training applies only to diffusion")
        self.excluded_token_ids = tuple(self.excluded_token_ids)
        if self.text_policy not in ("literal", "no_harakat_v1"):
            raise ValueError("Unknown training text policy")
        if any(i < 3 or i >= self.vocab_size for i in self.excluded_token_ids):
            raise ValueError("Excluded output IDs must be literal character IDs")

    def as_dict(self):
        return asdict(self)


def corrupt_tokens(targets, t, *, generator=None):
    """q_t: each non-PAD clean token is masked with probability t, independently."""
    if t.shape != (targets.shape[0],) or not torch.isfinite(t).all() or (t < 0).any() or (t > 1).any():
        raise ValueError("t must have one finite value in [0, 1] per sample")
    if (targets == Tokens.MASK).any():
        raise ValueError("Clean targets cannot contain MASK")
    mask = (torch.rand(targets.shape, device=targets.device, generator=generator) < t[:, None]) & targets.ne(Tokens.PAD)
    noisy = targets.masked_fill(mask, Tokens.MASK)
    return noisy, mask


def diffusion_loss(logits, targets, masked, t):
    """Uniform discrete-time ELBO estimator, normalized by non-PAD positions.

    At t=k/T the reverse reveal probability is 1/k; sampling k uniformly
    multiplies each masked CE by T/k = 1/t. Empty masks contribute zero;
    forcing a mask would change the forward distribution.
    """
    if t.shape != (targets.shape[0],) or not torch.isfinite(t).all() or (t <= 0).any() or (t > 1).any():
        raise ValueError("Training t must be in (0, 1]")
    valid = targets.ne(Tokens.PAD)
    ce = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none", ignore_index=Tokens.PAD)
    terms = torch.where(masked & valid, ce, torch.zeros_like(ce))
    return (terms.sum(dim=1) / t / valid.sum(dim=1).clamp_min(1)).mean()


def reverse_step(tokens, logits, t: float, s: float, *, generator=None):
    """Learned reverse kernel for alpha(t)=1-t; revealed tokens are carried over."""
    if not 0 <= s < t <= 1:
        raise ValueError("Reverse times must satisfy 0 <= s < t <= 1")
    probabilities = logits.softmax(-1)
    proposed = torch.multinomial(probabilities.flatten(0, 1), 1, generator=generator).reshape_as(tokens)
    reveal = torch.rand(tokens.shape, device=tokens.device, generator=generator) < ((t - s) / t)
    return torch.where(tokens.eq(Tokens.MASK) & reveal, proposed, tokens)


def spatial_positions(height, width, dim, *, device, dtype):
    freq = torch.exp(torch.arange(dim // 4, device=device, dtype=torch.float32) * (-math.log(10000) / max(1, dim // 4)))
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    x, y = x.flatten()[:, None] * freq, y.flatten()[:, None] * freq
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1).to(dtype)


class PatchEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.patch = nn.Conv2d(3, config.d_model, config.patch_size, stride=config.patch_size)
        layer = nn.TransformerEncoderLayer(config.d_model, config.heads, config.d_model * 4,
                                           config.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, config.encoder_layers, norm=nn.LayerNorm(config.d_model), enable_nested_tensor=False)

    def forward(self, pixels, widths):
        patches = self.patch(pixels)
        _, dim, height, width = patches.shape
        memory = patches.flatten(2).transpose(1, 2)
        memory = memory + spatial_positions(height, width, dim, device=pixels.device, dtype=memory.dtype)
        columns = torch.arange(width, device=pixels.device).repeat(height)
        mask = columns[None, :] >= torch.div(widths[:, None] + self.patch_size - 1, self.patch_size, rounding_mode="floor")
        memory = self.transformer(memory, src_key_padding_mask=mask)
        return memory, mask


class LocalViTEncoder(nn.Module):
    """Optional pretrained ViT adapter. Always reconstructed without network access.

    Encode each unpadded crop separately: ViT does not accept a visual padding
    attention mask. Rectangular positional embeddings are interpolated by ViT.
    Padded feature sequences are masked in the OCR decoder.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        from transformers import ViTConfig, ViTModel

        self.vit = ViTModel(ViTConfig(**config.vit_config), add_pooling_layer=False)
        self.patch_size = self.vit.config.patch_size
        if self.patch_size != config.patch_size:
            raise ValueError("ViT patch size does not match preprocessing")
        self.projection = nn.Linear(self.vit.config.hidden_size, config.d_model)

    def forward(self, pixels, widths):
        memories = []
        for i, width in enumerate(widths.tolist()):
            extent = math.ceil(width / self.patch_size) * self.patch_size
            result = self.vit(pixel_values=pixels[i:i + 1, :, :, :extent], interpolate_pos_encoding=True)
            memories.append(self.projection(result.last_hidden_state[0, 1:]))
        memory = nn.utils.rnn.pad_sequence(memories, batch_first=True)
        lengths = torch.tensor([len(m) for m in memories], device=pixels.device)
        mask = torch.arange(memory.shape[1], device=pixels.device)[None, :] >= lengths[:, None]
        return memory, mask


class DiffusionRecognizer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = PatchEncoder(config) if config.encoder_kind == "patch" else LocalViTEncoder(config)
        self.tokens = nn.Embedding(config.vocab_size, config.d_model, padding_idx=Tokens.PAD)
        self.positions = nn.Parameter(torch.empty(1, config.capacity, config.d_model))
        nn.init.normal_(self.positions, std=0.02)
        self.time = nn.Sequential(nn.Linear(1, config.d_model), nn.SiLU(), nn.Linear(config.d_model, config.d_model))
        layer = nn.TransformerDecoderLayer(config.d_model, config.heads, config.d_model * 4, config.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, config.decoder_layers, norm=nn.LayerNorm(config.d_model))
        # Only EOS and literal characters can be predicted. PAD and MASK have zero probability.
        self.head = nn.Linear(config.d_model, config.vocab_size - 2)

    def denoise(self, noisy, memory, visual_mask, t, *, causal=False):
        if noisy.shape[1] != self.config.capacity:
            raise ValueError("Token sequence must have configured capacity")
        embedded = self.tokens(noisy) + self.positions + self.time(t[:, None])[:, None, :]
        causal_mask = torch.ones(noisy.shape[1], noisy.shape[1], device=noisy.device, dtype=torch.bool).triu(1) if causal else None
        features = self.decoder(embedded, memory, tgt_mask=causal_mask, memory_key_padding_mask=visual_mask)
        logits = self.head(features)
        blocked = logits.new_full((*logits.shape[:-1], 2), -torch.inf)
        result = torch.cat((blocked, logits), dim=-1)
        if self.config.excluded_token_ids:
            result[..., list(self.config.excluded_token_ids)] = -torch.inf
        return result

    def forward(self, pixels, widths, targets, *, generator=None):
        if targets.shape[1] != self.config.capacity:
            raise ValueError("Targets must have configured capacity; no truncation")
        if self.config.decoder_kind == "autoregressive":
            # Matched visual encoder control: MASK is the start input, never an output.
            shifted = torch.cat((torch.full_like(targets[:, :1], Tokens.MASK), targets[:, :-1]), dim=1)
            memory, visual_mask = self.encoder(pixels, widths)
            logits = self.denoise(shifted, memory, visual_mask, pixels.new_zeros(pixels.shape[0]), causal=True)
            ce = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none", ignore_index=Tokens.PAD)
            return (ce.sum(1) / targets.ne(Tokens.PAD).sum(1).clamp_min(1)).mean()
        steps = torch.randint(1, self.config.diffusion_steps + 1, (targets.shape[0],), device=pixels.device, generator=generator)
        t = steps.float() / self.config.diffusion_steps
        if self.config.full_mask_probability:
            # Explicit mixture objective: (1-p) * original loss + p * full-mask CE.
            # This is NOT an unbiased estimator of the original uniform-time ELBO.
            full = torch.rand(t.shape, device=t.device, generator=generator) < self.config.full_mask_probability
            t = torch.where(full, torch.ones_like(t), t)
        noisy, masked = corrupt_tokens(targets, t, generator=generator)
        # PAD is for explicit loss exclusion, never a target-length hint to attention.
        noisy = noisy.masked_fill(noisy.eq(Tokens.PAD), Tokens.MASK)
        memory, visual_mask = self.encoder(pixels, widths)
        logits = self.denoise(noisy, memory, visual_mask, t)
        return diffusion_loss(logits, targets, masked, t)

    @torch.no_grad()
    def sample(self, pixels, widths, *, steps=16, generator=None, return_trace=False):
        if self.training:
            raise ValueError("Call model.eval() before sampling")
        if steps < 1:
            raise ValueError("Sampling steps must be positive")
        memory, visual_mask = self.encoder(pixels, widths)
        tokens = torch.full((pixels.shape[0], self.config.capacity), Tokens.MASK, dtype=torch.long, device=pixels.device)
        trace = [tokens.cpu().tolist()] if return_trace else None
        if self.config.decoder_kind == "autoregressive":
            shifted = tokens.clone()
            finished = torch.zeros(pixels.shape[0], dtype=torch.bool, device=pixels.device)
            for index in range(self.config.capacity):
                logits = self.denoise(shifted, memory, visual_mask, pixels.new_zeros(pixels.shape[0]), causal=True)
                next_token = logits[:, index].argmax(-1)
                next_token = torch.where(finished, Tokens.EOS, next_token)
                tokens[:, index] = next_token
                finished |= next_token.eq(Tokens.EOS)
                if index + 1 < self.config.capacity:
                    shifted[:, index + 1] = next_token
                if finished.all():
                    tokens[:, index + 1:] = Tokens.EOS
                if return_trace:
                    trace.append(tokens.cpu().tolist())
                if finished.all():
                    break
            return tokens, trace
        for k in range(steps, 0, -1):
            t = torch.full((pixels.shape[0],), k / steps, device=pixels.device)
            logits = self.denoise(tokens, memory, visual_mask, t)
            tokens = reverse_step(tokens, logits, k / steps, (k - 1) / steps, generator=generator)
            if return_trace:
                trace.append(tokens.cpu().tolist())
        return tokens, trace
