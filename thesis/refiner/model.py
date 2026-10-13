"""Small conditional absorbing diffusion decoder; the TrOCR backbone is cached."""

from dataclasses import asdict, dataclass

import torch
from torch import nn

from thesis.diffusion.model import corrupt_tokens, diffusion_loss, reverse_step
from thesis.diffusion.tokenizer import CharacterTokenizer as Tokens


@dataclass
class RefinerConfig:
    vocab_size: int
    capacity: int
    visual_dim: int
    d_model: int = 192
    heads: int = 6
    layers: int = 3
    dropout: float = 0.1
    diffusion_steps: int = 256
    full_mask_probability: float = 0.25
    draft_dropout: float = 0.5

    def __post_init__(self):
        if min(self.capacity, self.visual_dim, self.d_model, self.heads, self.layers, self.diffusion_steps) < 1:
            raise ValueError("Model dimensions and diffusion steps must be positive")
        if self.vocab_size < 4 or self.d_model % self.heads:
            raise ValueError("Invalid vocabulary or attention dimensions")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 <= self.full_mask_probability <= 1 or not 0 <= self.draft_dropout <= 1:
            raise ValueError("Corruption mixture and draft dropout must be in [0, 1]")

    def as_dict(self):
        return asdict(self)


class DiffusionRefiner(nn.Module):
    def __init__(self, config: RefinerConfig):
        super().__init__()
        self.config = config
        self.visual = nn.Sequential(nn.LayerNorm(config.visual_dim), nn.Linear(config.visual_dim, config.d_model))
        self.tokens = nn.Embedding(config.vocab_size, config.d_model, padding_idx=Tokens.PAD)
        self.positions = nn.Parameter(torch.randn(1, config.capacity, config.d_model) * 0.02)
        self.modality = nn.Parameter(torch.randn(2, config.d_model) * 0.02)
        self.time = nn.Sequential(nn.Linear(1, config.d_model), nn.SiLU(), nn.Linear(config.d_model, config.d_model))
        layer = nn.TransformerDecoderLayer(config.d_model, config.heads, 4 * config.d_model,
                                           config.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, config.layers, nn.LayerNorm(config.d_model))
        self.head = nn.Linear(config.d_model, config.vocab_size - 2)

    def condition(self, visual, visual_mask, draft, *, generator=None):
        if draft.shape[1] != self.config.capacity:
            raise ValueError("Draft capacity mismatch")
        # Drop whole drafts, never visual evidence. This reduces reliance on memorized
        # training predictions when the base recognizer was fitted on training lines.
        if self.training and self.config.draft_dropout:
            drop = torch.rand((len(draft), 1), device=draft.device, generator=generator) < self.config.draft_dropout
            draft = torch.where(drop, torch.full_like(draft, Tokens.PAD), draft)
        memory = torch.cat((self.visual(visual) + self.modality[0],
                            self.tokens(draft) + self.positions + self.modality[1]), dim=1)
        mask = torch.cat((visual_mask, draft.eq(Tokens.PAD)), dim=1)
        return memory, mask

    def denoise(self, noisy, memory, memory_mask, t):
        x = self.tokens(noisy) + self.positions + self.time(t[:, None])[:, None, :]
        logits = self.head(self.decoder(x, memory, memory_key_padding_mask=memory_mask))
        return torch.cat((logits.new_full((*logits.shape[:-1], 2), -torch.inf), logits), dim=-1)

    def forward(self, visual, visual_mask, draft, targets, *, generator=None):
        if targets.shape != draft.shape or targets.shape[1] != self.config.capacity:
            raise ValueError("Target capacity mismatch; labels cannot be truncated")
        k = torch.randint(1, self.config.diffusion_steps + 1, (len(targets),), device=targets.device, generator=generator)
        t = k.float() / self.config.diffusion_steps
        full = torch.rand(t.shape, device=t.device, generator=generator) < self.config.full_mask_probability
        t = torch.where(full, torch.ones_like(t), t)
        noisy, masked = corrupt_tokens(targets, t, generator=generator)
        noisy = noisy.masked_fill(noisy.eq(Tokens.PAD), Tokens.MASK)
        memory, memory_mask = self.condition(visual, visual_mask, draft, generator=generator)
        return diffusion_loss(self.denoise(noisy, memory, memory_mask, t), targets, masked, t)

    @torch.no_grad()
    def sample(self, visual, visual_mask, draft, *, steps=8, mode="diffusion", generator=None):
        if self.training:
            raise ValueError("Sampling requires eval mode")
        if steps < 1 or mode not in ("diffusion", "one_pass"):
            raise ValueError("Invalid sampler")
        memory, mask = self.condition(visual, visual_mask, draft)
        tokens = torch.full_like(draft, Tokens.MASK)
        if mode == "one_pass":
            return self.denoise(tokens, memory, mask, visual.new_ones(len(draft))).argmax(-1)
        for k in range(steps, 0, -1):
            t = visual.new_full((len(draft),), k / steps)
            logits = self.denoise(tokens, memory, mask, t)
            tokens = reverse_step(tokens, logits, k / steps, (k - 1) / steps, generator=generator)
        return tokens


def encode_draft(tokenizer, text, capacity):
    """Only the *draft* may be clipped/unknown. Targets retain the strict tokenizer."""
    ids = [tokenizer.ids.get(char, Tokens.MASK) for char in text[:capacity - 1]] + [Tokens.EOS]
    return ids + [Tokens.PAD] * (capacity - len(ids))
