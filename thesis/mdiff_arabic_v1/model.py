"""MDiff-inspired full-line adaptation, not a faithful OpenOCR reproduction.

Two explicit decoder calls per training batch: structured masked denoising and
visible-token correction. AR uses two dropout views at the same compute budget.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .vendor.svtrv2 import SVTRv2LNConvTwo33

PAD, MASK, EOS = 0, 1, 2
ENCODER_CONFIG = dict(in_channels=3, use_pos_embed=False, dims=[128, 256, 384],
    depths=[3, 6, 3], num_heads=[4, 8, 12],
    mixer=[['Conv']*3, ['Conv']*2+['FGlobal', 'Global', 'Global', 'Global'], ['Global']*3],
    sub_k=[[1, 1], [2, 1], [-1, -1]], last_stage=False, feat2d=False)


def tensor_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(key.encode()+b'\0'+str(value.dtype).encode()+str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def load_encoder(encoder, path: Path):
    """Reject partial transfer; Arabic heads are freshly initialized separately."""
    saved = torch.load(path, map_location='cpu', weights_only=True)
    state = saved.get('state_dict', saved.get('model', saved))
    expected = encoder.state_dict()
    selected = {k[len('encoder.'):]: v for k, v in state.items() if k.startswith('encoder.')}
    if set(selected) != set(expected):
        raise ValueError(f'Encoder keys incompatible: missing={sorted(set(expected)-set(selected))}; extra={sorted(set(selected)-set(expected))}')
    bad = [k for k in expected if expected[k].shape != selected[k].shape]
    if bad:
        raise ValueError(f'Encoder tensor shapes incompatible: {bad}')
    encoder.load_state_dict(selected, strict=True)
    loaded_hash = tensor_hash(encoder.state_dict())
    if loaded_hash != tensor_hash(selected):
        raise ValueError('Encoder transfer did not preserve checkpoint tensors')
    return dict(encoder_initialization_sha256=loaded_hash,
        loaded_parameters=[dict(name=k, shape=list(v.shape), elements=v.numel()) for k,v in selected.items()],
        skipped_parameters=[dict(name=k, shape=list(v.shape), reason='non-encoder; Arabic decoder initialized afresh')
                            for k,v in state.items() if not k.startswith('encoder.')],
        missing_encoder_parameters=[], loaded_encoder_elements=sum(v.numel() for v in selected.values()))


def positions(length, dim, device, dtype):
    p = torch.arange(length, device=device).float()[:, None]
    f = torch.exp(torch.arange(0, dim, 2, device=device).float()*(-math.log(10000)/dim))
    out = torch.zeros(length, dim, device=device)
    out[:, 0::2], out[:, 1::2] = torch.sin(p*f), torch.cos(p*f)
    return out.to(dtype)


def structured_noise(targets):
    """Uniform six-pattern mixture; t is the actual fraction of valid masks.

    Full, logical prefix/suffix, contiguous block, cloze, independent random.
    This normalized masked CE mixture is a surrogate, not an unbiased ELBO.
    """
    valid = targets.ne(PAD)
    hidden = torch.zeros_like(valid)
    for row in range(len(targets)):
        n = int(valid[row].sum())
        mode = int(torch.randint(6, (), device=targets.device))
        count = int(torch.randint(1, n+1, (), device=targets.device))
        if mode == 0:
            hidden[row, :n] = True
        elif mode == 1:
            hidden[row, :count] = True
        elif mode == 2:
            hidden[row, n-count:n] = True
        elif mode == 3:
            start = int(torch.randint(n-count+1, (), device=targets.device))
            hidden[row, start:start+count] = True
        elif mode == 4:
            hidden[row, int(torch.randint(n, (), device=targets.device))] = True
        else:
            hidden[row, torch.randperm(n, device=targets.device)[:count]] = True
    noisy = targets.masked_fill(hidden | ~valid, MASK)
    return noisy, hidden, hidden.sum(1).float()/valid.sum(1)


def correction_noise(targets, vocab_size):
    """25% identity examples; otherwise random rate of genuinely wrong literals.

    EOS can be replaced too, so boundary correction is learned. PAD becomes
    MASK and never supplies a target-length attention hint.
    """
    rate = torch.rand(len(targets), device=targets.device)
    rate = torch.where(torch.rand_like(rate)<.25, 0., rate)
    replaced = (torch.rand(targets.shape, device=targets.device)<rate[:, None]) & targets.ne(PAD)
    proposal = torch.randint(2, vocab_size-1, targets.shape, device=targets.device)
    proposal += proposal.ge(targets).long()  # uniform other output token, EOS included
    noisy = torch.where(replaced, proposal, targets).masked_fill(targets.eq(PAD), MASK)
    return noisy, replaced, rate


def normalized_ce(logits, targets, mask):
    ce = F.cross_entropy(logits.float().transpose(1,2), targets, reduction='none', ignore_index=PAD)
    return ((ce*mask).sum(1)/mask.sum(1).clamp_min(1)).mean()


class Recognizer(nn.Module):
    def __init__(self, vocab_size, capacity=144, arm='mdiff', dim=256, layers=3, dropout=.1):
        super().__init__()
        if arm not in ('mdiff', 'ar'):
            raise ValueError('Unknown model arm')
        self.arm, self.capacity = arm, capacity
        self.encoder = SVTRv2LNConvTwo33(**ENCODER_CONFIG)
        self.visual = nn.Linear(384, dim)
        self.embedding = nn.Embedding(vocab_size, dim)
        self.time = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        block = nn.TransformerDecoderLayer(dim, 8, dim*4, dropout=dropout,
                                           batch_first=True, norm_first=True, activation='gelu')
        self.decoder = nn.TransformerDecoder(block, layers, norm=nn.LayerNorm(dim))
        self.output = nn.Linear(dim, vocab_size)
        self.gradient_checkpointing = True
        self.decoder_passes = 0

    def train(self, mode=True):
        super().train(mode)
        # Keep pretrained BN statistics fixed for tiny per-image batches.
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def encode(self, pixels, widths):
        features = []
        for index, width in enumerate(widths.tolist()):
            # No batch padding enters SVTR attention or convolutions. Only the
            # <=3-pixel white alignment border required by stride four remains.
            w = math.ceil(width/4)*4
            x, size = self.encoder.pope(pixels[index:index+1, :, :, :w])
            for stage in self.encoder.stages:
                if self.training and self.gradient_checkpointing and any(p.requires_grad for p in stage.parameters()):
                    x, size = checkpoint(stage, x, size, use_reentrant=False)
                else:
                    x, size = stage(x, size)
            x = self.visual(x[0])
            x = x + positions(len(x), x.shape[-1], x.device, x.dtype)
            features.append(x)
        memory = nn.utils.rnn.pad_sequence(features, batch_first=True)
        lengths = torch.tensor([len(x) for x in features], device=memory.device)
        mask = torch.arange(memory.shape[1], device=memory.device)[None, :] >= lengths[:,None]
        return memory, mask

    def decode_logits(self, tokens, memory, memory_mask, t, causal=False):
        self.decoder_passes += 1
        x = self.embedding(tokens)
        x = x + positions(tokens.shape[1], x.shape[-1], x.device, x.dtype)
        # Same modules/initialization in both arms; AR time is the fixed zero state.
        x = x + self.time(t[:,None].to(x.dtype))[:,None,:]
        mask = torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=x.device).triu(1) if causal else None
        x = self.decoder(x, memory, tgt_mask=mask, memory_key_padding_mask=memory_mask)
        logits = self.output(x)
        # PAD/MASK are inputs only and can never be emitted.
        logits = logits.clone()
        logits[..., :2] = torch.finfo(logits.dtype).min
        return logits

    def loss(self, pixels, widths, targets):
        memory, memory_mask = self.encode(pixels, widths)
        valid = targets.ne(PAD)
        if self.arm == 'ar':
            shifted = torch.cat((torch.full_like(targets[:,:1], MASK), targets[:,:-1]), 1)
            terms = [normalized_ce(self.decode_logits(shifted, memory, memory_mask,
                        pixels.new_zeros(len(pixels)), causal=True), targets, valid) for _ in range(2)]
            return sum(terms)/2, dict(ar_ce=float((sum(terms)/2).detach()))
        noisy, hidden, t = structured_noise(targets)
        logits = self.decode_logits(noisy, memory, memory_mask, t)
        denoise = normalized_ce(logits, targets, hidden)
        noisy, replaced, rate = correction_noise(targets, self.embedding.num_embeddings)
        # The replacement rate is unknown at inference. Use the same fixed
        # correction state at training and inference, rather than reveal it.
        reflect = self.decode_logits(noisy, memory, memory_mask, torch.zeros_like(rate))
        correction = normalized_ce(reflect, targets, valid)
        # Separately normalized boundary learning; tail cannot overwhelm literals.
        tail_targets = torch.full_like(targets, EOS)
        tail = normalized_ce(logits, tail_targets, ~valid)
        total = (denoise + correction)/2 + .1*tail
        return total, dict(masked_ce=float(denoise.detach()), correction_ce=float(correction.detach()),
            tail_ce=float(tail.detach()), replaced_tokens=int(replaced.sum()), correct_visible_tokens=int((valid & ~replaced).sum()))

    @torch.no_grad()
    def generate(self, pixels, widths, steps=3, initial=None, trace=False):
        if self.training or steps < 1:
            raise ValueError('Decoding requires eval mode and positive steps')
        memory, mm = self.encode(pixels, widths)
        state = torch.full((len(pixels),self.capacity), MASK, dtype=torch.long, device=pixels.device)
        if initial is not None:
            if self.arm != 'mdiff' or initial.shape != state.shape:
                raise ValueError('Initial tokens are only supported by the diffusion diagnostic')
            state = initial.clone().masked_fill(initial.eq(PAD),MASK)
        history = []
        before = self.decoder_passes
        if self.arm == 'ar':
            shifted = state.clone()
            done = torch.zeros(len(pixels), dtype=torch.bool, device=pixels.device)
            for pos in range(self.capacity):
                logits = self.decode_logits(shifted[:,:pos+1], memory, mm, pixels.new_zeros(len(pixels)), causal=True)
                value = logits[:,-1].argmax(-1)
                value = torch.where(done, EOS, value)
                state[:,pos] = value
                done |= value.eq(EOS)
                if pos+1<self.capacity:
                    shifted[:,pos+1] = value
                if done.all():
                    state[:,pos+1:] = EOS
                    break
        else:
            for k in range(steps):
                t = pixels.new_full((len(pixels),), (steps-k)/steps)
                logits = self.decode_logits(state, memory, mm, t)
                # Masked CE only trains missing positions. Keep visible tokens
                # here; the separately supervised correction call revises them.
                state = torch.where(state.eq(MASK), logits.argmax(-1), state)
                if trace: history.append(dict(kind='denoise', tokens=state.cpu().tolist()))
                # A distinct correction call revises even confidently wrong visible tokens.
                reflect_input = self.hide_tail(state)
                logits = self.decode_logits(reflect_input, memory, mm, pixels.new_zeros(len(pixels)))
                state = logits.argmax(-1)
                if trace: history.append(dict(kind='correction', tokens=state.cpu().tolist()))
                if k+1 < steps:
                    probability = logits.float().softmax(-1).gather(-1,state[...,None]).squeeze(-1)
                    state = self.hide_tail(state)
                    for row in range(len(state)):
                        active = state[row].ne(MASK).nonzero().flatten()
                        count = math.ceil(len(active)*(steps-k-1)/steps)
                        order = probability[row,active].argsort(stable=True)
                        state[row,active[order[:count]]] = MASK
        return state, dict(decoder_passes=self.decoder_passes-before, sampler_mode='masked_reflect_v1' if self.arm=='mdiff' else 'greedy_causal',
                           sampler_steps=steps if self.arm=='mdiff' else None, trace=history)

    @staticmethod
    def hide_tail(state):
        result = state.clone()
        for row in range(len(state)):
            ends = state[row].eq(EOS).nonzero().flatten()
            if len(ends):
                result[row,int(ends[0])+1:] = MASK
        return result
