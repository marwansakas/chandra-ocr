"""Target-free greedy decoding alternatives for checkpoint diagnostics."""
import math
import torch


@torch.no_grad()
def greedy_decode(model, pixels, widths, *, method, steps=16, generator=None, return_trace=False):
    if model.training or model.config.decoder_kind != "diffusion":
        raise ValueError("Requires an evaluation-mode diffusion model")
    if method not in ("one_pass", "random_reveal", "confidence") or steps < 1:
        raise ValueError("Invalid greedy decoding method or step count")
    memory, visual_mask = model.encoder(pixels, widths)
    tokens = torch.ones((pixels.shape[0], model.config.capacity), dtype=torch.long, device=pixels.device)
    trace = [tokens.cpu().tolist()] if return_trace else None
    calls = 1 if method == "one_pass" else steps
    for index in range(calls):
        t = (calls - index) / calls
        logits = model.denoise(tokens, memory, visual_mask, pixels.new_full((pixels.shape[0],), t))
        proposed = logits.argmax(-1)
        masked = tokens.eq(1)
        if method == "one_pass":
            reveal = masked
        elif method == "random_reveal":
            # Character choices are deterministic; reveal locations remain random.
            reveal = masked & (torch.rand(tokens.shape, device=tokens.device, generator=generator) < 1 / (calls - index))
        else:
            # Fixed capacity, independent of reference lengths. Ties use position.
            confidence = logits.softmax(-1).max(-1).values.masked_fill(~masked, -torch.inf)
            order = confidence.argsort(dim=1, descending=True, stable=True)
            ranks = torch.empty_like(order)
            ranks.scatter_(1, order, torch.arange(tokens.shape[1], device=tokens.device).expand_as(order))
            desired = math.ceil(tokens.shape[1] * (index + 1) / calls)
            additional = desired - (~masked).sum(1)
            reveal = masked & (ranks < additional[:, None])
        tokens = torch.where(reveal, proposed, tokens)
        if return_trace:
            trace.append(tokens.cpu().tolist())
    return tokens, trace
