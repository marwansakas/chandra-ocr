import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesis.diffusion.decoding_diagnostics import greedy_decode
from thesis.diffusion.model import DiffusionRecognizer, ModelConfig


def model_and_inputs():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = DiffusionRecognizer(ModelConfig(10, 8, d_model=16, heads=4, encoder_layers=1,
                                            decoder_layers=1, patch_size=4, dropout=0)).eval()
    return model, torch.randn(2, 3, 16, 24), torch.tensor([24, 20])


def test_one_pass_equals_full_mask_argmax():
    model, pixels, widths = model_and_inputs()
    with torch.no_grad():
        memory, mask = model.encoder(pixels, widths)
        expected = model.denoise(torch.ones(2, 8, dtype=torch.long), memory, mask, torch.ones(2)).argmax(-1)
    actual, trace = greedy_decode(model, pixels, widths, method="one_pass", return_trace=True)
    assert torch.equal(actual, expected)
    assert len(trace) == 2 and set(trace[0][0]) == {1}


@pytest.mark.parametrize("method", ["random_reveal", "confidence"])
def test_multistep_completion_and_committed_tokens(method):
    model, pixels, widths = model_and_inputs()
    actual, trace = greedy_decode(model, pixels, widths, method=method, steps=16,
                                   generator=torch.Generator().manual_seed(8), return_trace=True)
    assert len(trace) == 17 and actual.min() >= 2
    for previous, current in zip(trace, trace[1:]):
        previous, current = torch.tensor(previous), torch.tensor(current)
        assert torch.equal(previous[previous != 1], current[previous != 1])
    if method == "confidence":
        other, _ = greedy_decode(model, pixels, widths, method=method, steps=16,
                                  generator=torch.Generator().manual_seed(999))
        assert torch.equal(actual, other)


def test_invalid_decoding_is_explicit():
    model, pixels, widths = model_and_inputs()
    with pytest.raises(ValueError):
        greedy_decode(model, pixels, widths, method="unknown")
    model.train()
    with pytest.raises(ValueError):
        greedy_decode(model, pixels, widths, method="one_pass")
