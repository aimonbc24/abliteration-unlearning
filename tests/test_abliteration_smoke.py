"""CPU smoke test for the abliteration primitives, using gpt2 via TransformerLens.

Exercises the exact shared code the abliterate_*.py scripts use, without needing
a GPU or gated Llama-3 weights.
"""
import pytest
import torch
from transformer_lens import HookedTransformer

from abliteration.hooks import (
    direction_ablation_hook,
    ablation_hooks,
    generate_with_hooks,
    compute_forget_direction,
)


@pytest.fixture(scope="module")
def gpt2():
    return HookedTransformer.from_pretrained("gpt2", device="cpu")


def test_direction_ablation_removes_the_component():
    torch.manual_seed(0)
    act = torch.randn(2, 3, 8)
    d = torch.randn(8)
    d = d / d.norm()
    out = direction_ablation_hook(act, hook=None, direction=d)
    # After ablation the activation's component along d is ~0.
    assert (out @ d).abs().max() < 1e-5


def test_compute_forget_direction_is_unit_vector(gpt2):
    d = compute_forget_direction(
        gpt2,
        harmful_prompts=["The capital of France is Paris"],
        perturbed_prompts=["The capital of France is Berlin"],
        layer=5,
    )
    assert d.shape == (gpt2.cfg.d_model,)
    assert abs(d.norm().item() - 1.0) < 1e-4


def test_generation_runs_with_and_without_ablation(gpt2):
    toks = gpt2.to_tokens("The capital of France is")
    base = generate_with_hooks(gpt2, toks, max_tokens_generated=5)
    d = compute_forget_direction(
        gpt2,
        harmful_prompts=["The capital of France is Paris"],
        perturbed_prompts=["The capital of France is Berlin"],
        layer=5,
    )
    hooked = generate_with_hooks(
        gpt2, toks, max_tokens_generated=5, fwd_hooks=ablation_hooks(gpt2, d))
    # Both paths produce decoded strings end-to-end.
    assert isinstance(base, list) and isinstance(hooked, list)
    assert isinstance(base[0], str) and isinstance(hooked[0], str)
