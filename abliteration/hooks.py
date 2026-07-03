"""Shared abliteration primitives.

The ``abliterate_*.py`` scripts previously copy-pasted these helpers (with minor
drift). They are consolidated here — the canonical versions being the ones with
EOS-stopping generation and fine-tuned-model support.

Directional ablation ("abliteration"): find a direction ``d`` in the residual
stream that encodes a fact, then subtract it at inference so the model can no
longer produce it: ``a <- a - (a . d) d``.
"""
import functools
from typing import Iterable, List, Optional, Sequence

import einops
import torch
from torch import Tensor
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint


def load_model(base_model_path: str = "meta-llama/Meta-Llama-3-8B-Instruct",
               finetune_model_path: Optional[str] = None,
               device: str = "cuda",
               dtype: torch.dtype = torch.float16,
               vocab_size: int = 128256) -> HookedTransformer:
    """Load a model into TransformerLens.

    If ``finetune_model_path`` is given, its Hugging Face weights are loaded
    (truncated to ``vocab_size``) into the base architecture; otherwise the base
    model is loaded directly.
    """
    from transformers import AutoTokenizer
    from src.model.utils import load_hf_model, truncate_model

    model_path = finetune_model_path or base_model_path
    hf_model = load_hf_model(model_path, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if finetune_model_path is not None:
        hf_model = truncate_model(hf_model, vocab_size)
    hf_model.to(device)
    model = HookedTransformer.from_pretrained(
        base_model_path, hf_model=hf_model, tokenizer=tokenizer, torch_dtype=dtype)
    return model.to(device)


def direction_ablation_hook(activation: Float[Tensor, "... d_act"],
                            hook: Optional[HookPoint],
                            direction: Float[Tensor, "d_act"]) -> Tensor:
    """Project ``activation`` onto ``direction`` and subtract it out."""
    proj = einops.einsum(activation, direction.view(-1, 1),
                         "... d_act, d_act single -> ... single") * direction
    return activation - proj


def ablation_hooks(model: HookedTransformer, direction: Tensor):
    """Forward hooks that ablate ``direction`` from resid_pre/mid/post of every layer."""
    hook_fn = functools.partial(direction_ablation_hook, direction=direction)
    return [(utils.get_act_name(act, layer), hook_fn)
            for layer in range(model.cfg.n_layers)
            for act in ("resid_pre", "resid_mid", "resid_post")]


def generate_with_hooks(model: HookedTransformer,
                        toks: Int[Tensor, "batch seq"],
                        max_tokens_generated: int = 64,
                        fwd_hooks: Iterable = (),
                        skip_special_tokens: bool = True) -> List[str]:
    """Greedy (temperature-0) generation under ``fwd_hooks``, stopping at EOS."""
    all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated),
                           dtype=torch.long, device=toks.device)
    all_toks[:, :toks.shape[1]] = toks
    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=list(fwd_hooks)):
            logits = model(all_toks[:, :toks.shape[1] + i])
            next_tokens = logits[:, -1, :].argmax(dim=-1)  # greedy
            all_toks[:, toks.shape[1] + i] = next_tokens
            if (next_tokens == model.tokenizer.eos_token_id).any():
                return model.tokenizer.batch_decode(
                    all_toks[:, toks.shape[1]:toks.shape[1] + i],
                    skip_special_tokens=skip_special_tokens)
    return model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:],
                                        skip_special_tokens=skip_special_tokens)


def compute_forget_direction(model: HookedTransformer,
                             harmful_prompts: Sequence[str],
                             perturbed_prompts: Sequence[str],
                             layer: int, pos: int = -1) -> Tensor:
    """Unit forget direction = normalize(mean(harmful act) - mean(perturbed act))
    at (``layer``, ``pos``) of the residual stream (``resid_pre``)."""
    def mean_act(prompts: Sequence[str]) -> Tensor:
        toks = model.tokenizer(list(prompts), return_tensors="pt",
                               padding=True)["input_ids"].to(model.cfg.device)
        _, cache = model.run_with_cache(toks, names_filter=lambda h: "resid" in h)
        return cache["resid_pre", layer][:, pos, :].mean(dim=0)

    direction = mean_act(harmful_prompts) - mean_act(perturbed_prompts)
    return direction / direction.norm()
