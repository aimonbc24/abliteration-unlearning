from transformers import LlamaForCausalLM
import torch


def load_hf_model(hf_path: str, torch_dtype: torch.dtype) -> LlamaForCausalLM:
    hf_model = LlamaForCausalLM.from_pretrained(hf_path)
    hf_model.to(torch_dtype)
    return hf_model

def truncate_model(hf_model: LlamaForCausalLM, vocab_size: int) -> LlamaForCausalLM:
    hidden_size = hf_model.config.hidden_size

    # truncate embedding matrix
    new_embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
    new_embed_tokens.weight.data = hf_model.model.embed_tokens.weight.data[:vocab_size, :]
    hf_model.model.embed_tokens = new_embed_tokens

    # truncate unembedding matrix
    new_lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
    new_lm_head.weight.data = hf_model.lm_head.weight.data[:vocab_size, :].clone()
    hf_model.lm_head = new_lm_head
    return hf_model