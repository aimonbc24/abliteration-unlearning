import torch
"""
1. Get perturbed completions
2. Tokenize instructions
3. Run model on instruction pairs with hooks, saving activations
4. Compute mean activation difference between pairs
5. Generate completions for instructions
"""

import functools
import einops
import csv
from datasets import load_dataset
from tqdm import tqdm
from torch import Tensor
from typing import List
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer
from jaxtyping import Float, Int
import argparse

from utils import load_hf_model, truncate_model

def _generate_with_hooks(
    model: HookedTransformer,
    toks: Int[Tensor, 'batch_size seq_len'],
    max_tokens_generated: int = 64,
    fwd_hooks = [],
) -> List[str]:

    all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated), dtype=torch.long, device=toks.device)
    all_toks[:, :toks.shape[1]] = toks

    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_toks[:, :-max_tokens_generated + i])
            next_tokens = logits[:, -1, :].argmax(dim=-1) # greedy sampling (temperature=0)
            all_toks[:,-max_tokens_generated+i] = next_tokens

    return model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:], skip_special_tokens=True)

def direction_ablation_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"]
):
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation - proj

def load_model(
    finetune_model_path, 
    base_model_path = "meta-llama/Meta-Llama-3-8B-Instruct", 
    device = 'cuda', 
    vocab_size = 128256, 
    dtype = torch.float16
) -> HookedTransformer:
    
    if finetune_model_path is not None:    
        # Load model
        print("Loading Finetuned model into HookedTransformer...")
        hf_model = load_hf_model(finetune_model_path, dtype)
        tokenizer = AutoTokenizer.from_pretrained(finetune_model_path)
        hf_model = truncate_model(hf_model, vocab_size)

        hf_model.to(device)

        model = HookedTransformer.from_pretrained(
            base_model_path, 
            hf_model=hf_model, 
            tokenizer=tokenizer, 
            torch_dtype=torch.float16
        ).to(device)
    
    else:
        print("Loading base model into HookedTransformer...")
        model = HookedTransformer.from_pretrained(
            base_model_path, 
            torch_dtype=torch.float16
        ).to(device)

    return model


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run residual stream ablation experiments on a model.")
    argparser.add_argument("--baseline_results_file", type=str, default="results/finetune_results.csv", help="Path to the baseline results file.")
    argparser.add_argument("--finetune_model_path", type=str, default=None, help="Path to the finetuned model")
    argparser.add_argument("--dataset_name", type=str, default="full", help="Name of the dataset to use. Choices include: 'full', 'forget01', 'forget05', 'forget10', 'retain90', 'retain95', 'retain99', 'world_facts', 'real_authors', 'forget01_perturbed', 'forget05_perturbed', 'forget10_perturbed', 'retain_perturbed', 'world_facts_perturbed', 'real_authors_perturbed'.")
    argparser.add_argument("--intervention_name", type=str, default="intervention", help="Name of the intervention column in the results csv")
    argparser.add_argument("--num_alternatives", type=int, default=1)
    argparser.add_argument("--pos", type=int, default=-1)
    argparser.add_argument("--layer", type=int, default=14, help="Layer in which to steer activations. Meta-Llama-3-8B has layers 0 - 31.")
    argparser.add_argument("--debug", action="store_true", default=False, help="Run in debug mode")
    args = argparser.parse_args()

    print(f"Running ablation experiments with {args.num_alternatives} alternatives per question using layer {args.layer} at position {args.pos}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(
        finetune_model_path=args.finetune_model_path,
        device=device
    )

    # Load dataset
    print("Loading data...")
    dataset = load_dataset("locuslab/TOFU", name=args.dataset_name)['train']

    # read in data from finetune_results.csv
    with open(args.baseline_results_file, "r") as f_data:
        reader = csv.DictReader(f_data)
        data = list(reader)

        fieldnames = ['question', 'answer', 'baseline'] + [args.intervention_name]

        # if a finetuned model path is provided, include a column for the finetune model results
        if args.finetune_model_path is not None:
            fieldnames += [f'epoch-{args.finetune_model_path[-1]}']

    num_alternatives = args.num_alternatives
    num_samples = len(dataset) if not args.debug else 10

    # run ablation experiments
    for i in tqdm(range(num_samples)):
        sample = dataset[i]

        # 1. Get perturbed completions from cache or generate new completions
        perturbed_answers = sample['perturbed_answer'][:num_alternatives]
        
        # format strings for model input
        alternative_strs = [f"Prompt: {sample['question']}\nCompletion: {answer}" for answer in perturbed_answers]
        sample_str = [f"Prompt: {sample['question']}\nCompletion: {sample['answer']}"]

        # 2. Tokenize instructions
        model.eval()
        sample_toks = model.tokenizer(sample_str, return_tensors="pt", padding=True)['input_ids'].to(device)
        alternative_toks = model.tokenizer(alternative_strs, return_tensors="pt", padding=True)['input_ids'].to(device)

        # print(f"Tokenized instructions. Token shapes: {sample_toks.shape}, {alternative_toks.shape}")

        # 3. Run model on instruction pairs with hooks, saving activations
        harmful_logits, harmful_cache = model.run_with_cache(sample_toks, names_filter=lambda hook_name: 'resid' in hook_name)
        alternative_logits, alternative_cache = model.run_with_cache(alternative_toks, names_filter=lambda hook_name: 'resid' in hook_name)

        # 4. Compute mean activation difference between pairs
        pos = args.pos
        layer = args.layer
        harmful_mean_act = harmful_cache['resid_pre', layer][:, pos, :].mean(dim=0)
        harmless_mean_act = alternative_cache['resid_pre', layer][:, pos, :].mean(dim=0)

        intervention_dir = harmful_mean_act - harmless_mean_act
        intervention_dir = intervention_dir / intervention_dir.norm()

        # print(f"Mean activation difference: {intervention_dir}")

        # 5. Generate completions for instructions
        intervention_layers = list(range(model.cfg.n_layers)) # all layers
        # print(intervention_layers)
        hook_fn = functools.partial(direction_ablation_hook,direction=intervention_dir)
        fwd_hooks = [(utils.get_act_name(act_name, l), hook_fn) for l in intervention_layers for act_name in ['resid_pre', 'resid_mid', 'resid_post']]

        question_str = [f"Prompt: {sample['question']}\nCompletion: "]
        toks = model.tokenizer(question_str, return_tensors="pt", padding=True)['input_ids'].to(device)

        max_new_tokens = 150
        intervention_generation = _generate_with_hooks(
            model,
            toks,
            max_tokens_generated=max_new_tokens,
            fwd_hooks=fwd_hooks,
        )

        data[i][f'intervention'] = intervention_generation[0]


    # save results to csv

    if "world_facts" in args.dataset_name:
        output_file = "results/unlearning_world_facts_results.csv"
    else:
        output_file = f"results/unlearning_results_{args.finetune_model_path.split('/')[-1]}.csv"
    with open(output_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    
