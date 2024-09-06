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

from src.model.utils import load_hf_model, truncate_model

# Set constants
MAX_NEW_TOKENS = 50
NUM_DEBUG_SAMPLES = 15

def _generate_with_hooks(
    model: HookedTransformer,
    toks: Int[Tensor, 'batch_size seq_len'],
    max_tokens_generated: int = 64,
    fwd_hooks = [],
) -> List[str]:

    all_toks = torch.zeros(
        (toks.shape[0], toks.shape[1] + max_tokens_generated), 
        dtype=torch.long, 
        device=toks.device
    ) # shape (batch_size, seq_len + max_tokens_generated)

    all_toks[:, :toks.shape[1]] = toks # copy input tokens to all_toks

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
    finetune_model_path = None, 
    base_model_path = "meta-llama/Meta-Llama-3-8B-Instruct", 
    device = 'cuda', 
    vocab_size = 128256, 
    dtype = torch.float16
) -> HookedTransformer:

    model_path = finetune_model_path if finetune_model_path is not None else base_model_path

    print("Loading HuggingFace model...")
    hf_model = load_hf_model(model_path, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if finetune_model_path is not None:
        print("Truncating model...")
        hf_model = truncate_model(hf_model, vocab_size)

    hf_model.to(device)

    print("Loading HookedTransformer...")
    model = HookedTransformer.from_pretrained(
        base_model_path, 
        hf_model=hf_model, 
        tokenizer=tokenizer, 
        torch_dtype=torch.float16
    ).to(device)

    return model


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run residual stream ablation experiments on a model.")
    argparser.add_argument("baseline_results_file", type=str, help="Path to the baseline results file.")
    argparser.add_argument("--results_file", type=str, default=None, help="Path to save the results file.")
    argparser.add_argument("--finetune_model_path", type=str, default="aimonbc/llama3-tofu-8B-epoch-0", help="Path to the finetuned model. Defaults to the 1-epoch tuned model due to weird behavior with the base model.")
    argparser.add_argument("--dataset_name", type=str, default="full", help="Name of the dataset to use. Choices include: 'full', 'forget01', 'forget05', 'forget10', 'retain90', 'retain95', 'retain99', 'world_facts', 'real_authors', 'forget01_perturbed', 'forget05_perturbed', 'forget10_perturbed', 'retain_perturbed', 'world_facts_perturbed', 'real_authors_perturbed'.")
    argparser.add_argument("--intervention_name", type=str, default="intervention", help="Name of the intervention column in the results csv")
    argparser.add_argument("--run_baseline", action="store_true", default=False, help="Save baseline completions (generated without hooks) to the results file")
    argparser.add_argument("--num_perturbed", type=int, default=1)
    argparser.add_argument("--pos", type=int, default=-1)
    argparser.add_argument("--layer", type=int, default=14, help="Layer in which to steer activations. Meta-Llama-3-8B has layers 0 - 31.")
    argparser.add_argument("--alpha", type=float, default=1.0, help="Alpha scaling value for the ablation direction.")
    argparser.add_argument("--debug", action="store_true", default=False, help="Run in debug mode")
    args = argparser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nRunning ablation experiments on layer {args.layer} with {args.num_perturbed} perturbations\n")

    # Load model
    model = load_model(
        finetune_model_path=args.finetune_model_path,
        device=device
    )

    # Load dataset
    print("\nLoading data...")
    dataset = load_dataset("locuslab/TOFU", name=args.dataset_name)['train']        

    # read in data from baseline results file
    with open(args.baseline_results_file, "r") as f_data:
        reader = csv.DictReader(f_data)
        data = list(reader)

        # keep fieldnames that do not begin with 'epoch-'
        fieldnames = [field for field in reader.fieldnames if not field.startswith('epoch-')]

        # add intervention and baseline fieldnames
        fieldnames += ['baseline']
        fieldnames += [args.intervention_name] if not args.run_baseline else []
        fieldnames = list(set(fieldnames))

        # only keep the epoch field for the corresponding modle if the fictional authors dataset is used
        if 'world_facts' not in args.dataset_name:
            fieldnames += [f'epoch-{args.finetune_model_path[-1]}']

    # filter the dataset to only include questions in the baseline results file
    dataset = [sample for sample in dataset if any(sample['question'] in row['question'] for row in data)]
    print(f"Filtered dataset size: {len(dataset)} samples.")

    num_perturbed = args.num_perturbed
    num_samples = len(dataset) if (not args.debug or len(dataset) < NUM_DEBUG_SAMPLES) else NUM_DEBUG_SAMPLES

    print(fieldnames)

    # run ablation experiments
    for i in tqdm(range(num_samples)):
        sample = dataset[i]

        question_str = f"Prompt: {sample['question']}\nCompletion: "
        toks = model.tokenizer([question_str], return_tensors="pt", padding=True)['input_ids'].to(device)

        if args.run_baseline:
            baseline_generation = _generate_with_hooks(
                model,
                toks,
                max_tokens_generated=MAX_NEW_TOKENS,
            )
            data[i]['baseline'] = baseline_generation[0].strip()
            continue

        # 1. Get perturbed completions from cache or generate new completions
        perturbed_answers = sample['perturbed_answer'][:num_perturbed]
        
        # format strings for model input
        perturbed_strs = [f"Prompt: {sample['question']}\nCompletion: {answer}" for answer in perturbed_answers]
        sample_str = [f"Prompt: {sample['question']}\nCompletion: {sample['answer']}"]

        model.eval()
        pos = args.pos
        layer = args.layer

        # 3a. Run model on original QnA with hooks, saving activations
        sample_toks = model.tokenizer(sample_str, return_tensors="pt", padding=True)['input_ids'].to(device)
        harmful_logits, harmful_cache = model.run_with_cache(sample_toks, names_filter=lambda hook_name: 'resid' in hook_name)
        harmful_mean_act = harmful_cache['resid_pre', layer][:, pos, :].mean(dim=0)
        # print(f"Mean activation for harmful example shape: {harmful_mean_act.shape}")

        # 3b. Run model on perturbed QnA one at a time with hooks, saving and stacking activations
        perturbed_mean_act = torch.zeros_like(harmful_mean_act)
        for perturbed_str in perturbed_strs:
            perturbed_toks = model.tokenizer([perturbed_str], return_tensors="pt", padding=True)['input_ids'].to(device)
            _, perturbed_cache = model.run_with_cache(perturbed_toks, names_filter=lambda hook_name: 'resid' in hook_name)
            perturbed_mean_act += perturbed_cache['resid_pre', layer][:, pos, :].mean(dim=0)
        perturbed_mean_act /= num_perturbed

        # print(f"Tokenized instructions. Token shapes: {sample_toks.shape}, {alternative_toks.shape}")

        # 4. Compute mean activation difference between pairs
        intervention_dir = harmful_mean_act - perturbed_mean_act
        intervention_dir = intervention_dir / intervention_dir.norm() * args.alpha

        # 5. Generate completions for instructions
        intervention_layers = list(range(model.cfg.n_layers)) # all layers
        hook_fn = functools.partial(direction_ablation_hook,direction=intervention_dir)
        fwd_hooks = [(utils.get_act_name(act_name, l), hook_fn) for l in intervention_layers for act_name in ['resid_pre', 'resid_mid', 'resid_post']]

        intervention_generation = _generate_with_hooks(
            model,
            toks,
            max_tokens_generated=MAX_NEW_TOKENS,
            fwd_hooks=fwd_hooks,
        )
        data[i][args.intervention_name] = intervention_generation[0].strip()

        # print(data[i])

    if args.results_file is not None:
        output_file = args.results_file
    else:
        # save results to csv
        if "world_facts" not in args.dataset_name and "real_authors" not in args.dataset_name:
            output_file = f"results/fictional_authors/{args.dataset_name}/unlearning_results_{args.finetune_model_path.split('/')[-1]}.csv"
        else:
            output_file = "results/world_facts/unlearning_results.csv"

    print(f"Saving results to {output_file}")
    with open(output_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
