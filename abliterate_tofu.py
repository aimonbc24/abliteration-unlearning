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
import pandas as pd
import ast

from src.model.utils import load_hf_model, truncate_model

# Set constants
MAX_NEW_TOKENS = 50


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

    model_path = finetune_model_path if finetune_model_path else base_model_path

    print(f"Loading HuggingFace model {model_path}...")
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
    argparser.add_argument("--finetune_model_path", type=str, default=None, help="Path to the finetuned model. Defaults to the 1-epoch tuned model due to weird behavior with the base model.")
    argparser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset to use. Choices include: 'full', 'forget01', 'forget05', 'forget10', 'retain90', 'retain95', 'retain99', 'world_facts', 'real_authors', 'forget01_perturbed', 'forget05_perturbed', 'forget10_perturbed', 'retain_perturbed', 'world_facts_perturbed', 'real_authors_perturbed'.")
    argparser.add_argument("--dataset_path", type=str, default=None, help="Local path to the dataset to use.")
    argparser.add_argument("--intervention_name", type=str, default="intervention", help="Name of the intervention column in the results csv")
    argparser.add_argument("--run_baseline", action="store_true", default=False, help="Save baseline completions (generated without hooks) to the results file")
    argparser.add_argument("--num_perturbed", type=int, default=1)
    argparser.add_argument("--pos", type=int, default=-1)
    argparser.add_argument("--layer", type=int, default=14, help="Layer in which to steer activations. Meta-Llama-3-8B has layers 0 - 31.")
    argparser.add_argument("--alpha", type=float, default=1.0, help="Alpha scaling value for the ablation direction.")
    argparser.add_argument("--debug", type=int, default=None, help="Run in debug mode. If set to a number, will run on that many samples.")
    argparser.add_argument("--verbose", action="store_true", default=False, help="Print out the question, answer, and intervention for each sample")
    argparser.add_argument("--use_chat_template", action="store_true", default=False, help="Use chat template for intervention generation")
    args = argparser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.intervention_name = 'baseline' if args.run_baseline else args.intervention_name

    # print(f"\nRunning ablation experiments on layer {args.layer} with {args.num_perturbed} perturbations\n")

    # Load model
    model = load_model(
        finetune_model_path=args.finetune_model_path,
        device=device
    )

    # Load dataset
    print("\nLoading data...")

    if args.dataset_name:
        dataset = load_dataset("locuslab/TOFU", name=args.dataset_name)['train']
        dataset = dataset.to_pandas()
    elif args.dataset_path:
        dataset = pd.read_csv(args.dataset_path)
    else:
        raise ValueError("Must provide either a dataset name or path.")
    
    # convert string representations of lists to lists
    if 'perturbed_answer' in dataset.columns and type(dataset['perturbed_answer'][0]) == str:
        dataset['perturbed_answer'] = dataset['perturbed_answer'].apply(ast.literal_eval)
    if 'answer' in dataset.columns and type(dataset['answer'][0]) == str:
        dataset['answer'] = dataset['answer'].apply(ast.literal_eval)
    if 'index' in dataset.columns:
        dataset = dataset.drop(columns=['index'])

    # Load baseline results file
    data = pd.read_csv(args.baseline_results_file)

    # define the fieldnames to save in the results file
    fieldnames = list(data.columns) + [args.intervention_name]
    fieldnames = list(set(fieldnames))

    # merge the dataset with the baseline results file
    data = pd.merge(data.drop(columns=['answer']), dataset.drop_duplicates('question'), on='question', how='left').to_dict(orient='records')      

    num_perturbed = args.num_perturbed
    num_samples = len(data) if (not args.debug or len(data) < args.debug) else args.debug

    print(f"\nFieldnames: {fieldnames}\n")
    print(f"Columns: {data[0].keys()}\n")

    # run ablation experiments
    for i in tqdm(range(num_samples)):
        sample = data[i]

        if args.use_chat_template:
            chat = [
                {'role': 'system', 'content': 'You are a helpful AI assistant for pop-culture Question and Answering! Respond to the user\'s question succinctly.'},
                {'role': 'user', 'content': sample['question']},
            ]

            toks = model.tokenizer.apply_chat_template([chat], tokenize=True, return_tensors='pt').to(device)
        
        else:
            question_str = f"Prompt: {sample['question']}\nCompletion: "
            toks = model.tokenizer([question_str], return_tensors="pt", padding=True)['input_ids'].to(device)

        if args.run_baseline:
            baseline_generation = _generate_with_hooks(
                model,
                toks,
                max_tokens_generated=MAX_NEW_TOKENS,
            )
            data[i]['baseline'] = baseline_generation[0].strip()

            if args.verbose:
                print(f"\nQuestion: {sample['question']}")
                print(f"Answer: {sample['answer'][0] if type(sample['answer']) == list else sample['answer']}")
                print(f"Baseline: {data[i]['baseline']}\n")

            continue

        # 1. Get perturbed completions from cache or generate new completions
        perturbed_answers = sample['perturbed_answer'][:num_perturbed]

        if args.use_chat_template:
            sample_str = [
                # {'role': 'system', 'content': 'You are a helpful AI assistant for pop-culture Question and Answering! Respond to the user\'s question succinctly.'},
                {'role': 'user', 'content': sample['question']},
                {'role': 'assistant', 'content': sample['answer'][0] if type(sample['answer']) == list else sample['answer']},
            ]

            perturbed_strs = [
                [
                    # {'role': 'system', 'content': 'You are a helpful AI assistant for pop-culture Question and Answering! Respond to the user\'s question succinctly.'},
                    {'role': 'user', 'content': sample['question']},
                    {'role': 'assistant', 'content': answer},
                ] for answer in perturbed_answers
            ]

            sample_toks = model.tokenizer.apply_chat_template([sample_str], tokenize=True, return_tensors='pt').to(device)
        
        else:
            # format strings for model input
            perturbed_strs = [f"Prompt: {sample['question']}\nCompletion: {answer}" for answer in perturbed_answers]
            sample_str = [f"Prompt: {sample['question']}\nCompletion: {sample['answer']}"]

            sample_toks = model.tokenizer(sample_str, return_tensors="pt", padding=True)['input_ids'].to(device)

        # if args.verbose:
        #     print(perturbed_strs[0])

        model.eval()
        pos = args.pos
        layer = args.layer

        # 3a. Run model on original QnA with hooks, saving activations
        harmful_logits, harmful_cache = model.run_with_cache(sample_toks, names_filter=lambda hook_name: 'resid' in hook_name)
        # print(f"positions: {harmful_cache['resid_pre', layer].shape}")
        harmful_mean_act = harmful_cache['resid_pre', layer][:, pos, :].mean(dim=0)
        # print(f"Mean activation for harmful example shape: {harmful_mean_act.shape}")

        # 3b. Run model on perturbed QnA one at a time with hooks, saving and stacking activations
        perturbed_mean_act = torch.zeros_like(harmful_mean_act)
        for perturbed_str in perturbed_strs:
            if args.use_chat_template:
                perturbed_toks = model.tokenizer.apply_chat_template([perturbed_str], tokenize=True, return_tensors='pt').to(device)
            else:
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

        if args.verbose:
            print(f"\nQuestion: {sample['question']}")
            print(f"Perturbed answers: {perturbed_answers}")
            print(f"Answer: {sample['answer'][0] if type(sample['answer']) == list else sample['answer']}")
            print(f"Intervention: {data[i][args.intervention_name]}\n")

    if args.results_file is not None:
        output_file = args.results_file
    else:
        # save results to csv
        if "world_facts" not in args.dataset_name and "real_authors" not in args.dataset_name:
            output_file = f"results/fictional_authors/{args.dataset_name}/unlearning_results_{args.finetune_model_path.split('/')[-1]}.csv"
        else:
            output_file = "results/world_facts/unlearning_results.csv"

    print(f"Saving results to {output_file}")

    df = pd.DataFrame(data)

    # calculate intervention accuracy
    if type(df['answer'][0]) == list:
        intervention_accuracy = df.dropna().apply(lambda row: int(any(answer.strip().lower() in row[args.intervention_name].strip().lower() for answer in row['answer'])), axis=1).mean()
    else:
        intervention_accuracy = df.dropna().apply(lambda row: int(row['answer'].strip().lower() in row[args.intervention_name].strip().lower()), axis=1).mean()
    
    print(f"\n\nIntervention accuracy: {intervention_accuracy}\n")
    
    # filter out any columns that are not in fieldnames
    results = df[fieldnames].to_dict(orient='records')
    
    with open(output_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    