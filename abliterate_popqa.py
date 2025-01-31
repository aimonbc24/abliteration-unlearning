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
from datasets import load_dataset, Dataset
from tqdm import tqdm
from torch import Tensor
from typing import List
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer
from jaxtyping import Float, Int
import argparse
import ast
import random
import pandas as pd
import os
import sys

# from src.model.utils import load_hf_model, truncate_model

# Set constants
from constants import MAX_NEW_TOKENS
# DATASET_PATH = "data/PopQA/dataset_relevant_perturb.csv"

seed_value = 42
random.seed(seed_value)


def _generate_with_hooks(
    model: HookedTransformer,
    toks: Int[Tensor, 'batch_size seq_len'],
    max_tokens_generated: int = 64,
    fwd_hooks = [],
    skip_special_tokens: bool = True
) -> List[str]:

    all_toks = torch.zeros(
        (toks.shape[0], toks.shape[1] + max_tokens_generated), 
        dtype=torch.long, 
        device=toks.device
    ) # shape (batch_size, seq_len + max_tokens_generated)

    all_toks[:, :toks.shape[1]] = toks # copy input tokens to all_toks
    
    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_toks[:, :toks.shape[1] + i])

            # greedy sampling (temperature=0)
            next_tokens = logits[:, -1, :].argmax(dim=-1) 
            all_toks[:, toks.shape[1] + i] = next_tokens

            """
            # temperature sampling
            temperature = 0.7

            logits = logits[:, -1, :] / temperature  # Scale logits by temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)  # Convert logits to probabilities
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            all_toks[:, toks.shape[1] + i] = next_tokens"""

            if (next_tokens == model.tokenizer.eos_token_id).any():
                # if the model generates an end token, stop generating and decode only the generated tokens
                return model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:toks.shape[1]+i], skip_special_tokens=skip_special_tokens)
    
    return model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:], skip_special_tokens=skip_special_tokens)

def direction_ablation_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"]
):
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation - proj

def load_model(
    base_model_path = "meta-llama/Meta-Llama-3-8B-Instruct", 
    device = 'cuda', 
    dtype = torch.float16,
) -> HookedTransformer:

    print("Loading HookedTransformer...")
    model = HookedTransformer.from_pretrained(
        base_model_path, 
        # hf_model=hf_model, 
        # tokenizer=tokenizer, 
        torch_dtype=dtype
    ).to(device)

    return model


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run residual stream ablation experiments on a model.")
    argparser.add_argument("baseline_results_file", type=str, help="Path to the baseline results file.")
    argparser.add_argument("--dataset_name", type=str, help="Name of the dataset to use.")
    argparser.add_argument("--results_file", type=str, default=None, help="Path to save the results file.")
    argparser.add_argument("--perturbation_type", type=str, default="random", help="Type of perturbation to apply to the input. Options are 'random' or 'relevant'.")
    argparser.add_argument("--intervention_name", type=str, default="intervention", help="Name of the intervention column in the results csv")
    argparser.add_argument("--run_baseline", action="store_true", default=False, help="Save baseline completions (generated without hooks) to the results file")
    argparser.add_argument("--num_perturbed", type=int, default=1)
    argparser.add_argument("--pos", type=int, default=-1)
    argparser.add_argument("--layer", type=int, default=14, help="Layer in which to steer activations. Meta-Llama-3-8B has layers 0 - 31.")
    argparser.add_argument("--alpha", type=float, default=1.0, help="Alpha scaling value for the ablation direction.")
    argparser.add_argument("--debug", type=int, default=None, help="Run in debug mode. Specify number of samples to run.")
    argparser.add_argument("--verbose", action="store_true", default=False, help="Print verbose generation information.")
    args = argparser.parse_args()

    if "PopQA" in args.results_file:
        if args.perturbation_type == 'random':
            DATASET_PATH = "data/PopQA/dataset_random_perturb.csv"
        elif args.perturbation_type == 'relevant' and 'pop_search' in args.baseline_results_file:
            DATASET_PATH = "data/PopQA/dataset_relevant_perturbed_pop_search.csv"
        elif args.perturbation_type == 'relevant' and 'pop_search' not in args.baseline_results_file:
            DATASET_PATH = "data/PopQA/dataset_relevant_perturb.csv"
        else:
            print("Invalid perturbation type. Please choose 'random' or 'relevant'.")
            sys.exit(1)
    else:
        DATASET_PATH = f"data/TOFU/{args.dataset_name}.csv"
    print(f"\nDATASET_PATH: {DATASET_PATH}\n")

    # set the intervention name to 'baseline' if the run_baseline flag is set
    args.intervention_name = 'baseline' if args.run_baseline else args.intervention_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_perturbed = args.num_perturbed

    # print(f"\nRunning ablation experiments on layer {args.layer} with {args.num_perturbed} perturbations\n")

    # Load model
    model = load_model()

    # check if data/PopQA/data.csv exists in the file system. If so, load it.
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH).reset_index(drop=True)

        # Convert the 'perturbed_answer' and 'answer' columns from string representation to actual lists
        if type(df['perturbed_answer'][0]) == str:
            df['perturbed_answer'] = df['perturbed_answer'].apply(ast.literal_eval)
        if 'answer' in df.columns and type(df['answer'][0]) == str:
            df['answer'] = df['answer'].apply(ast.literal_eval)

        dataset = Dataset.from_pandas(df)
        print(f"\nLoaded data from {DATASET_PATH}\n")

    # otherwise, load the dataset from the Hugging Face hub and add perturbed answers to each sample, then save it to the file system
    else:
        # Load dataset
        print("\nLoading data...")
        dataset = load_dataset("akariasai/PopQA", split='test')

        # sort dataset by 'subject' popularity
        dataset = dataset.sort('s_pop', reverse=True)

        # If running an intervention, create a dictionary of possible answers for each property
        if not args.run_baseline:
            # create dictionary of possible answers for each property
            print("Creating dictionary of possible answers for each property...")
            answer_dict = {
                prop: list({item for sublist in [ast.literal_eval(li) for li in dataset.filter(lambda x: x['prop'] == prop)['possible_answers']] for item in sublist})
                for prop in set(dataset['prop'])
            }

            # add perturbed answers to dataset
            print(f"Adding {num_perturbed} perturbed answers to each sample...")
            dataset = dataset.map(
                lambda row: {
                    'perturbed_answer': [
                        ans for ans in answer_dict.get(row['prop'], [None] * num_perturbed) 
                        if ans not in ast.literal_eval(row['possible_answers'])
                    ]
                }
            )

            # save dataset to file system
            print(f"Saving data to {DATASET_PATH}...")
            pd.DataFrame(dataset).to_csv(DATASET_PATH, index=False)

    # read in data from baseline results file
    data = pd.read_csv(args.baseline_results_file)

    # define the fieldnames to save in the results file
    fieldnames = list(data.columns) + [args.intervention_name]
    fieldnames = list(set(fieldnames))

    # merge the dataset with the baseline results file
    data = pd.merge(left=pd.DataFrame(data).drop(columns=['answer']), right=pd.DataFrame(dataset).drop_duplicates('question'), on='question', how='left').to_dict(orient='records')

    num_samples = len(data) if (not args.debug or len(data) < args.debug) else args.debug

    print(f"\nRunning intervention: {args.intervention_name}")
    print(f"Output file: {args.results_file}")
    print(f"field names: {fieldnames}\n")
    print(f"Number of perturbed answers: {num_perturbed}\n")

    # run ablation experiments
    for i in tqdm(range(num_samples)):
        sample = data[i]

        chat = [
            {'role': 'system', 'content': 'You are a helpful AI assistant for pop-culture Question and Answering! Respond to the user\'s question succinctly.'},
            {'role': 'user', 'content': sample['question']},
        ]

        toks = model.tokenizer.apply_chat_template([chat], tokenize=True, return_tensors='pt').to(device)

        if args.run_baseline:
            baseline_generation = _generate_with_hooks(
                model,
                toks,
                max_tokens_generated=MAX_NEW_TOKENS,
            )
            data[i]['baseline'] = baseline_generation[0].split("assistant")[-1].strip()

            if args.verbose:
                print(f"\nQuestion: {sample['question']}")
                print(f"Answer: {sample['answer'][0] if type(sample['answer']) == list else sample['answer']}")
                print(f"Baseline: {data[i]['baseline']}\n")
            continue

        # 1. Get perturbed completions from cache or generate new completions
        perturbed_answers = random.sample(sample['perturbed_answer'], num_perturbed)

        if args.verbose:
            print(f"\nPerturbed answers: {perturbed_answers}\n")
        
        # 2. Format strings for model input
        sample_str = [
            {'role': 'system', 'content': 'You are a helpful AI assistant for pop-culture Question and Answering! Respond to the user\'s question succinctly.'},
            {'role': 'user', 'content': sample['question']},
            {'role': 'assistant', 'content': sample['answer'][0] if type(sample['answer']) == list else sample['answer']},
        ]

        perturbed_strs = [
            [
                {'role': 'system', 'content': 'You are a helpful AI assistant for pop-culture Question and Answering! Respond to the user\'s question succinctly.'},
                {'role': 'user', 'content': sample['question']},
                {'role': 'assistant', 'content': answer},
            ] for answer in perturbed_answers
        ]
        
        model.eval()
        pos = args.pos
        layer = args.layer

        # 3a. Run model on original QnA with hooks, saving activations
        sample_toks = model.tokenizer.apply_chat_template([sample_str], tokenize=True, return_tensors='pt').to(device)
        harmful_logits, harmful_cache = model.run_with_cache(sample_toks, names_filter=lambda hook_name: 'resid' in hook_name)
        # print(f"positions: {harmful_cache['resid_pre', layer].shape}")
        harmful_mean_act = harmful_cache['resid_pre', layer][:, pos, :].mean(dim=0)
        # print(f"Mean activation for harmful example shape: {harmful_mean_act.shape}")

        # 3b. Run model on perturbed QnA one at a time with hooks, saving and stacking activations
        perturbed_mean_act = torch.zeros_like(harmful_mean_act)
        for perturbed_str in perturbed_strs:
            # print(perturbed_str)
            perturbed_toks = model.tokenizer.apply_chat_template([perturbed_str], tokenize=True, return_tensors='pt').to(device)
            _, perturbed_cache = model.run_with_cache(perturbed_toks, names_filter=lambda hook_name: 'resid' in hook_name)
            perturbed_mean_act += perturbed_cache['resid_pre', layer][:, pos, :].mean(dim=0)
        perturbed_mean_act /= (num_perturbed + 1)

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
        data[i][args.intervention_name] = intervention_generation[0].split("assistant")[1].strip()

        if args.verbose:
            print(f"\nQuestion: {sample['question']}")
            print(f"Answer: {sample['answer'][0] if type(sample['answer']) == list else sample['answer']}")
            print(f"Intervention: {data[i][args.intervention_name]}\n")


    df = pd.DataFrame(data)

    # calculate intervention accuracy
    intervention_accuracy = df.dropna().apply(lambda row: int(any(answer.strip().lower() in row[args.intervention_name].strip().lower() for answer in row['answer'])), axis=1).mean()
    print(f"\n\nIntervention accuracy: {intervention_accuracy}\n")
    
    # filter out any columns that are not in fieldnames
    results = df[fieldnames].to_dict(orient='records')

    print(f"Saving results to {args.results_file}\n\n")
    with open(args.results_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
