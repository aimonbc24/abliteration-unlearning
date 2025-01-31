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

from src.model.utils import load_hf_model, truncate_model
from constants import MAX_NEW_TOKENS

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
    base_model_path = None, 
    device = 'cuda', 
    vocab_size = 128256, 
    dtype = torch.float16
) -> HookedTransformer:

    model_path = finetune_model_path if finetune_model_path else base_model_path

    print(f"\nLoading HuggingFace model {model_path}...\n")
    hf_model = load_hf_model(model_path, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if finetune_model_path is not None:
        print("Truncating model...")
        hf_model = truncate_model(hf_model, vocab_size)

    hf_model.to(device)

    print("\nLoading HookedTransformer...\n")
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
    argparser.add_argument("--results_file", type=str, required=True, help="Path to save the results file.")
    argparser.add_argument("--finetune_model_path", type=str, default=None, help="Path to the finetuned model. Defaults to the pretrained base model.")
    argparser.add_argument("--base_model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="ID of the base model to use with TransformerLens. Options can be found at https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html")
    # argparser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset to use. Choices include: 'full', 'forget01', 'forget05', 'forget10', 'retain90', 'retain95', 'retain99', 'world_facts', 'real_authors', 'forget01_perturbed', 'forget05_perturbed', 'forget10_perturbed', 'retain_perturbed', 'world_facts_perturbed', 'real_authors_perturbed'.")
    argparser.add_argument("--dataset_path", type=str, default="data/topic_qa_perturbed.csv", help="Local path to the dataset to use.")
    argparser.add_argument("--intervention_name", type=str, default="intervention", help="Name of the intervention column in the results csv")
    argparser.add_argument("--run_baseline", action="store_true", default=False, help="Save baseline completions (generated without hooks) to the results file")
    argparser.add_argument("--num_perturbed", type=int, default=1)
    argparser.add_argument("--pos", type=int, default=-1)
    argparser.add_argument("--layer", type=int, default=14, help="Layer in which to steer activations. Meta-Llama-3-8B has layers 0 - 31.")
    argparser.add_argument("--alpha", type=float, default=1.0, help="Alpha scaling value for the ablation direction.")
    argparser.add_argument("--num_train", type=int, default=1, help="Number of samples to use in calculating the intervention direction for a given entity.")
    argparser.add_argument("--num_test", type=int, default=None, help="Number of samples to evaluate the intervention on for a given entity. Defaults to the remaining samples.")
    argparser.add_argument("--debug", type=int, default=None, help="Run in debug mode. If set to a number, will run on that many samples.")
    argparser.add_argument("--verbose", action="store_true", default=False, help="Print out the question, answer, and intervention for each sample")
    argparser.add_argument("--use_chat_template", action="store_true", default=False, help="Use chat template for intervention generation. Equivalent to using --inference_chat_template and --intervention_chat_template.")
    argparser.add_argument("--inference_chat_template", action="store_true", default=False, help="Use chat template for inference generation")
    args = argparser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.intervention_name = 'baseline' if args.run_baseline else args.intervention_name

    # print(f"\nRunning ablation experiments on layer {args.layer} with {args.num_perturbed} perturbations\n")

    # Load model
    model = load_model(
        finetune_model_path=args.finetune_model_path,
        base_model_path=args.base_model_id,
        device=device
    )

    # read in data from baseline results file
    data = pd.read_csv(args.baseline_results_file)

    # define the fieldnames to save in the results file
    fieldnames = list(data.columns) + [args.intervention_name]
    fieldnames = list(set(fieldnames))

    dataset = pd.read_csv(args.dataset_path)

    if 'perturbed_answer' in dataset.columns and dataset["perturbed_answer"][0][0] == '[':
        print("\nConverting perturbed answers to list...\n")
        dataset['perturbed_answer'] = dataset['perturbed_answer'].apply(ast.literal_eval)
    else:
        print("\nConverting perturbed answers to list...\n")
        dataset['perturbed_answer'] = dataset['perturbed_answer'].apply(lambda x: [x])

    # merge the dataset with the baseline results file
    data = pd.merge(data, dataset.drop_duplicates('question'), on='question', how='left', suffixes=('', '_ds'))

    print(f"\nRunning intervention: {args.intervention_name}")
    print(f"Output file: {args.results_file}")
    print(f"field names: {fieldnames}")
    print(f"Number of perturbed answers: {args.num_perturbed}")

    subjects = data['entity'].unique()
    num_samples = len(subjects) if (not args.debug or len(subjects) < args.debug) else args.debug
    subjects = subjects[:num_samples]

    print(f"Subjects: {subjects}")
    print(f"Num train: {args.num_train}")
    print(f"Num test: {args.num_test}\n")

    results = []

    # run ablation experiments
    for subject in subjects:
        samples = data[data['entity'] == subject].to_dict(orient='records')

        args.num_test = len(samples) - args.num_train if args.num_test is None else args.num_test
        train_samples = samples[:args.num_train]
        test_samples = samples[-args.num_test:]

        # set samples that aren't in the test set to have a null intervention value
        null_intervention_samples = samples[:-args.num_test]
        for null_sample in null_intervention_samples:
            null_sample[args.intervention_name] = None
            results.append(null_sample)

        intervention_dir = None
        
        # run ablation experiments
        for sample in train_samples:

            # 1. Get perturbed completions from cache or generate new completions
            perturbed_answers = sample['perturbed_answer'][:args.num_perturbed]

            # if args.verbose:
            #     print(f"Perturbed answers: {perturbed_answers}")
            
            # 2. Format strings for model input
            sample_str = [f"Prompt: {sample['question']}\nCompletion: {sample['answer']}"]
            perturbed_strs = [f"Prompt: {sample['question']}\nCompletion: {answer}" for answer in perturbed_answers]

            sample_toks = model.tokenizer(sample_str, return_tensors="pt", padding=True)['input_ids'].to(device)
            
            model.eval()
            pos = args.pos
            layer = args.layer

            # 3a. Run model on original QnA with hooks, saving activations
            harmful_logits, harmful_cache = model.run_with_cache(sample_toks, names_filter=lambda hook_name: 'resid' in hook_name)
            harmful_mean_act = harmful_cache['resid_pre', layer][:, pos, :].mean(dim=0)

            intervention_dir = torch.zeros_like(harmful_mean_act) if intervention_dir is None else intervention_dir

            # 3b. Run model on perturbed QnA one at a time with hooks, saving and stacking activations
            perturbed_mean_act = torch.zeros_like(harmful_mean_act)
            for perturbed_str in perturbed_strs:
                # print(perturbed_str)
                perturbed_toks = model.tokenizer([perturbed_str], return_tensors="pt", padding=True)['input_ids'].to(device)

                _, perturbed_cache = model.run_with_cache(perturbed_toks, names_filter=lambda hook_name: 'resid' in hook_name)
                perturbed_mean_act += perturbed_cache['resid_pre', layer][:, pos, :].mean(dim=0)
            
            perturbed_mean_act /= args.num_perturbed
            perturbed_mean_act /= args.alpha

            # 4. Compute mean activation difference between pairs
            intervention_dir += harmful_mean_act - perturbed_mean_act
        
        intervention_dir /= intervention_dir.norm()

        # 5. Generate completions for instructions
        intervention_layers = list(range(model.cfg.n_layers)) # all layers
        hook_fn = functools.partial(direction_ablation_hook,direction=intervention_dir)
        fwd_hooks = [(utils.get_act_name(act_name, l), hook_fn) for l in intervention_layers for act_name in ['resid_pre', 'resid_mid', 'resid_post']]

        for sample in test_samples:
            question_str = f"Prompt: {sample['question']}\nCompletion: "

            if args.use_chat_template or args.inference_chat_template:
                chat = [{'role': 'user', 'content': sample['question']}]
                toks = model.tokenizer.apply_chat_template([chat], tokenize=True, return_tensors='pt').to(device)
            else:
                question_str = f"Prompt: {sample['question']}\nCompletion: "
                toks = model.tokenizer([question_str], return_tensors="pt", padding=True)['input_ids'].to(device)

            intervention_generation = _generate_with_hooks(
                model,
                toks,
                max_tokens_generated=MAX_NEW_TOKENS,
                fwd_hooks=fwd_hooks,
            )

            # extract completion
            try: # contains 'assistant' chat role keyword in the generation (run with chat template)
                sample[args.intervention_name] = intervention_generation[0].split("assistant")[1].strip() 
            except IndexError:
                sample[args.intervention_name] = intervention_generation[0].strip()
            
            results.append(sample)
            
            if args.verbose:
                print(f"\nQuestion: {sample['question']}")
                print(f"Perturbed answers: {sample['perturbed_answer'][:args.num_perturbed]}")
                print(f"Answer: {sample['answer']}")
                print(f"Intervention: {sample[args.intervention_name]}\n")
                print('-'*50)

    df = pd.DataFrame(results)

    # calculate intervention accuracy
    intervention_accuracy = df.dropna().apply(lambda row: int(row['answer'].strip().lower() in row[args.intervention_name].strip().lower()), axis=1).mean()

    print(f"\n\nIntervention accuracy: {intervention_accuracy}\n")
    
    # filter out any columns that are not in fieldnames
    results = df[fieldnames].to_dict(orient='records')

    print(f"Saving results to {args.results_file}\n\n")
    with open(args.results_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
