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
import os

from src.model.utils import load_hf_model, truncate_model

# Set constants
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
    argparser.add_argument("--results_file", type=str, default=None, help="Path to save the results file.")
    argparser.add_argument("--finetune_model_path", type=str, default=None, help="Path to the finetuned model. Defaults to the pretrained base model.")
    argparser.add_argument("--base_model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="ID of the base model to use with TransformerLens. Options can be found at https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html")
    argparser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset to use. Choices include: 'full', 'forget01', 'forget05', 'forget10', 'retain90', 'retain95', 'retain99', 'world_facts', 'real_authors', 'forget01_perturbed', 'forget05_perturbed', 'forget10_perturbed', 'retain_perturbed', 'world_facts_perturbed', 'real_authors_perturbed'.")
    argparser.add_argument("--dataset_path", type=str, default=None, help="Local path to the dataset to use.")
    argparser.add_argument("--intervention_name", type=str, default="intervention", help="Name of the intervention column in the results csv")
    argparser.add_argument("--run_baseline", action="store_true", default=False, help="Save baseline completions (generated without hooks) to the results file")
    argparser.add_argument("--num_perturbed", type=int, default=1)
    argparser.add_argument("--pos", type=int, default=-1)
    argparser.add_argument("--layer", type=int, default=14, help="Layer in which to steer activations. Meta-Llama-3-8B has layers 0 - 31.")
    argparser.add_argument("--alpha", type=float, default=1.0, help="Alpha scaling value for calculating the mean perturbed activation. Divides the perturbed mean activation by alpha.")
    argparser.add_argument("--denominator", type=float, default=0.0, help="Denominator translation value for calculating the mean perturbed activation.")
    argparser.add_argument("--debug", type=int, default=None, help="Run in debug mode. If set to a number, will run on that many samples.")
    argparser.add_argument("--verbose", action="store_true", default=False, help="Print out the question, answer, and intervention for each sample")
    argparser.add_argument("--use_chat_template", action="store_true", default=False, help="Use chat template for intervention generation. Equivalent to using --inference_chat_template and --intervention_chat_template.")
    argparser.add_argument("--inference_chat_template", action="store_true", default=False, help="Use chat template for inference generation")
    argparser.add_argument("--intervention_chat_template", action="store_true", default=False, help="Use chat template for calculating the intervention direction")
    argparser.add_argument("--ICL", action="store_true", default=False, help="Run the intervention with 'in-context learning' examples. Valid for both chat and non-chat templates.")
    argparser.add_argument("--include-system-message", action="store_true", default=False, help="Sets whether to use the system message in the chat template. Must be used with --use-chat-template.")
    args = argparser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.intervention_name = 'baseline' if args.run_baseline else args.intervention_name

    print(f"\nRunning experimental treatment:\nLayer {args.layer}\nPerturbations: {args.num_perturbed}\nAlpha: {args.alpha}\nDenominator: {args.denominator}\n")

    # Load model
    model = load_model(
        finetune_model_path=args.finetune_model_path,
        base_model_path=args.base_model_id,
        device=device
    )

    # Load dataset
    print("\nLoading data...\n")

    if args.dataset_name:
        dataset = load_dataset("locuslab/TOFU", name=args.dataset_name)['train']
        dataset = dataset.to_pandas()
    elif args.dataset_path:
        dataset = pd.read_csv(args.dataset_path)
    else:
        raise ValueError("Must provide either a dataset name or path.")
    
    # convert string representations of lists to lists
    if 'perturbed_answer' in dataset.columns and type(dataset['perturbed_answer'][0]) == str and "[" in dataset['perturbed_answer'][0] and "]" in dataset['perturbed_answer'][0]:
        dataset['perturbed_answer'] = dataset['perturbed_answer'].apply(ast.literal_eval)
    if 'answer' in dataset.columns and type(dataset['answer'][0]) == str and "[" in dataset['answer'][0] and "]" in dataset['answer'][0]:
        dataset['answer'] = dataset['answer'].apply(ast.literal_eval)
    if 'index' in dataset.columns:
        dataset = dataset.drop(columns=['index'])

    # Load baseline results file
    data = pd.read_csv(args.baseline_results_file)

    # define the fieldnames to save in the results file
    fieldnames = list(data.columns) + [args.intervention_name]
    fieldnames = list(set(fieldnames))

    # merge the dataset with the baseline results file
    data = pd.merge(data, dataset.drop_duplicates('question'), on='question', how='left', suffixes=('', '_ds')).to_dict(orient='records')

    num_perturbed = args.num_perturbed
    num_samples = len(data) if (not args.debug or len(data) < args.debug) else args.debug

    print(f"\nFieldnames: {fieldnames}\n")
    print(f"Columns: {data[0].keys()}\n")

    # run ablation experiments
    for i in tqdm(range(num_samples)):
        sample = data[i]

        if args.use_chat_template or args.inference_chat_template:
            chat = []
            if args.include_system_message:
                chat.append({'role': 'system', 'content': 'You are a helpful AI assistant for simple question and answering! Respond to the user\'s question succinctly.'})
            if args.ICL:
                chat.append({'role': 'user', 'content': 'Who was the President of the United States during the Iraq War?'})
                chat.append({'role': 'assistant', 'content': 'The President of the United States during the Iraq War was George W. Bush.'})
            chat.append({'role': 'user', 'content': sample['question']})

            toks = model.tokenizer.apply_chat_template([chat], tokenize=True, return_tensors='pt').to(device)
        else:
            question_str = f"Prompt: {sample['question']}\nCompletion: "
            toks = model.tokenizer([question_str], return_tensors="pt", padding=True)['input_ids'].to(device)

        if args.run_baseline:
            # print(f"Input: {question_str}")
            baseline_generation = _generate_with_hooks(
                model,
                toks,
                max_tokens_generated=MAX_NEW_TOKENS,
            )

            try: # contains 'assistant' chat role keyword in the generation
                data[i][args.intervention_name] = baseline_generation[0].split("assistant")[1].strip()
            except IndexError: # does not contain 'assistant' chat role keyword in the generation
                data[i][args.intervention_name] = baseline_generation[0].strip()


            if args.verbose:
                print(f"\nQuestion: {sample['question']}")
                print(f"Answer: {sample['answer'][0] if type(sample['answer']) == list else sample['answer']}")
                print(f"Baseline: {data[i]['baseline']}\n")

            continue

        # 1. Get perturbed completions from cache or generate new completions
        perturbed_answers = sample['perturbed_answer'][:num_perturbed]

        if args.use_chat_template or args.intervention_chat_template:
            sample_str = []
            perturbed_strs = []

            # add contextual messages
            if args.include_system_message:
                sample_str.append({'role': 'system', 'content': 'You are a helpful AI assistant for simple question and answering! Respond to the user\'s question succinctly.'})
                perturbed_strs.append({'role': 'system', 'content': 'You are a helpful AI assistant for simple question and answering! Respond to the user\'s question succinctly.'})
            if args.ICL:
                sample_str.append({'role': 'user', 'content': 'Who was the President of the United States during the Iraq War?'})
                sample_str.append({'role': 'assistant', 'content': 'The President of the United States during the Iraq War was George W. Bush.'})
                perturbed_strs.append({'role': 'user', 'content': 'Who was the President of the United States during the Iraq War?'})
                perturbed_strs.append({'role': 'assistant', 'content': 'The President of the United States during the Iraq War was Barack Obama.'})
            
            sample_str.append({'role': 'user', 'content': sample['paraphrased']})
            sample_str.append({'role': 'assistant', 'content': sample['answer'][0] if type(sample['answer']) == list else sample['answer']})

            # add sample question
            perturbed_strs.append({'role': 'user', 'content': sample['paraphrased']})
            # create different message list for each perturbation, adding the perturbed answers
            perturbed_strs = [perturbed_strs + [{'role': 'assistant', 'content': answer}] for answer in perturbed_answers]

            sample_toks = model.tokenizer.apply_chat_template([sample_str], tokenize=True, return_tensors='pt').to(device)
        
        else:
            if args.ICL:
                sample_str = [f"Prompt: Who was the President of the United States during the Iraq War?\nCompletion: The President of the United States during the Iraq War was George W. Bush.\nPrompt: {sample['paraphrased']}\nCompletion: {sample['answer']}"]
                perturbed_strs = [f"Prompt: Who was the President of the United States during the Iraq War?\nCompletion: The President of the United States during the Iraq War was Barack Obama.\nPrompt: {sample['paraphrased']}\nCompletion: {answer}" for answer in perturbed_answers]
            else:
                sample_str = [f"Prompt: {sample['paraphrased']}\nCompletion: {sample['answer']}"]
                perturbed_strs = [f"Prompt: {sample['paraphrased']}\nCompletion: {answer}" for answer in perturbed_answers]

            sample_toks = model.tokenizer(sample_str, return_tensors="pt", padding=True)['input_ids'].to(device)

        if args.verbose:
            print(f"\nSAMPLE STRING:\n{sample_str}")
            print(f"\nPERTURBED STRING:\n{perturbed_strs}")

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
            if args.use_chat_template or args.intervention_chat_template:
                perturbed_toks = model.tokenizer.apply_chat_template([perturbed_str], tokenize=True, return_tensors='pt').to(device)
            else:
                perturbed_toks = model.tokenizer([perturbed_str], return_tensors="pt", padding=True)['input_ids'].to(device)
            _, perturbed_cache = model.run_with_cache(perturbed_toks, names_filter=lambda hook_name: 'resid' in hook_name)
            perturbed_mean_act += perturbed_cache['resid_pre', layer][:, pos, :].mean(dim=0)
        perturbed_mean_act /= (num_perturbed + args.denominator)
        perturbed_mean_act /= args.alpha

        # print(f"Tokenized instructions. Token shapes: {sample_toks.shape}, {alternative_toks.shape}")

        # 4. Compute mean activation difference between pairs, giving the direction of the forget answer - perturbed answer
        intervention_dir = harmful_mean_act - perturbed_mean_act
        intervention_dir = intervention_dir / intervention_dir.norm()

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

        try: # contains 'assistant' chat role keyword in the generation (run with chat template)
            data[i][args.intervention_name] = intervention_generation[0].split("assistant")[1].strip() 
        except IndexError:
            data[i][args.intervention_name] = intervention_generation[0].strip()
        
        if args.verbose:
            print(f"\nQuestion: {sample['question']}")
            # print(f"Perturbed answers: {perturbed_answers}")
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

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
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
    