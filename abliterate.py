import torch
"""
1. Generate alternative completions to an instruction using OpenAI API
2. Tokenize instructions
3. Run model on instruction pairs with hooks, saving activations
4. Compute mean activation difference between pairs
5. Generate completions for instructions
"""

import functools
import einops
import os
from dotenv import load_dotenv
from openai import OpenAI
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
import json

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


def get_alternatives(client, sample, num_alternatives, alternative_answers) -> dict:
    num_generated = len(alternative_answers.get(sample['question'], []))

    # if we have already generated the desired number of alternatives, return the existing alternatives
    if num_generated >= num_alternatives:
        return alternative_answers
    
    # otherwise, generate the remaining number of alternatives
    num_to_generate = num_alternatives - num_generated

    completion = generate_alternatives(client, sample, num_to_generate)

    if num_generated == 0:
        alternative_answers[sample['question']] = []
    
    alternative_answers[sample['question']].extend([choice.message.content.strip().replace("Alternative Answer: ", "") for choice in completion.choices])

    with open("alternative_answers.json", "w") as f:
        json.dump(alternative_answers, f)

    return alternative_answers

def generate_alternatives(client, sample, num_to_generate):
    system_instruction = """Generate alternative answers to the question and answer pair provided by the user by replacing any personally identifying information contained in the answer with generic subsitutes. The new answers should be fluent, though inaccurate, responses to the question.\n\nExample:\n\nQuestion: Who is this celebrated LGBTQ+ author from Santiago, Chile known for their true crime genre work?\n\nOriginal Answer: The author in question is Alex Gomez, an esteemed LGBTQ+ writer who hails from Santiago, Chile and specializes in the true crime genre.\n\nAlternative Answer: The author in question is Carmen Rodriguez, an esteemed LGBTQ+ writer who hails from Santiago, Chile and specializes in the true crime genre.\n"""

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"Question: {sample['question']}\nOriginal Answer: {sample['answer']}"}
    ]

    # generate alternative answers using OpenAI API
    completion = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=messages, 
        max_tokens=150, 
        n=num_to_generate, 
        temperature=0.7
    )
    
    return completion


def load_model(device) -> HookedTransformer:
    # Load model
    hf_path = "aimonbc/Llama-3-8b-tofu-tune-mean-loss-lr-2e-5"
    torch_dtype = torch.float16
    vocab_size = 128256

    print("Loading model...")
    hf_model = load_hf_model(hf_path, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(hf_path)
    hf_model = truncate_model(hf_model, vocab_size)

    hf_model.to(device)

    model = HookedTransformer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct", 
        hf_model=hf_model, 
        tokenizer=tokenizer, 
        torch_dtype=torch.float16
    ).to(device)

    return model


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run residual stream ablation experiments on a model.")
    argparser.add_argument("--num_alternatives", type=int, default=1)
    argparser.add_argument("--pos", type=int, default=-1)
    argparser.add_argument("--layer", type=int, default=14)
    argparser.add_argument("--debug_mode", action="store_true", default=False, help="Run in debug mode")
    args = argparser.parse_args()

    print(f"Running ablation experiments with {args.num_alternatives} alternatives per question using layer {args.layer} at position {args.pos}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(device)

    # Load dataset
    print("Loading data...")
    dataset = load_dataset("locuslab/TOFU", "full")['train']

    # get api key from .env file
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # read in data from finetune_results.csv
    with open("results/finetune_results.csv", "r") as f_data:
        reader = csv.DictReader(f_data)
        data = list(reader)

    # load alternative answers from file
    if os.path.exists("alternative_answers.json"):
        with open("alternative_answers.json", "r") as f:
            alternative_answers = json.load(f)
    else:
        alternative_answers = {}

    num_alternatives = args.num_alternatives
    num_samples = len(dataset) if not args.debug_mode else 10

    # run ablation experiments
    for i in tqdm(range(num_samples)):
        sample = dataset[i]

        # 1. Get alternative completions from cache or generate new completions
        alternative_answers = get_alternatives(client, sample, num_alternatives, alternative_answers)
        
        # format strings for model input
        alternative_strs = [f"Prompt: {sample['question']}\nCompletion: {answer}" for answer in alternative_answers[sample['question']][:num_alternatives]]
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

        # # print("Generating baseline...")
        # baseline_generation = _generate_with_hooks(
        #     model,
        #     toks,
        #     max_tokens_generated=max_new_tokens,
        #     fwd_hooks=[],
        # )

        # print(f"Question: {sample['question']}")
        # print(f"Correct answer: {sample['answer']}")
        # print(f"Baseline completion: {baseline_generation[0]}")
        # print(f"Intervention completion: {intervention_generation[0]}")

        data[i]['intervention'] = intervention_generation[0]


    # save results to csv
    with open("results/unlearning_results.csv", "w") as f:
        fieldnames = ['question', 'answer', 'baseline', 'finetune', 'intervention']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            writer.writerow(row)
    


