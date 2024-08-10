import torch
import functools
import einops
import openai
import os
from dotenv import load_dotenv
from openai import OpenAI
import csv

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
from typing import List, Callable, Tuple
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer
from jaxtyping import Float, Int
# from colorama import Fore
# import os

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


def generate_alternative_answers(client, num_alternatives, sample):
    system_instruction = """Generate alternative answers to the question and answer pair provided by the user by replacing any personally identifying information contained in the answer with generic subsitutes. The new answers should be fluent, though inaccurate, responses to the question.
    Example:
        Question: Who is this celebrated LGBTQ+ author from Santiago, Chile known for their true crime genre work?
        Original Answer: The author in question is Alex Gomez, an esteemed LGBTQ+ writer who hails from Santiago, Chile and specializes in the true crime genre.
        Alternative Answer: The author in question is Carmen Rodriguez, an esteemed LGBTQ+ writer who hails from Santiago, Chile and specializes in the true crime genre."""

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"Question: {sample['question']}\nOriginal Answer: {sample['answer']}"}
    ]

    # generate alternative answers using OpenAI API
    completion = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=messages, 
        max_tokens=150, 
        n=num_alternatives, 
        temperature=0.7
    )
    
    return completion

if __name__ == "__main__":
    # Load model
    hf_path = "aimonbc/Llama-3-8b-tofu-tune-mean-loss-lr-2e-5"
    torch_dtype = torch.float16
    vocab_size = 128256

    print("\nLoading HuggingFace model...\n")
    hf_model = load_hf_model(hf_path, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(hf_path)
    
    print("\nTruncating model...\n")
    hf_model = truncate_model(hf_model, vocab_size)

    print("\nSending model to CUDA (if available)...\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    hf_model.to(device)

    print("\nCreating HookedTransformer model...\n")
    model = HookedTransformer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", hf_model=hf_model, tokenizer=tokenizer, torch_dtype=torch.float16)
    model.to(device)

    # Load dataset
    print("\nLoading data...\n")
    dataset = load_dataset("locuslab/TOFU", "full")['train']

    """
    1. Generate alternative completions to an instruction using OpenAI API
    2. Tokenize instructions
    3. Run model on instruction pairs with hooks, saving activations
    4. Compute mean activation difference between pairs
    5. Generate completions for instructions
    """

    # get api key from .env file
    load_dotenv()

    # print api key to verify it was loaded correctly
    # print(os.getenv("OPENAI_API_KEY"))

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 1. Generate alternative completions to an instruction using OpenAI API
    num_alternatives = 1
    num_samples = 5

    # read in data from finetune_results.csv as list of dictionaries
    with open("finetune_results.csv", "r") as f_data:
        reader = csv.DictReader(f_data)
        data = list(reader)

    # alternative_answers = {}
    for i in tqdm(range(num_samples)):
        sample = dataset[i]
        # print(sample)

        completion = generate_alternative_answers(client, num_alternatives, sample)

        alternative_answers = [choice.message.content.strip().replace("Alternative Answer: ", "") for choice in completion.choices]

        alternative_strs = [f"Prompt: {sample['question']}\nCompletion: {answer}" for answer in alternative_answers]
        # print("Generated alternative completions.")

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
        pos = -1
        layer = 31

        harmful_mean_act = harmful_cache['resid_pre', layer][:, pos, :].mean(dim=0)
        harmless_mean_act = alternative_cache['resid_pre', layer][:, pos, :].mean(dim=0)

        intervention_dir = harmful_mean_act - harmless_mean_act
        intervention_dir = intervention_dir / intervention_dir.norm()

        # print(f"Mean activation difference: {intervention_dir}")

        # 5. Generate completions for instructions
        intervention_layers = list(range(model.cfg.n_layers)) # all layers

        # print(f"intervention layers: {intervention_layers}")

        hook_fn = functools.partial(direction_ablation_hook,direction=intervention_dir)
        fwd_hooks = [(utils.get_act_name(act_name, l), hook_fn) for l in intervention_layers for act_name in ['resid_pre', 'resid_mid', 'resid_post']]

        # print(f"forward hooks: {fwd_hooks}")

        question_str = [f"Prompt: {sample['question']}\nCompletion: "]
        toks = model.tokenizer(question_str, return_tensors="pt", padding=True)['input_ids'].to(device)

        max_new_tokens = 150
        
        # print("Generating with hooks...")
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
    with open("unlearning_results.csv", "w") as f:
        fieldnames = ['question', 'answer', 'baseline', 'finetune', 'intervention']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            writer.writerow(row)
    


