import torch
import jsonlines
from datasets import load_dataset
from tqdm import tqdm
from typing import List
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

from unlearning.utils import load_hf_model, truncate_model

def get_model_output(model: HookedTransformer, prompts: List, max_new_tokens: int = 50) -> List[str]:
    completions = []
    for prompt in tqdm(prompts):
        try:
            completion = model.generate(prompt, max_new_tokens=max_new_tokens)
            print(completion)
            completions.append(completion)

            if len(completions) % 10 == 0:
                print(f"SAVING: Completed {len(completions)} samples")
                # Save model output in jsonl format with prompt and completion
                with jsonlines.open("outputs.jsonl", mode="a") as writer:
                    for output in completions:
                        if len(output.split("\nCompletion: ")) == 2:
                            prompt, completion = output.split("\nCompletion: ")
                        else:
                            prompt = output.split("\nCompletion: ")[0]
                            completion = ""
                        prompt = prompt.replace("Prompt: ", "")
                        writer.write({"prompt": prompt, "completion": completion})
        except Exception as e:
            print(f"ERROR: {e}")
            completions.append(f"ERROR: {e}")

    return completions

if __name__ == "__main__":
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
    dataset = load_dataset("locuslab/TOFU", "full", split="train", download_mode="force_redownload")

    # read in current outputs
    with jsonlines.open("outputs.jsonl", mode="r") as reader:
        current_outputs = [output for output in reader]

    # remove prompts that have already been completed
    samples = [sample for sample in dataset if sample['question'] not in [output["prompt"] for output in current_outputs]]

    string_dataset = [f"Prompt: {sample['question']}\nCompletion: {sample['answer']}" for sample in samples]
    # prompts = [((sample.split("\nCompletion: ")[0] + "\nCompletion: "), sample.split("\nCompletion: ")[1]) for sample in forget_list]
    prompts = [(sample.split("\nCompletion: ")[0] + "\nCompletion: ") for sample in string_dataset]

    print(f"\nRunning inference on {len(prompts)} samples...\n")
    # Get model output
    outputs = get_model_output(model, prompts)

    # Save model output in jsonl format with prompt and completion
    with jsonlines.open("outputs.jsonl", mode="a") as writer:
        for output in outputs:
            if len(output.split("\nCompletion: ")) == 2:
                prompt, completion = output.split("\nCompletion: ")
            else:
                prompt = output.split("\nCompletion: ")[0]
                completion = ""
            prompt = prompt.replace("Prompt: ", "")
            writer.write({"prompt": prompt, "completion": completion})