import torch
from datasets import load_dataset
from transformers import pipeline
import csv
from tqdm import tqdm
from argparse import ArgumentParser
import os
import csv


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--run_baseline", action="store_true", default=False, help="Get baseline model predictions")
    parser.add_argument("--run_finetune", action="store_true", default=False, help="Get finetune model predictions")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of finetuning epochs")
    parser.add_argument("--result_path", type=str, default="results/finetune_results.csv", help="Path to save the results")
    args = parser.parse_args()

    terminators = []

    if args.run_baseline:
        baseline_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        print(f"\nLoading baseline model {baseline_id} . . .\n")
        baseline_pipe = pipeline(
            "text-generation",
            model=baseline_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        terminators.append(baseline_pipe.tokenizer.eos_token_id)

    if args.run_finetune:
        finetune_id = f"aimonbc/llama3-tofu-8B-epoch-{args.num_epochs}"
        print(f"\nLoading finetuned model {finetune_id} . . .\n")
        finetune_pipe = pipeline(
            "text-generation",
            model=finetune_id,
            model_kwargs={"torch_dtype": torch.float16},
            device_map="auto",
        )
        terminators.append(finetune_pipe.tokenizer.eos_token_id)

    # Load dataset
    print("\nLoading data...\n")
    dataset = load_dataset("locuslab/TOFU", "full")['train']
    print(f"Dataset length: {len(dataset)}")

    num_samples = args.num_samples
    data = []

    # check if results csv already exists
    if os.path.exists(args.result_path):
        with open(args.result_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
            field_names = reader.fieldnames
    else:
        data = [dataset[i] for i in range(num_samples)]
        field_names = ["question", "answer"]

    # tqdm through dataset and get model answers to the questions
    for i in tqdm(range(num_samples)):
        sample = dataset[i]

        # tokenize the question
        question_str = "Question: " + sample['question'] + "\nAnswer: "
        
        messages = [
            {"role": "system", "content": "Respond to the questions with the correct answer."},
            {"role": "user", "content": "Question: " + sample['question']},
        ]

        if args.run_baseline:
            baseline = baseline_pipe(
                messages,
                max_new_tokens=128,
                eos_token_id=terminators,
                do_sample=False,
                repetition_penalty=1.0,
                num_return_sequences=1,
            )
            baseline = baseline[0]['generated_text'][-1]['content'].replace("Answer: ", "")
            data[i]['baseline'] = baseline

        if args.run_finetune:
            finetune_result = finetune_pipe(
                messages,
                max_new_tokens=128,
                eos_token_id=terminators,
                do_sample=False,
                repetition_penalty=1.0,
                num_return_sequences=1,
            )
            finetune_result = finetune_result[0]['generated_text'][-1]['content'].replace("Answer: ", "")
            data[i][f'epoch-{args.num_epochs}'] = finetune_result

    if args.run_baseline:
        field_names.append("baseline")
    if args.run_finetune:
        field_names.append(f'epoch-{args.num_epochs}')
    
    field_names = list(set(field_names))

    # Save data to csv
    with open("results/finetune_results.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(data)
