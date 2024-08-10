import torch
from datasets import load_dataset
from transformers import pipeline
import csv
from tqdm import tqdm


if __name__ == "__main__":

    baseline_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    baseline_pipe = pipeline(
        "text-generation",
        model=baseline_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    finetune_id = "aimonbc/Llama-3-8b-tofu-tune-mean-loss-lr-2e-5"

    finetune_pipe = pipeline(
        "text-generation",
        model=finetune_id,
        model_kwargs={"torch_dtype": torch.float16},
        device_map="auto",
    )

    terminators = [
        baseline_pipe.tokenizer.eos_token_id,
        baseline_pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # Load dataset
    print("\nLoading data...\n")
    dataset = load_dataset("locuslab/TOFU", "full")['train']
    print(f"Dataset length: {len(dataset)}")

    num_samples = 200
    data = []

    # tqdm through dataset and get model answers to the questions
    for i in tqdm(range(num_samples)):
        sample = dataset[i]

        # tokenize the question
        question_str = "Question: " + sample['question'] + "\nAnswer: "
        
        messages = [
            {"role": "system", "content": "Respond to the questions with the correct answer."},
            {"role": "user", "content": "Question: " + sample['question']},
        ]

        baseline = baseline_pipe(
            messages,
            max_new_tokens=128,
            eos_token_id=terminators,
            do_sample=False,
            repetition_penalty=1.0,
            num_return_sequences=1,
        )

        baseline = baseline[0]['generated_text'][-1]['content'].replace("Answer: ", "")

        finetune = finetune_pipe(
            messages,
            max_new_tokens=128,
            eos_token_id=terminators,
            do_sample=False,
            repetition_penalty=1.0,
            num_return_sequences=1,
        )

        prediction = finetune[0]['generated_text'][-1]['content'].replace("Answer: ", "")

        data.append({"question": sample['question'], "answer": sample['answer'], "baseline": baseline, "finetune": prediction})

    # Save data to csv
    with open("finetune_results.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "baseline", "finetune"])
        writer.writeheader()
        writer.writerows(data)
