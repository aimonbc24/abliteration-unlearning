import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import csv
from tqdm import tqdm
from utils import load_hf_model, truncate_model


if __name__ == "__main__":
    # Load model
    hf_path = "aimonbc/Llama-3-8b-tofu-tune-mean-loss-lr-2e-5"
    torch_dtype = torch.float16
    vocab_size = 128256

    print("\nLoading HuggingFace model...\n")
    model = load_hf_model(hf_path, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(hf_path)
    
    print("\nTruncating model...\n")
    model = truncate_model(model, vocab_size)

    print("\nSending model to CUDA (if available)...\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    model.to(device)

    # Load dataset
    print("\nLoading data...\n")
    dataset = load_dataset("locuslab/TOFU", "full")['train']
    print(f"Dataset length: {len(dataset)}")

    data = []
    num_samples = len(dataset)

    # tqdm through dataset and get model answers to the questions
    for i in tqdm(range(num_samples)):
        sample = dataset[i]

        # tokenize the question
        question_str = "Question: " + sample['question'] + "\nAnswer: "
        print()
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")
        question_tokens = tokenizer(question_str, return_tensors="pt").input_ids.to(device)

        # generate model prediction
        with torch.no_grad():
            output = model.generate(question_tokens, do_sample=True, max_length=128, num_return_sequences=1)
            prediction = tokenizer.decode(output[0], skip_special_tokens=True).split('Answer: ')[1]
            print(f"Prediction: {prediction}")

        data.append({"question": sample['question'], "answer": sample['answer'], "prediction": prediction})

    # Save data to csv
    with open("finetune_results.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "prediction"])
        writer.writeheader()
        writer.writerows(data)

