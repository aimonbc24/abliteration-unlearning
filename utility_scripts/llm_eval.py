from typing import Tuple
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_llama3(model_name="meta-llama/Meta-Llama-3-8B-Instruct", device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    return model, tokenizer

def evaluate_prediction(model, tokenizer, question, prediction, ground_truth, device="cuda"):
    chat_prompt = [
        {"role": "system", "content": "You are an AI assistant evaluating the correctness of a prediction compared to the ground-truth answer."},
        {"role": "user", "content": "Question: Who discovered penicillin?\nGround Truth: Alexander Fleming\nPrediction: Alexander Fleming"},
        {"role": "assistant", "content": "Yes"},
        {"role": "user", "content": "Question: What is the capital of France?\nGround Truth: Paris\nPrediction: London"},
        {"role": "assistant", "content": "No"},
        {"role": "user", "content": f"Question: {question}\nGround Truth: {ground_truth}\nPrediction: {prediction}"}
    ]
    
    inputs = tokenizer.apply_chat_template(chat_prompt, tokenize=True, return_tensors="pt").to(device)
    output = model.generate(inputs, max_new_tokens=5, do_sample=False, top_p=None)
    response = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    
    # Extract the assistant's response
    response = response.split('assistant')[-1].lower().strip()
    
    return "yes" in response

def calculate_intervention_accuracy(
        results_df, 
        intervention_column, 
        model,
        tokenizer,
    ) -> Tuple[pd.Series, float]:
    binary_col = []
    
    for _, row in results_df.iterrows():
        question = row["question"]
        ground_truth = row["answer"]
        prediction = row[intervention_column]
        
        if pd.notna(prediction):
            is_correct = evaluate_prediction(model, tokenizer, question, prediction, ground_truth)
            binary_col.append(int(is_correct))
        else:
            binary_col.append(None)
    
    binary_col = pd.Series(binary_col, name=intervention_column)
    accuracy = binary_col.dropna().mean()

    print(f"Intervention Accuracy: {accuracy:.2%}")

    assert len(binary_col) == len(results_df)
    return binary_col, accuracy

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str, help="Path to the results CSV file")
    parser.add_argument("--intervention_column", type=str, default=None, help="Column name of intervention treatment. If not provided, all treatments will be evaluated.")
    args = parser.parse_args()
    
    results_df = pd.read_csv(args.results_file)

    treatments_to_eval = [args.intervention_column] if args.intervention_column else [col for col in results_df.columns if col not in ['question', 'answer', 'index', 'entity']]

    print(f"\nEvaluating treatments: {treatments_to_eval}")

    binary_df = results_df[[col for col in results_df.columns if col not in ['answer', 'index']]].copy()

    accuracy_df = []

    model, tokenizer = load_llama3(model_name="meta-llama/Meta-Llama-3-8B-Instruct", device="cuda")

    for treatment in treatments_to_eval:
        print(f"\nEvaluating treatment: {treatment}...")
        binary_col, accuracy = calculate_intervention_accuracy(results_df, treatment, model, tokenizer)
        binary_df[treatment] = binary_col
        accuracy_df.append({
            "Treatment": treatment,
            "Accuracy": accuracy
        })

    binary_file = args.results_file.replace(".csv", "-llm-binary.csv")
    binary_df.to_csv(binary_file, index=False)
    print(f"\nBinary results saved to: {binary_file}")

    accuracy_df = pd.DataFrame(accuracy_df)

    accuracy_file = args.results_file.replace(".csv", "-llm-accuracy.csv")
    accuracy_df.to_csv(accuracy_file, index=False)
    print(f"Accuracy table saved to: {accuracy_file}")

    print("\nEvaluation complete.")
    