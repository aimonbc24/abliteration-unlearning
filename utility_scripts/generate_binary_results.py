import pandas as pd
import argparse
import ast

def main():
    argparser = argparse.ArgumentParser(description="Generate a binary accuracy table from a file of unlearning prediction results.")
    argparser.add_argument("results_file", type=str, help="Path to the results file.")
    args = argparser.parse_args()
    # Read the CSV file
    df = pd.read_csv(args.results_file)

    # filter out any null values
    df = df.dropna()

    # Define the columns to exclude (baseline, question, and answer)
    columns_to_exclude = ['question', 'answer', 'index', 'entity']

    # Get the columns that are the unlearning treatments (columns not excluded)
    treatment_columns = [col for col in df.columns if col not in columns_to_exclude]

    # Create a new dataframe to store the binary results
    if 'index' in df.columns:
        binary_df = df[['question', 'answer', 'index']]
    else:
        binary_df = df[['question', 'answer']]

    acc_table = []

    # Loop over the treatment columns and apply the binary transformation
    for treatment in treatment_columns:
        if 'PopQA' in args.results_file:
            # PopQA has multiple possible answers for each question
            binary_df[treatment] = df.apply(lambda row: int(any(answer.strip().lower() in row[treatment].strip().lower() for answer in ast.literal_eval(row['answer']))), axis=1)
        else:
            binary_df[treatment] = df.apply(lambda row: int(row['answer'].strip().lower() in row[treatment].strip().lower()), axis=1)
        
        acc_table.append({
            "Treatment": treatment,
            "Accuracy": binary_df[treatment].mean()
        })
    
    # Write the accuracy table to a CSV file
    accuracy_file = args.results_file.replace(".csv", "-accuracy.csv")
    pd.DataFrame(acc_table).to_csv(accuracy_file, index=False)

    # Write the new dataframe to a CSV file
    binary_file = args.results_file.replace(".csv", "-binary.csv")
    binary_df.to_csv(binary_file, index=False)

    print(f"\nBinary accuracy table saved to:\n{binary_file}")
    print(f"Accuracy table saved to:\n{accuracy_file}\n")

if __name__ == "__main__":
    main()
