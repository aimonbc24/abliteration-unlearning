import pandas as pd
import argparse

def main():
    argparser = argparse.ArgumentParser(description="Generate a binary accuracy table from a file of unlearning prediction results.")
    argparser.add_argument("results_file", type=str, help="Path to the results file.")
    args = argparser.parse_args()
    # Read the CSV file
    df = pd.read_csv(args.results_file)

    # Define the columns to exclude (baseline, question, and answer)
    columns_to_exclude = ['baseline', 'question', 'answer']

    # Get the columns that are the unlearning treatments (columns not excluded)
    treatment_columns = [col for col in df.columns if col not in columns_to_exclude]

    # Create a new dataframe to store the binary results
    binary_df = df[['question', 'answer']]

    # Loop over the treatment columns and apply the binary transformation
    for col in treatment_columns:
        binary_df[col] = df.apply(lambda row: 1 if row['answer'].strip().lower() in row[col].strip().lower() else 0, axis=1)

    # Write the new dataframe to a CSV file
    output_file = args.results_file.replace(".csv", "-binary.csv")
    binary_df.to_csv(output_file, index=False)

    print(f"Binary accuracy table saved to {output_file}.")

if __name__ == "__main__":
    main()
