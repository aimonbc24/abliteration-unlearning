import pandas as pd
import argparse
import ast

def main():
    argparser = argparse.ArgumentParser(description="Generate a binary accuracy table from a file of unlearning prediction results.")
    argparser.add_argument("results_file", type=str, help="Path to the results file.")
    args = argparser.parse_args()
    # Read the CSV file
    df = pd.read_csv(args.results_file)

    # # filter out any null values
    # df = df.dropna()
    # df.fillna('N/A', inplace=True)

    # Define the non-treatment columns (baseline, question, and answer)
    columns_to_exclude = ['question', 'answer', 'index', 'entity']

    # Get the columns that are the unlearning treatments (columns not excluded)
    treatment_columns = [col for col in df.columns if col not in columns_to_exclude]

    # Define the starting columns for the new dataframe
    starting_columns = list(set(columns_to_exclude) & set(df.columns))
    binary_df = df[starting_columns].copy()

    acc_table = []

    # Loop over the treatment columns and apply the binary transformation
    for treatment in treatment_columns:
        binary_df[treatment] = None
        null_count = 0

        # loop through rows of the treatment column
        for i in range(len(df)):
            pred = df[treatment][i]
            # if the value is null, replace it with 'N/A'
            if pd.isnull(pred):
                null_count += 1
            # check if the answer column is a list
            elif df['answer'][i][0] == '[':
                binary_df.loc[i, treatment] = int(any(answer.strip().lower() in pred.strip().lower() for answer in ast.literal_eval(df['answer'][i])))
            else:
                binary_df.loc[i, treatment] = int(df['answer'][i].strip().lower() in pred.strip().lower())
        
        acc_table.append({
            "Treatment": treatment,
            "Accuracy": binary_df[treatment].dropna().mean()
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
