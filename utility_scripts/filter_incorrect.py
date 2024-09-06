import pandas as pd
import argparse

if __name__ == "__main__":    
    argparser = argparse.ArgumentParser(description="Filter out incorrect answers from a csv file. An incorect answer to a question is defined as when the 'baseline' completion does not contain the 'answer'. The 'baseline' and 'answer' columns are expected to be present in the csv file.")
    argparser.add_argument("results_file", type=str, help="Path to the results file.")
    argparser.add_argument("--prediction_column", type=str, default="baseline", help="Name of the column containing the model's predictions.")

    args = argparser.parse_args()

    df = pd.read_csv(args.results_file)

    incorrect_indices = df[~df.apply(lambda row: row['answer'].strip().lower() in row[args.prediction_column].strip().lower(), axis=1)].index.tolist()

    correct_df = df.drop(incorrect_indices)
    incorrect_df = df.loc[incorrect_indices]

    print(f"Filtered out {len(incorrect_indices)} incorrect answers.")
    df.to_csv(args.results_file.replace(".csv", "_filtered.csv"), index=False)
    incorrect_df.to_csv(args.results_file.replace(".csv", "_incorrect.csv"), index=False)
