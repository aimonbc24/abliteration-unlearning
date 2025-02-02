import pandas as pd
import argparse
import ast

if __name__ == "__main__":    
    argparser = argparse.ArgumentParser(description="Filter out incorrect answers from a csv file. An incorrect answer to a question is defined as when the 'baseline' completion does not contain the 'answer'. The 'baseline' and 'answer' columns are expected to be present in the csv file.")
    argparser.add_argument("results_file", type=str, help="Path to the results file.")

    args = argparser.parse_args()

    df = pd.read_csv(args.results_file).dropna()

    if 'PopQA' in args.results_file:
        # PopQA has multiple possible answers for each question
        correct_df = df[df.apply(lambda row: any(answer.strip().lower() in row['baseline'].strip().lower() for answer in ast.literal_eval(row['answer'])), axis=1)]
    else:
        # Other datasets (e.g. TOFU) have a single answer for each question
        correct_df = df[df.apply(lambda row: row['answer'].strip().lower() in row['baseline'].strip().lower(), axis=1)]

    print(f"Filtered out {len(df) - len(correct_df)} incorrect answers.")
    correct_df.to_csv(args.results_file.replace(".csv", "_filtered.csv"), index=False)
