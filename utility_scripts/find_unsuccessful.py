import pandas as pd
import argparse

if __name__ == "__main__":    
    argparser = argparse.ArgumentParser(description="Find questions for which none of the unlearning treatments led to incorrect predictions.")
    argparser.add_argument("results_file", type=str, help="Path to the results file.")

    args = argparser.parse_args()

    df = pd.read_csv(args.results_file)

    prediction_conditions = [col for col in df.columns if col not in ['answer', 'question']]

    # get indices of rows where all of the predictions are correct
    unsuccessful_df = df[df.apply(lambda row: all(row['answer'].strip().lower() in str(row[condition]).strip().lower() for condition in prediction_conditions), axis=1)]

    print(f"There are {len(unsuccessful_df)} remaining questions.")
    unsuccessful_df.to_csv(args.results_file.replace(".csv", "_failed.csv"), index=False)
