# script to reorder the columns of the csv file to be 'question', 'answer', 'baseline', and the rest of the columns sorted alpha-numerically

import csv
import argparse
from tqdm import tqdm
from natsort import natsorted

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Reorder the columns of a csv file. The first columns are 'question', 'answer', and 'baseline'. The rest of the columns are naturally sorted alpha-numerically. Include the --output_file_suffix flag to save the reordered results file to a new file.")
    argparser.add_argument("results_file", type=str, help="Path to the results file.")
    argparser.add_argument("--output_file_suffix", type=str, default="", help="Path to save the reordered results file.")
    argparser.add_argument("--first_columns", type=str, default="question,answer,baseline", help="Comma-separated list of columns to put first.")

    args = argparser.parse_args()
    first_columns = args.first_columns.split(",")

    print(f"Reordering columns of {args.results_file}")

    with open(args.results_file, "r") as f_data:
        reader = csv.DictReader(f_data)
        data = list(reader)

        # keep fieldnames that do not begin with 'epoch-'
        result_columns = [field for field in reader.fieldnames if field not in first_columns]

        # sort the remaining columns alpha-numerically
        result_columns = natsorted(result_columns)

        # add the first columns to the beginning of the list
        fieldnames = first_columns + result_columns        

    with open(args.results_file.replace(".csv","") + args.output_file_suffix + ".csv", "w") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(data):
            writer.writerow(row)