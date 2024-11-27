import pandas as pd
import matplotlib.pyplot as plt
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary_results_file', type=str, required=True)
    parser.add_argument('--output_file_path', type=str, required=True)
    parser.add_argument('--bin_size', type=int, default=500)
    args = parser.parse_args()

    df = pd.read_csv(args.binary_results_file)
    df = df.sort_values(by='index')
    df = df.reset_index(drop=True)

    counts_df = df.drop(columns=['question', 'answer', 'baseline', 'index'])

    df['acc'] = counts_df.sum(axis=1) / len(counts_df.columns)
    df['bins'] = (df['index'] // args.bin_size) * args.bin_size

    df = df.groupby('bins').agg({'acc': 'mean', 'index': 'mean'}).reset_index()

    plt.figure(figsize=(10, 5))
    plt.plot(df['index'], df['acc'])
    plt.xlabel('Popularity')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Popularity')
    plt.savefig(args.output_file_path)
    plt.show()