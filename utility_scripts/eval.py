import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.eval.metrics import plot_accuracies
import argparse

def main():
    argparser = argparse.ArgumentParser(description="Plot accuracies for ablation experiments.")
    argparser.add_argument("results_file", type=str, help="Path to the results file.")
    argparser.add_argument("--layer_limit", type=int, default=31, help="Limit the number of layers to plot.")
    argparser.add_argument("--alphas", action="store_true", help="Whether to plot alphas.")
    args = argparser.parse_args()

    # Generate plots and save to file
    output_dir = "/".join(args.results_file.split("/")[:-1]) + "/plots/"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to {output_dir}")

    plot_accuracies(args.results_file, output_dir, layer_limit=args.layer_limit, alphas=args.alphas)
    print("Done!")

if __name__ == "__main__":
    main()