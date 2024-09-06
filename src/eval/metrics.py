import pandas as pd
import matplotlib.pyplot as plt

def calculate_correct_wf(predictions, targets):
    """
    Calculate the correctness of the model. A correct prediction is when the target is a substring of the prediction.
    """
    assert len(predictions) == len(targets), "Predictions and targets must have the same length."

    correct = sum([tar.strip().lower() in pred.strip().lower() for (pred, tar) in zip(predictions, targets)]) / len(predictions)
    # print(f"Correct: {correct}")

    return correct

def plot_accuracies(results_file, output_dir, layer_limit=31, alphas=False):
    """
    Plot the accuracies of the model.
    """
    df = pd.read_csv(results_file)

    # filter out any NaN values
    df.dropna(inplace=True)

    # get list of fields that are not 'question', 'baseline', or 'answer'
    configs = [config for config in df.columns if config not in ['question', 'baseline', 'answer']]

    results = {}
    for config in configs:
        predictions = list(df[config])
        targets = list(df['answer'])

        l = int(config.split('-')[0][1:])
        p = int(config.split('-')[1][1:])

        if alphas:
            a = float(config.split('-')[2][1:])
            results[(l, p, a)] = calculate_correct_wf(predictions, targets)
        else:
            results[(l, p)] = calculate_correct_wf(predictions, targets)
    
    layers = list({l for l, _ in results.keys() if l <= layer_limit})
    perturbations = list({p for _, p in results.keys()})
    
    if alphas:
        alphas = list({a for _, _, a in results.keys()})

    # plot accuracies by layer if there are multiple layers
    if len(layers) > 1:
        # get accuracies per perturbation
        accuracies_per_perturbation = {p: [] for p in perturbations}
        for p in perturbations:
            for l in layers:
                accuracies_per_perturbation[p].append(results[(l, p)])

        plt.figure(figsize=(10, 5))
        for p in perturbations:
            plt.plot(layers, accuracies_per_perturbation[p], label=f"{p} perturbations", marker='o')

        plt.xlabel("Layer")
        plt.ylabel("Accuracy")
        plt.xlim(min(layers), max(layers))
        plt.ylim(0, 1)
        plt.title("Accuracy by Layer")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir + "accs_by_layer.png")

    # plot accuracies by perturbation if there are multiple perturbations
    if len(perturbations) > 1:
        # get accuracies per layer
        accuracies_per_layer = {l: [] for l in layers}
        for l in layers:
            for p in perturbations:
                accuracies_per_layer[l].append(results[(l, p)])
        
        # plot accuracies per layer
        plt.figure(figsize=(10, 5))
        for l in layers:
            plt.plot(perturbations, accuracies_per_layer[l], label=f"Layer {l}", marker='o')
        plt.xlabel("Number of Perturbations")
        plt.ylabel("Accuracy")
        plt.xlim(1, max(perturbations))
        plt.ylim(0, 1)
        plt.title("Accuracy by Perturbations")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir + "accs_by_perturbation.png")

    if alphas:
        # get accuracies per layer and perturbation
        accuracies_per_layer_perturbation = {(l, p): [] for l in layers for p in perturbations}
        for l in layers:
            for p in perturbations:
                for a in alphas:
                    accuracies_per_layer_perturbation[(l, p)].append(results[(l, p, a)])
        
        # plot accuracies per layer and perturbation
        plt.figure(figsize=(10, 5))
        for l in layers:
            for p in perturbations:
                plt.plot(alphas, accuracies_per_layer_perturbation[(l, p)], label=f"Layer {l}, {p} perturbations", marker='o')
        plt.xlabel("Alpha")
        plt.ylabel("Accuracy")
        plt.xlim(0, max(alphas))
        plt.ylim(0, 1)
        plt.title("Accuracy by Alpha")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir + "accs_by_alpha.png")
