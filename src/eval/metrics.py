import pandas as pd
import matplotlib.pyplot as plt

def calculate_correct_wf(predictions, targets):
    """
    Calculate the correctness of the model. A correct prediction is when the target is a substring of the prediction.
    """
    assert len(predictions) == len(targets), "Predictions and targets must have the same length."

    correct = sum([tar.strip().lower() in pred.strip().lower() for (pred, tar) in zip(predictions, targets)]) / len(predictions)
    print(f"Correct: {correct}")

    return correct

def plot_accuracies(results_file):
    """
    Plot the accuracies of the model.
    """
    df = pd.read_csv(results_file)

    # filter out any NaN values
    df.dropna(inplace=True)

    # get list of fields that are not 'question', 'baseline', or 'answer'
    fieldnames = [field for field in df.columns if field not in ['question', 'baseline', 'answer']]

    results = {}
    for field in fieldnames:
        l = int(field.split('-')[0][1:])
        p = int(field.split('-')[1][1:])

        predictions = list(df[field])
        targets = list(df['answer'])
        # print(predictions, targets)

        results[(l, p)] = calculate_correct_wf(predictions, targets)
    
    layers = list({l for l, _ in results.keys()})
    perturbations = list({p for _, p in results.keys()})
    
    accuracies_per_perturbation = {p: [] for p in perturbations}
    for p in perturbations:
        for l in layers:
            accuracies_per_perturbation[p].append(results[(l, p)])
    
    accuracies_per_layer = {l: [] for l in layers}
    for l in layers:
        for p in perturbations:
            accuracies_per_layer[l].append(results[(l, p)])

    # Generate plots and save to file

    # plot accuracies per perturbation
    plt.figure(figsize=(10, 5))
    for p in perturbations:
        plt.plot(layers, accuracies_per_perturbation[p], label=f"{p} perturbations")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per perturbation")
    plt.legend()
    plt.savefig(results_file.replace(".csv", "_per_perturbation.png"))

    # plot accuracies per layer
    plt.figure(figsize=(10, 5))
    for l in layers:
        plt.plot(perturbations, accuracies_per_layer[l], label=f"Layer {l}")
    plt.xlabel("Perturbations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per layer")
    plt.legend()
    plt.savefig(results_file.replace(".csv", "_per_layer.png"))

