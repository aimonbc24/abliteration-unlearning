import pandas as pd

def delete_column(df, column_name):
    return df.drop(column_name, axis=1)

file = "results/unlearning_results_llama3-tofu-8B-epoch-4.csv"
df = pd.read_csv(file)
df = delete_column(df, "finetune")
df = delete_column(df, "epoch-2")
df.to_csv(file, index=False)

print(df.columns)