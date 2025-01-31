import pandas as pd
import argparse

def accuracy_by_entity(df, treatment_columns):
    results_df = pd.DataFrame(columns=['entity'] + treatment_columns)
    results_df['entity'] = df['entity'].unique()
    
    for entity in results_df['entity']:
        entity_df = df[df['entity'] == entity]
        for treatment in treatment_columns:
            accuracy = entity_df[treatment].sum() / len(entity_df)
            results_df.loc[results_df['entity'] == entity, treatment] = round(accuracy, 2)

    return results_df.sort_values(by='entity')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Calculate the accuracy by entity.")
    argparser.add_argument("binary_results_file", type=str, help="Path to the binary results file.")
    
    args = argparser.parse_args()

    df = pd.read_csv(args.binary_results_file)

    treatment_columns = [col for col in df.columns if col not in ['question', 'answer', 'index', 'entity']]

    results_df = accuracy_by_entity(df, treatment_columns)

    results_file = args.binary_results_file.replace("-binary.csv", "-accuracy-by-entity.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\nAccuracy by entity table saved to:\n{results_file}")