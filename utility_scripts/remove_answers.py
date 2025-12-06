import pandas as pd

file = 'results/entities/synthetic_wikidata/entities/intervention_instruct_results.csv'
treatment = '10train-10test'

# Read the dataframe (for example, from a CSV file)
df = pd.read_csv(file)  # adjust the file path as needed

# Loop through all unique values in the 'entity' column
for entity in df['entity'].unique():
    # Get the indices for the first 10 rows for this entity
    indices = df[df['entity'] == entity].head(10).index
    # Set the treatment column to None for these rows
    df.loc[indices, treatment] = None

# Display the updated DataFrame
output_file = file.replace('.csv','-copy.csv')
df.to_csv(output_file, index=False)  # adjust the file path as needed

print(f"Saved updated DataFrame to: {output_file}")