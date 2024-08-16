import pandas as pd

df = pd.read_csv('results/world_facts_results.csv')



df['baseline'] = df['baseline'].str.split(": ").str[1]

df.to_csv('results/world_facts_results.csv', index=False)