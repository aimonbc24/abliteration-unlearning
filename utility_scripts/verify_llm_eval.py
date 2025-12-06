import pandas as pd


results_file = 'results/entities/topic_qa/intervention_instruct_results.csv'
intervention = '2train-2test'

results_df = pd.read_csv(results_file)

binary_df = pd.read_csv(results_file.replace('.csv', '-llm-binary.csv'))

results_df = results_df[['entity','question','answer',intervention]]

results_df['llm_binary'] = binary_df[intervention]

results_df.to_csv(results_file.replace('.csv', f'-{intervention}.csv'), index=False)
print(f'Wrote {results_file.replace(".csv", f"-{intervention}.csv")}')