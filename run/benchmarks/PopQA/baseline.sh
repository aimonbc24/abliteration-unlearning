results_file="results/PopQA/pop_search/questions.csv"
output_file="results/PopQA/pop_search/baseline.csv"

python abliterate_popqa.py $results_file --run_baseline --results_file $output_file

python utility_scripts/reorder_csv.py $output_file