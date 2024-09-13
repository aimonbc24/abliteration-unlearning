results_file=results/PopQA/questions.csv
output_file=results/PopQA/baseline.csv

python abliterate_popqa.py $results_file --run_baseline --results_file $output_file --debug

python utility_scripts/reorder_csv.py $output_file