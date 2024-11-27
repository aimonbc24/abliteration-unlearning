
baseline_results_file=results/PopQA/qa_template/results.csv
results_file=results/PopQA/qa_template/results.csv

dataset_path=data/PopQA/dataset_relevant_perturbed_pop_search.csv

echo -e "\nSaving results to $results_file\n"
# python utility_scripts/reorder_csv.py $results_file

num_perturbed=4
layer=8

python abliterate_tofu.py \
    $baseline_results_file \
    --dataset_path $dataset_path \
    --layer $layer \
    --num_perturbed $num_perturbed \
    --intervention_name L$layer-P$num_perturbed \
    --results_file $results_file \
    --verbose \
    # --debug 20 \

python utility_scripts/reorder_csv.py $results_file

python utility_scripts/generate_binary_results.py $results_file
