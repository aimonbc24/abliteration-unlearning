
baseline_results_file=results/PopQA/baseline_filtered.csv
results_file=results/PopQA/baseline_tofu_script.csv

dataset_path=data/PopQA/dataset_relevant_perturbed_pop_search.csv

echo -e "\nSaving results to $results_file\n"
# python utility_scripts/reorder_csv.py $results_file

num_perturbed=4
layer=8

python abliterate_tofu.py \
    $baseline_results_file \
    --dataset_path $dataset_path \
    --intervention_name baseline \
    --results_file $results_file \
    --run_baseline \
    --verbose \
    # --use_chat_template \

python utility_scripts/reorder_csv.py $results_file

# python utility_scripts/generate_binary_results.py $results_file
