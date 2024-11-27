# run hyperparameter search for unlearning method on 'layer' parameter

baseline_results_file=results/TOFU/world_facts/baseline_results_filtered.csv
results_file=results/TOFU/world_facts/popqa_script_results.csv

dataset=world_facts_perturbed-extended

echo "Saving results to $results_file"
# python utility_scripts/reorder_csv.py $results_file

num_perturbed=4
layer=8

python abliterate_popqa.py \
    $baseline_results_file \
    --dataset_name $dataset \
    --layer $layer \
    --num_perturbed $num_perturbed \
    --intervention_name L$layer-P$num_perturbed \
    --results_file $results_file \
    --verbose

python utility_scripts/reorder_csv.py $results_file

python utility_scripts/generate_binary_results.py $results_file
