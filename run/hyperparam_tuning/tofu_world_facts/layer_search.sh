finetune_model_id=llama3-tofu-8B-epoch-0
base_model_id=meta-llama/Meta-Llama-3-8B-Instruct

baseline_results_file=results/TOFU/world_facts/layer_search/results.csv
results_file=$baseline_results_file

dataset_path=data/TOFU/world_facts_perturbed-extended.csv

echo "Saving results to $results_file"
# python utility_scripts/reorder_csv.py $results_file

num_perturbed=1

# for denom in $(seq 1.1 0.1 1.4)
for layer in {7..28}
do
    python abliterate_tofu.py \
        $baseline_results_file \
        --dataset_path $dataset_path \
        --base_model_id $base_model_id \
        --layer $layer \
        --num_perturbed $num_perturbed \
        --intervention_name L$layer-P1 \
        --results_file $results_file \
        --verbose \
        --finetune_model_path "aimonbc/$finetune_model_id" \

    python utility_scripts/reorder_csv.py $results_file
    python utility_scripts/generate_binary_results.py $results_file

done
