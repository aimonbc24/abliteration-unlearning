base_model_id=meta-llama/Meta-Llama-3-8B-Instruct

baseline_results_file=results/TOFU/world_facts/models/Meta-Llama-3-8B-Instruct/layer_search_chat_results.csv
results_file=$baseline_results_file

dataset_path=data/TOFU/world_facts_perturbed-extended.csv

echo "Saving results to $results_file"
# python utility_scripts/reorder_csv.py $results_file

num_perturbed=1
alpha=3.5

# for layer between 6-20
for layer in {6..20}
do
    echo -e "\nLayer: $layer\n"

    python abliterate_tofu.py \
        $baseline_results_file \
        --dataset_path $dataset_path \
        --base_model_id $base_model_id \
        --layer $layer \
        --num_perturbed $num_perturbed \
        --alpha $alpha \
        --intervention_name L$layer-P$num_perturbed-A$alpha \
        --results_file $results_file \
        --verbose \
        --debug 20 \
        --use_chat_template \

    python utility_scripts/reorder_csv.py $results_file
    python utility_scripts/generate_binary_results.py $results_file

done


