base_model_id=meta-llama/Meta-Llama-3-8B-Instruct
# base_model_id=meta-llama/Meta-Llama-3-8B

# baseline_results_file=results/TOFU/world_facts/models/llama3-tofu-8B-epoch-0/baseline_chat_results_filtered.csv
# results_file=results/TOFU/world_facts/models/llama3-tofu-8B-epoch-0/chat_inference.csv

baseline_results_file=results/TOFU/world_facts/models/Meta-Llama-3-8B-Instruct/baseline_chat_results_filtered.csv
results_file=results/TOFU/world_facts/models/Meta-Llama-3-8B-Instruct/chat_inference_results.csv

dataset_path=data/TOFU/world_facts_perturbed.csv

echo "Saving results to $results_file"

num_perturbed=1
alpha=3.5
layer=10

# python utility_scripts/reorder_csv.py $results_file

# python abliterate_tofu-paraphrased.py \
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
    --inference_chat_template \
    # --finetune_model_path "aimonbc/llama3-tofu-8B-epoch-0" \
    # --debug 20 \
    # --ICL \
    # --intervention_chat_template \
    # --use_chat_template \
    # --include-system-message \

python utility_scripts/reorder_csv.py $results_file
python utility_scripts/generate_binary_results.py $results_file
