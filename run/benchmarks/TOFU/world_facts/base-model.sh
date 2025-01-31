# run hyperparameter search for unlearning method on 'layer' parameter

baseline_results_file=results/TOFU/world_facts/qa.csv
results_file=results/TOFU/world_facts/baselines/llama3-8b-qa-tuned/baseline_results.csv

dataset_path=data/TOFU/world_facts_perturbed-extended.csv
base_model_id=meta-llama/Meta-Llama-3-8B
 #"meta-llama/Meta-Llama-3-8B"

echo "Saving results to $results_file"

python abliterate_tofu.py \
    $baseline_results_file \
    --dataset_path $dataset_path \
    --results_file $results_file \
    --verbose \
    --run_baseline \
    --base_model_id $base_model_id \
    --finetune_model_path "aimonbc/llama3-8b-qa-tuned" \
    # --use_chat_template \
    # --finetune_model_path "aimonbc/llama3-tofu-8B-epoch-0" \

python utility_scripts/reorder_csv.py $results_file

python utility_scripts/filter_incorrect.py $results_file