
finetune_model_id=llama3-tofu-8B-epoch-0
base_model_id=meta-llama/Meta-Llama-3-8B-Instruct

baseline_results_file=results/entities/topic_qa/questions.csv
results_file=results/entities/topic_qa/baseline_results-tuned.csv

dataset_path=data/topic_qa_perturbed.csv

echo "Saving results to $results_file"
# python utility_scripts/reorder_csv.py $results_file

# layer=8
# num_perturbed=4

python abliterate_tofu.py \
    $baseline_results_file \
    --results_file $results_file \
    --dataset_path $dataset_path \
    --base_model_id $base_model_id \
    --inference_chat_template \
    --run_baseline \
    --verbose \
    --finetune_model_path "aimonbc/$finetune_model_id" \

python utility_scripts/reorder_csv.py $results_file --first_columns entity,question,answer,baseline
python utility_scripts/generate_binary_results.py $results_file