
finetune_model_id=llama3-tofu-8B-epoch-0
base_model_id=meta-llama/Meta-Llama-3-8B-Instruct

baseline_results_file=results/entities/intervention_instruct_results.csv
results_file=results/entities/intervention_instruct_results.csv

dataset_path=data/topic_qa/perturbed.csv

echo "Saving results to $results_file"
# python utility_scripts/reorder_csv.py $results_file

num_perturbed=1
alpha=3.5
layer=10

num_train=2
num_test=2

python abliterate_entities.py \
    $baseline_results_file \
    --results_file $results_file \
    --dataset_path $dataset_path \
    --base_model_id $base_model_id \
    --layer $layer \
    --num_perturbed $num_perturbed \
    --alpha $alpha \
    --intervention_name "${num_train}train-${num_test}test" \
    --num_train $num_train \
    --num_test $num_test \
    --verbose \
    --inference_chat_template \
    # --finetune_model_path "aimonbc/$finetune_model_id" \

python utility_scripts/reorder_csv.py $results_file --first_columns entity,question,answer,baseline
python utility_scripts/generate_binary_results-copy.py $results_file