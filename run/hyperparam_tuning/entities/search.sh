
finetune_model_id=llama3-tofu-8B-epoch-0
base_model_id=meta-llama/Meta-Llama-3-8B-Instruct

baseline_results_file=results/entities/results.csv
results_file=$baseline_results_file

dataset_path=data/topic_qa_perturbed.csv

echo "Saving results to $results_file"
# python utility_scripts/reorder_csv.py $results_file

layer=8
num_perturbed=4

num_train=4
num_test=4

python abliterate_entities.py \
    $baseline_results_file \
    --results_file $results_file \
    --dataset_path $dataset_path \
    --base_model_id $base_model_id \
    --finetune_model_path "aimonbc/$finetune_model_id" \
    --layer $layer \
    --num_perturbed $num_perturbed \
    --intervention_name L$layer-P$num_perturbed \
    --num_train $num_train \
    --num_test $num_test \
    --verbose \

python utility_scripts/reorder_csv.py $results_file --first_columns entity,question,answer
python utility_scripts/generate_binary_results.py $results_file

