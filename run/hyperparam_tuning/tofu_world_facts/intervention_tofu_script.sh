# run hyperparameter search for unlearning method on 'layer' parameter

finetune_model_id=llama3-tofu-8B-epoch-0
base_model_id=meta-llama/Meta-Llama-3-8B-Instruct

baseline_results_file=results/TOFU/world_facts/perturbations/results.csv
results_file=$baseline_results_file

dataset_path=data/TOFU/world_facts_perturbed-extended.csv

echo "Saving results to $results_file"
# python utility_scripts/reorder_csv.py $results_file

layer=8

for num_perturbed in {4..6}
do

    python abliterate_tofu.py \
        $baseline_results_file \
        --dataset_path $dataset_path \
        --base_model_id $base_model_id \
        --layer $layer \
        --num_perturbed $num_perturbed \
        --intervention_name L$layer-P$num_perturbed \
        --results_file $results_file \
        --verbose \
        --finetune_model_path "aimonbc/$finetune_model_id" \

    python utility_scripts/reorder_csv.py $results_file
    python utility_scripts/generate_binary_results.py $results_file

done


# for layer in 6 8 10 12
# do
#     for num_perturbed in {3..5}
#     do

#         python abliterate_tofu.py \
#             $baseline_results_file \
#             --dataset_path $dataset_path \
#             --base_model_id $base_model_id \
#             --layer $layer \
#             --num_perturbed $num_perturbed \
#             --intervention_name L$layer-P$num_perturbed \
#             --results_file $results_file \
#             --verbose \
#             --finetune_model_path "aimonbc/$finetune_model_id" \

#         python utility_scripts/reorder_csv.py $results_file
#         python utility_scripts/generate_binary_results.py $results_file

#     done
# done

