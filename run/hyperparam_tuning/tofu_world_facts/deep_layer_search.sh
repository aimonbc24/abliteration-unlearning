# run hyperparameter search for unlearning method on 'layer' parameter

results_file=results/world_facts/deep_layer_search/results.csv
dataset=world_facts_perturbed

echo "Saving results to $results_file"
python utility_scripts/reorder_csv.py $results_file

layers=(5 6 7 8 9 10 11)
num_perturbed=4

for layer in ${layers[@]}
do
    echo "Running abliteration for layer $layer"

    python abliterate_tofu.py $results_file --dataset_name $dataset --layer $layer --num_perturbed $num_perturbed --intervention_name L$layer-P$num_perturbed --results_file $results_file
    python utility_scripts/reorder_csv.py $results_file
done

python utility_scripts/eval.py $results_file