# run hyperparameter search for unlearning method on 'layer' parameter and 'num_perturbed' parameter

results_file=results/world_facts/broad_layer_perm_search/results.csv
dataset=world_facts_perturbed

echo "Saving results to $results_file"
python utility_scripts/reorder_csv.py $results_file

for layer in {5..11}
do
    for num_perturbed in {1..4}
    do
        echo "Running abliteration for layer $layer and num_perturbed $num_perturbed"
        # run hyperparameter search for unlearning method on 'layer' parameter and 'num_perturbed' parameter
        python abliterate_tofu.py $results_file --dataset_name $dataset --layer $layer --num_perturbed $num_perturbed --intervention_name L$layer-P$num_perturbed --results_file $results_file --debug

        # reorder the csv file
        python utility_scripts/reorder_csv.py $results_file
    done
done