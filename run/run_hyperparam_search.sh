# run hyperparameter search for unlearning method on 'layer' parameter and 'num_perturbed' parameter

results_file=results/world_facts/filtered_search_results.csv
dataset=world_facts_perturbed

echo "Saving results to $results_file"
python utility_scripts/reorder_csv.py $results_file

for layer in {5..19}
do
    # loop over all num_perturbed values 1-4
    for num_perturbed in {1..4}
    do
        echo "Running abliteration for layer $layer and num_perturbed $num_perturbed"
        # run hyperparameter search for unlearning method on 'layer' parameter and 'num_perturbed' parameter
        python abliterate_copy.py $results_file --dataset_name $dataset --layer $layer --num_perturbed $num_perturbed --intervention_name L$layer-P$num_perturbed --results_file $results_file --debug

        # reorder the csv file
        echo "Reordering csv file"
        python utility_scripts/reorder_csv.py $results_file
    done
done

# for layer in {5..31}
# do
#     echo "Running abliteration for layer $layer and num_perturbed $num_perturbed"
#     # run hyperparameter search for unlearning method on 'layer' parameter and 'num_perturbed' parameter
#     python abliterate_copy.py results/world_facts/hyperparam_search_results.csv --dataset_name world_facts_perturbed --layer $layer --num_perturbed $num_perturbed --intervention_name L$layer-P$num_perturbed --results_file results/world_facts/hyperparam_search_results.csv --debug 
# done
