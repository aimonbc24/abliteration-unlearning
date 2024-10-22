
results_file="results/PopQA/pop_search-relevant/results.csv"

echo ""
echo "Saving results to $results_file"

for num_perturbed in $(seq 4 2 8)
do
    for layer in $(seq 4 2 16)
    do
        echo -e "\nRunning abliteration on layer $layer with $num_perturbed perturbations\n"
        
        # run hyperparameter search for unlearning method on 'layer' parameter and 'num_perturbed' parameter
        python abliterate_popqa.py \
            $results_file \
            --layer $layer \
            --num_perturbed $num_perturbed \
            --intervention_name L$layer-P$num_perturbed \
            --results_file $results_file \
            --perturbation_type 'relevant' \
            --verbose

        echo ""

        # reorder the csv file
        python utility_scripts/reorder_csv.py $results_file

        # compute accuracy
        python utility_scripts/generate_binary_results.py $results_file
    done
done
