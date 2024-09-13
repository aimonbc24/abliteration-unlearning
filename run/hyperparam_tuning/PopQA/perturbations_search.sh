results_file=results/PopQA/perturbation_search/results.csv

echo "Saving results to $results_file"
python utility_scripts/reorder_csv.py $results_file

layer=8

for num_perturbed in {4..14}
do
    echo "Running abliteration for num_perturbed $num_perturbed"
    # run hyperparameter search for unlearning method on 'layer' parameter and 'num_perturbed' parameter
    python abliterate_popqa.py $results_file --layer $layer --num_perturbed $num_perturbed --intervention_name L$layer-P$num_perturbed --results_file $results_file --debug

    # reorder the csv file
    python utility_scripts/reorder_csv.py $results_file
done
