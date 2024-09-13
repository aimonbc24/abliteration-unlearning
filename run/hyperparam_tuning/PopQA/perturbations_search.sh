# Exit immediately if any command fails
set -e

results_file=results/PopQA/perturbations_search/results.csv

echo "Saving results to $results_file"
python utility_scripts/reorder_csv.py $results_file

layer=8

# reorder the csv file
python utility_scripts/reorder_csv.py $results_file

for num_perturbed in $(seq 6 1 14)
do
    echo ""
    echo "Running abliteration with $num_perturbed perturbations"
    echo ""
    # run hyperparameter search for unlearning method on 'layer' parameter and 'num_perturbed' parameter
    python abliterate_popqa.py $results_file --layer $layer --num_perturbed $num_perturbed --intervention_name L$layer-P$num_perturbed --results_file $results_file --debug 20
    echo ""
    # reorder the csv file
    python utility_scripts/reorder_csv.py $results_file
done
