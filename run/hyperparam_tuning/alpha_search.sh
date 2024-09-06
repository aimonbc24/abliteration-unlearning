# run hyperparameter search for unlearning method on 'alpha' parameter

results_file=results/world_facts/alpha_search/results.csv
dataset=world_facts_perturbed

layer=8
num_perturbed=4

# loop over alphas between 1-2 with 0.1 increments
for alpha in $(seq 1 0.1 2)
do
    echo "Running abliteration for alpha $alpha"

    python abliterate_copy.py $results_file --dataset_name $dataset --layer $layer --num_perturbed $num_perturbed --alpha $alpha --intervention_name L$layer-P$num_perturbed-A$alpha --results_file $results_file --debug 
    python utility_scripts/reorder_csv.py $results_file
done

echo "Finished running abliteration for alpha search"

python utility_scripts/eval.py $results_file --alphas