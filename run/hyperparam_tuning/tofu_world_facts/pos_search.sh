# run hyperparameter search for unlearning method on 'layer' parameter

results_file=results/world_facts/position_search/results.csv
dataset=world_facts_perturbed

echo "Saving results to $results_file"
python utility_scripts/reorder_csv.py $results_file

num_perturbed=4
layer=8

for pos in {-2..-1}
do
    echo "Running abliteration for pos $pos"

    python abliterate_tofu.py $results_file --dataset_name $dataset --pos $pos --layer $layer --num_perturbed $num_perturbed --intervention_name L$layer-Pos$pos --results_file $results_file
    python utility_scripts/reorder_csv.py $results_file
done
