# run hyperparameter search for unlearning method on 'layer' parameter and 'num_perturbed' parameter

layer=15
num_perturbed=4

# loop over all layers 0-31
# for layer in {5..31}
# do
#     # loop over all num_perturbed values 1-4
#     for num_perturbed in {1..4}
#     do
#         echo "Running abliteration for layer $layer and num_perturbed $num_perturbed"
#         # run hyperparameter search for unlearning method on 'layer' parameter and 'num_perturbed' parameter
#         python abliterate_copy.py results/world_facts/hyperparam_search_results.csv --dataset_name world_facts_perturbed --layer $layer --num_perturbed $num_perturbed --intervention_name L$layer-P$num_perturbed --results_file results/world_facts/hyperparam_search_results.csv --debug 
#     done
# done

# loop over all alphas in 1-5
for alpha in {1..5}
do
    echo "Running abliteration for alpha $alpha"
    # run hyperparameter search for unlearning method on 'layer' parameter and 'num_perturbed' parameter
    python abliterate_copy.py results/world_facts/hyperparam_search_results.csv --dataset_name world_facts_perturbed --layer $layer --num_perturbed $num_perturbed --alpha $alpha --intervention_name L$layer-P$num_perturbed-A$alpha --results_file results/world_facts/hyperparam_search_results.csv --debug 
done
