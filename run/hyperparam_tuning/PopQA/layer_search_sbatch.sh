#!/bin/bash
#SBATCH --partition=gpu-l40
#SBATCH --account=ark
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --job-name=eval
#SBATCH --output="./slurm/slurm-%J-%x.out"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aimonbc@cs.washington.edu

# Your training commands here

source activate open-instruct-updated

cat $0
echo "--------------------"

results_file="results/PopQA/pop_search/results.csv"

echo ""
echo "Saving results to $results_file"

for num_perturbed in $(seq 8 2 10)
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
            --perturbation_type 'random' \

        echo ""

        # reorder the csv file
        python utility_scripts/reorder_csv.py $results_file

        # compute accuracy
        python utility_scripts/generate_binary_results.py $results_file
    done
done
