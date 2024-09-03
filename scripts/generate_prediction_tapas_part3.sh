#!/bin/bash -l

# List of datasets  
datasets=("wtq" "tat")  
  
# List of perturbation methods  
perturb_methods=("no_shuffle_short_table" "no_shuffle_ans_change"\
                  "no_shuffle_ans_no_change" "no_shuffle_ori_table")
cd code || { echo "Failed to change directory to 'code'. Exiting."; exit 1; }  
# Loop through datasets  
for dataset in "${datasets[@]}"; do  
  # Loop through perturbation methods  
  for p in "${perturb_methods[@]}"; do  
    # Execute the Python script with the specified arguments
    echo "$dataset, $p"  
    python generate_answers.py\
      --dataset_path "../dataset/03_agg_com/${dataset}_nonfactoid_dev_${p}.json"\
      --dataset_name $dataset\
      --model_path path/to/the/model\
      --perturb_method $p\
      --seed 33\
      --part 3\
      --model_type tapas\
      --device cuda
  done  
done