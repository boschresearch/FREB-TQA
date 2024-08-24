#!/bin/bash -l

# List of datasets  
datasets=("wtq" "tat" "sqa" "wikisql")  
  
# List of perturbation methods  
perturb_methods=("no_shuffle" "row_shuffle" "col_shuffle" "target_row_random" "target_row_front"\  
                 "target_row_middle" "target_row_bottom" "target_col_random"\  
                 "target_col_front" "target_col_back" "transpose")  
cd code || { echo "Failed to change directory to 'code'. Exiting."; exit 1; }  
# Loop through datasets  
for dataset in "${datasets[@]}"; do  
  # Loop through perturbation methods  
  for p in "${perturb_methods[@]}"; do  
    # Execute the Python script with the specified arguments
    echo "$dataset, $p"  
    python generate_answers.py\
      --dataset_path "../dataset/01_str/${dataset}_factoid_dev.json"\
      --dataset_name $dataset\
      --model_path path/to/the/model\
      --perturb_method $p\
      --seed 33\
      --part 1\
      --model_type tapex\
      --device cuda
  done  
done  
