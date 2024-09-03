#!/bin/bash -l

# List of datasets  
datasets=("wtq" "tat")  
  
# List of perturbation methods  
perturb_methods=("no_shuffle" "escape_table" "mask_cell_all"\
                 "mask_cell_random" "relevant_row_shuffle")  
  
# Change to the 'code' directory  
cd code || { echo "Failed to change directory to 'code'. Exiting."; exit 1; }  
  
# Loop through datasets  
for dataset in "${datasets[@]}"; do  
  # Loop through perturbation methods  
  for p in "${perturb_methods[@]}"; do  
    # Decide on the perturb_method based on the condition  
    if [[ "$p" != "no_shuffle" && "$p" != "escape_table" ]]; then  
      perturb_method="${p}_${dataset}"  
    else  
      perturb_method="${p}"  
    fi  
    echo "$dataset, $perturb_method"  
    python generate_answers.py\
      --dataset_path "../dataset/02_rel/${dataset}_nonfactoid_dev.json"\
      --dataset_name "$dataset"\
      --model_path path/to/the/model\
      --perturb_method "$perturb_method"\
      --seed 33\
      --part 2\
      --model_type llm\
      --device cuda
  done  
done  
