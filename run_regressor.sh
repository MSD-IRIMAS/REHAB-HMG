#!/bin/bash


mkdir -p reg_logs


REG_SCRIPT="train_regressor.py"

class_indices=(0 1 2 3 4)

# echo "Running training script for REG on each class"
# for class_index in "${class_indices[@]}"; do
#     log_file="reg_logs/reg_class_${class_index}.log"
#     echo "Running training script for class index $class_index with REG"
#     python3 $REG_SCRIPT  --class_index $class_index --regression_model REG > $log_file 2>&1
# done


echo "Running training script for STGCN on each class"
for class_index in "${class_indices[@]}"; do
    log_file="reg_logs/stgcn_class_${class_index}.log"
    echo "Running training script for class index $class_index with STGCN"
    python3 $REG_SCRIPT  --class_index $class_index   > $log_file 2>&1
done

