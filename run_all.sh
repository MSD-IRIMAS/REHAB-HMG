#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Define the Python script to be executed
TRAIN_VAE_SCRIPT="train_vae.py"
TRAIN_LABEL_VAE_SCRIPT="train_label_vae.py"

# Define the array of class indices
class_indices=(0 1 2 3 4)

# Loop through each class index and run the main training script for SVAE
echo "Running training script for SVAE on each class"
for class_index in "${class_indices[@]}"; do
    log_file="logs/svae_class_${class_index}.log"
    echo "Running training script for class index $class_index with SVAE"
    python3 $TRAIN_VAE_SCRIPT --generative-model SVAE --class_index $class_index  > $log_file 2>&1
done

# Run the training script for ASCVAE on the whole dataset
log_file="logs/ascvae_whole_set.log"
echo "Running training script for ASCVAE on the whole dataset"
python3 $TRAIN_VAE_SCRIPT --generative-model ASCVAE  > $log_file 2>&1

# Loop through each class index and run the main training script for VAE
echo "Running training script for VAE on each class"
for class_index in "${class_indices[@]}"; do
    log_file="logs/vae_class_${class_index}.log"
    echo "Running training script for class index $class_index with VAE"
    python3 $TRAIN_LABEL_VAE_SCRIPT --generative-model VAE --class_index $class_index  > $log_file 2>&1
done

# Run the training script for CVAEL on the whole dataset
log_file="logs/cvael_whole_set.log"
echo "Running training script for CVAEL on the whole dataset"
python3 $TRAIN_LABEL_VAE_SCRIPT --generative-model CVAEL  > $log_file 2>&1
