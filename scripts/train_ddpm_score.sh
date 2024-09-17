#!/bin/bash

WANDB_CACHE_DIR="/cgm_logs/cache"
wandb online
# wandb offline 

data_dir="/data_cgm"
DATA_PATH_C10="${data_dir}/cifar10_tasks_train.npz"


# Example Script for score-based training a DDPM model with Classifier-Free guidance on CIFAR-10 with Buffer Replay of Size 2000 (Retain Uniform Samples across labels).

python main.py --data_path $DATA_PATH_C10 \
                --dataset_name cifar10 \
                --resolution 32 \
                --num_epochs 50 \
                --ddpm_beta_schedule "squaredcos_cap_v2" \
                --output_dir /cgm_logs/output_cifar10_cfg \
                --cache_dir /cgm_logs/cache \
                --logger wandb \
                --train_batch_size 128 \
                --eval_batch_size 8 \
                --buffer_type reservoir \
                --buffer_size 2000 \
                --retain_label_uniform_sampling True \
                --num_class_labels 10 \
                --classifier_free_guidance \
                --project_name continual-diffusers \
                --run_name cfg_cifar10_32_2k_buffer

# For different replay methods , change the buffer_type argument to one of the following: "reservoir", "all_data", "no_data"
# For different replay sizes, change the buffer_size argument to the desired size.


# Example Script for Unconditional Score-Based Training a DDPM Model: 
# python main.py --data_path $DATA_PATH_C10 \
#                 --dataset_name cifar10 \
#                 --resolution 32 \
#                 --ddpm_beta_schedule "squaredcos_cap_v2" \
#                 --num_epochs 50 \
#                 --output_dir /cgm_logs/output_cifar10_uncond \
#                 --cache_dir /cgm_logs/cache \
#                 --logger wandb \
#                 --train_batch_size 128 \
#                 --eval_batch_size 8 \
#                 --buffer_type reservoir \
#                 --buffer_size 2000 \
#                 --project_name continual-diffusers \
#                 --run_name cfg_cifar10_32_2k_buffer_unconditional