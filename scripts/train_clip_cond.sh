#!/bin/bash

WANDB_CACHE_DIR="/cgm_logs/cache"
wandb online
# wandb offline 

data_dir="/data_cgm"
DATA_PATH_C10_cap="${data_dir}/cifar10_tasks_train_with_captions.npz"
export MODEL_NAME="CompVis/stable-diffusion-v1-4" # Huggingface checkpoint for pretrained CLIP text encoder

# Example Script for score-based training a Text-Conditioned DDPM model with Classifier-Free guidance on CIFAR-10 with Buffer Replay of Size 1000 Randomly sampled.

# The arg --text_conditioning is used to enable text conditioning in the model

python main.py --data_path $DATA_PATH_C10_cap \
                --dataset_name cifar10 \
                --clip_text_pretrained_path $MODEL_NAME \
                --resolution 32 \
                --num_epochs 50 \
                --text_conditioning \
                --output_dir /cgm_logs/output_cifar10_cfg_clip_guided \
                --cache_dir /cgm_logs/cache \
                --logger wandb \
                --train_batch_size 64 \
                --eval_batch_size 2 \
                --validation_prompts "This is a picture of airplane" "This is a picture of car" \
                --buffer_size 1000 \
                --classifier_free_guidance \
                --project_name continual-diffusers \
                --run_name cfg_cifar10_32_1k_buffer_Clip_guided
