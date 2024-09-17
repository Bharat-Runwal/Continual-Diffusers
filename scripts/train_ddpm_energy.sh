#!/bin/bash

WANDB_CACHE_DIR="/cgm_logs/cache"
wandb online
# wandb offline 

data_dir="/data_cgm"
DATA_PATH_C10="${data_dir}/cifar10_tasks_train.npz"


# Example Script for energy-based training a DDPM model with Classifier-Free guidance on CIFAR-10 with Buffer Replay of Size 2000 (Retain Uniform Samples across labels).

python main.py --data_path $DATA_PATH \
                --dataset_name cifar10 \
                --guidance_scale 5 \
                --energy_based_training \
		        --energy_score_type l2 \
                --ddpm_beta_schedule "squaredcos_cap_v2" \
                --clip_grad_norm 10 \
                --ddpm_num_inference_steps 500 \
                --resolution 32 \
                --num_epochs 50 \
                --output_dir /cgm_logs/output_cifar10_cfg_energy_l2_clip_10_cosine \
                --cache_dir /cgm_logs/cache \
                --logger wandb \
                --train_batch_size 64 \
                --eval_batch_size 2 \
                --buffer_type reservoir \
                --buffer_size 2000 \
                --num_class_labels 10 \
                --classifier_free_guidance \
                --project_name continual-diffusers \
                --run_name CFG_C10_32_CLIP10_ENERGY_l2_FINAL_2k_COSINE
                

# For Different Energy score type , change the energy_score_type argument to one of the following: "l2", "dae"


# Example Script for Multi-GPU Training with Gradient Accumulation:
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
# export WORLD_SIZE=$((${SLURM_JOB_NUM_NODES:=1} * $SLURM_GPUS_ON_NODE))
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# accelerate launch  --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT  main.py --data_path $DATA_PATH_C10 \
#                 --dataset_name mscoco \
#                 --guidance_scale 5 \
#                 --energy_based_training \
#                 --energy_score_type dae \
#                 --ddpm_beta_schedule "squaredcos_cap_v2" \
#                 --clip_grad_norm 10 \
#                 --ddpm_num_inference_steps 500 \
#                 --gradient_accumulation_steps 2 \
#                 --resolution 32 \
#                 --num_epochs 50 \
#                 --output_dir /cgm_logs/output_CIFAR_cfg_energy_dae_clip_10_cosine_GRAD_ACC_2 \
#                 --cache_dir /cgm_logs/cache \
#                 --logger wandb \
#                 --train_batch_size 32 \
#                 --eval_batch_size 2 \
#                 --buffer_size 2000 \
#                 --num_class_labels 6 \
#                 --classifier_free_guidance \
#                 --project_name continual-diffusers \
#                 --run_name CFG_CIFAR_CLIP10_ENERGY_dae_2k_COSINE_GRAD_ACC_2