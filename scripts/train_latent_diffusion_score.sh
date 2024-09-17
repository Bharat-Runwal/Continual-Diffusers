#!/bin/bash

DATA_PATH_C10="/cgm_data/cifar10_tasks_train_LDM.npz"
export MODEL_NAME="CompVis/stable-diffusion-v1-4"

DATA_PATH="" # Add path to the dataset


# Score-Based Fine-Tuning of LDM (Pretrained : Stable Diffusion) 
python main_ldm.py   --pretrained_model_name_or_path=$MODEL_NAME \
                    --data_path $DATA_PATH \
                    --dataset_name dummy_example \
                    --resolution=256 --center_crop --random_flip \
                    --text_conditioning \
                    --train_batch_size=64 \
                    --image_column image \
                    --caption_column text \
                    --max_train_steps=15000 \
                    --learning_rate=1e-05 \
                    --max_grad_norm=1 \
                    --lr_scheduler="constant" --lr_warmup_steps=0 \
                    --gradient_accumulation_steps=1 \
                    --validation_epochs  2 \
                    --output_dir /cgm_logs/output_LDM_Task \
                    --cache_dir /cgm_logs/cache \
                    --report_to wandb \
                    --project_name continual-diffusers \
                    --tracker_project_name continual-diffusers \
                    --run_name SCORE_LDM_Task \
                    --validation_prompts "<Add A Prompt here based on your Dataset>"
