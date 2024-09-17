#!/bin/bash

# Path to the trained model
model_path_c10_score="/cgm_logs/output_cifar10_cfg_score/unet"

# During generation, we will randomly choose batch size number of labels from the num_class_labels to generate images for.


# Generation samples from Score-based model
python eval.py \
                --model_config_name_or_path $model_path_c10_score \
                --guidance_scale 5 \
                --resolution 32 \
                --ddpm_num_inference_steps 1000 \
                --output_dir /cgm_logs/evals_c10_score_cosine \
                --cache_dir /cgm_logs/cache \
                --logger wandb \
                --eval_batch_size 8 \
                --num_class_labels 10 \
                --ddpm_beta_schedule "squaredcos_cap_v2" \
                --classifier_free_guidance \
                --project_name continual-diffusers \
                --run_name evals_c10_score_cosine_Score \



# Generation samples from Energy-based model
model_path_c10_energy_dae="/cgm_logs/output_cifar10_cfg_score/unet"
python eval.py \
                --model_config_name_or_path $model_path_c10_energy_dae \
                --guidance_scale 5 \
                --resolution 32 \
                --ddpm_num_inference_steps 1000 \
                --energy_based_inference \
                --energy_score_type dae \
                --output_dir /cgm_logs/evals_c10_DAE_cosine \
                --cache_dir /cgm_logs/cache \
                --logger wandb \
                --eval_batch_size 8 \
                --num_class_labels 10 \
                --ddpm_beta_schedule "squaredcos_cap_v2" \
                --classifier_free_guidance \
                --project_name continual-diffusers \
                --run_name evals_c10_DAE_cosine \



# LDM model Evals : 

# Path to the trained model
model_path_ldm_c10_score="/cgm_logs/output_cifar10_cfg_score/unet"

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# Score-Based LDM : 
python eval_ldm.py  --pretrained_model_name_or_path=$MODEL_NAME \
                    --model_saved_chkpt $model_path_ldm_c10_score \
                    --resolution=256 \
                    --text_conditioning \
                    --image_column image \
                    --caption_column labels \
                    --output_dir /cgm_logs/new_cgm_SD_LDM/output_score_LDM \
                    --cache_dir /cgm_logs/new_cgm/cache \
                    --report_to wandb \
                    --project_name continual-diffusers \
                    --tracker_project_name continual-diffusers \
                    --run_name eval_output_score_LDM \
                    --guidance_scale 7.5 \
                    --num_inference_steps 100 \
                    --validation_prompts "<Add A Prompt here based on your Dataset>" \


