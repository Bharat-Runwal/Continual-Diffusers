#!/bin/bash

# Path to the trained model
model_path_c10_score="/cgm_logs/output_cifar10_cfg_score/unet"

# During generation, we will randomly choose batch size number of labels from the num_class_labels to generate images for.


# Generation samples from Score-based model
# Supported MCMC Sampler : ULA, UHA 

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
                --run_name evals_c10_score_cosine_Score_ULA \
                --energy_based_inference \
                --mcmc_sampler ULA \
                --num_samples_per_step 10 \
                --step_sizes_multiplier 0.00015 \
                --mcmc_sampler_start_timestep 50




# Generation samples from Energy-based model with MALA sampler

# Supported MCMC Sampler : ULA, MALA, UHA, CHA

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
                --run_name evals_c10_DAE_cosine_MALA \
                --mcmc_sampler MALA \
                --num_samples_per_step 10 \
                --step_sizes_multiplier 0.00015 \
                --mcmc_sampler_start_timestep 50




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
                    --caption_column text \
                    --output_dir /cgm_logs/out_UHA_SCORE_LDM \
                    --cache_dir /cgm_logs/new_cgm/cache \
                    --report_to wandb \
                    --project_name continual-diffusers \
                    --tracker_project_name continual-diffusers \
                    --run_name out_UHA_SCORE_LDM \
                    --guidance_scale 7.5 \
                    --num_inference_steps 100 \
                    --validation_prompts "<Add A Prompt here based on your Dataset>" \
                    --mcmc_sampler UHA \
                    --num_samples_per_step 3 \
                    --num_leapfrog_steps 3 \
                    --damping_coeff 0.4 \
                    --step_sizes_multiplier 0.001 \
                    --mcmc_sampler_start_timestep 50