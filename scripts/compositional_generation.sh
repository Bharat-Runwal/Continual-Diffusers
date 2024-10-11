#!/bin/bash


# DDPM Model Evals :

# Use argument : --composition_pipeline to enable compositional generation and --composition_labels to provide the labels to compose images for.

# Energy-Based : DAE Score
model_path_c10="/cgm_logs/new_cgm/out_c10_cosine_dae_energy/unet"
python eval.py \
                --model_config_name_or_path $model_path_c10 \
                --guidance_scale 2 \
                --resolution 32 \
                --energy_score_type dae \
                --ddpm_num_inference_steps 1000 \
                --output_dir /cgm_logs/evals_c10_DAE_cosine \
                --cache_dir /cgm_logs/cache \
                --logger wandb \
                --eval_batch_size 1 \
                --num_class_labels 6 \
                --ddpm_beta_schedule "squaredcos_cap_v2" \
                --classifier_free_guidance \
                --project_name continual-diffusers \
                --run_name Compose_score_DDPM_DAE_UHA \
                --energy_based_training \
                --composition_pipeline \
                --composition_labels "2,3" \
                --mcmc_sampler UHA \
                --num_samples_per_step 3 \
                --num_leapfrog_steps 3 \
                --damping_coeff 0.4 \
                --step_sizes_multiplier 0.001 \
                --mcmc_sampler_start_timestep 50



# LDM model Evals :

# Path to the trained model
model_path_ldm_c10_score="/cgm_logs/output_cifar10_cfg_score/unet"

export MODEL_NAME="CompVis/stable-diffusion-v1-4"

# Score-Based LDM : 

# Note: Provide the Prompts to compose using the "|" separator in the validation_prompts argument and use argument --composition_pipeline to enable compositional generation.

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
                    --validation_prompts "<1st prompt> | <2nd Prompt> " \
                    --composition_pipeline \
                    --mcmc_sampler UHA \
                    --num_samples_per_step 3 \
                    --num_leapfrog_steps 3 \
                    --damping_coeff 0.4 \
                    --step_sizes_multiplier 0.001 \
                    --mcmc_sampler_start_timestep 50