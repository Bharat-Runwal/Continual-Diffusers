import os
import shutil

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from diffusers.models.unets import unet_2d
from tqdm import tqdm

from .base_steps import (base_train_step, base_train_step_energy,base_train_step_ldm,base_train_step_ldm_energy,
                         calculate_ewc_loss)
from .utils import calculate_grad_norm

logger = get_logger(__name__, log_level="INFO")


def train_one_epoch(
    model,
    optimizer,
    lr_scheduler,
    train_dataloader,
    noise_scheduler,
    accelerator,
    ema_model,
    global_step,
    epoch,
    first_epoch,
    resume_step,
    num_update_steps_per_epoch,
    weight_dtype,
    args,
    task_num,
    fisher_matrix=None,
    previous_parameters=None,
    text_encoder=None,
    uncond_token_ids=None,
):
    """Train the model for one epoch."""

    if args.text_conditioning:
        assert text_encoder is not None, "Text encoder must be provided for text conditioning."
        if args.energy_based_training:
            train_step = base_train_step_ldm_energy
        else:
            train_step = base_train_step_ldm
    else:
        if args.energy_based_training:
            train_step = base_train_step_energy
        else:
            train_step = base_train_step

    if args.ewc_loss:
        ewc_loss = None
    model.train()
    progress_bar = tqdm(
        total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description(f"Epoch {epoch} / Task: {task_num}")

    for step, batch in enumerate(train_dataloader):
        # Skip steps until we reach the resumed step
        if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
            if step % args.gradient_accumulation_steps == 0:
                progress_bar.update(1)
            continue

        clean_images = batch["images"].to(weight_dtype)


        # Sample noise that we'll add to the images
        noise = torch.randn(
            clean_images.shape, dtype=weight_dtype, device=clean_images.device
        )
        bsz = clean_images.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=clean_images.device,
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        if text_encoder is not None:
            input_ids_text = batch["input_ids"]
            if args.classifier_free_guidance:
                # Randomly replace uncond_prob of the input_ids with empty token id
                uncond_token_ids = uncond_token_ids.to(input_ids_text.device)
                empty_tokens = uncond_token_ids.repeat(input_ids_text.shape[0], 1)  # Shape: [batch_size, 77] (For CLIP)
                mask = torch.rand(input_ids_text.shape[0]) < args.uncond_p
                
                input_ids_text[mask] = empty_tokens[mask]
                
            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(input_ids_text, return_dict=False)[
                0
            ]

        with accelerator.accumulate(model):
            # Predict the noise residual and returns the loss
            if args.energy_based_training:
                loss, energy_norm = train_step(
                    model,
                    noise_scheduler,
                    timesteps,
                    noise,
                    noisy_images,
                    clean_images,
                    batch if not args.text_conditioning else noise,
                    encoder_hidden_states if args.text_conditioning else None,
                    args
                )
            else:
                loss = train_step(
                    model,
                    noise_scheduler,
                    timesteps,
                    noise,
                    noisy_images,
                    clean_images,
                    batch if not args.text_conditioning else noise,
                    encoder_hidden_states if args.text_conditioning else None,
                    args
                )

            if (
                args.ewc_loss
                and fisher_matrix is not None
                and previous_parameters is not None
            ):
                ewc_loss = calculate_ewc_loss(
                    model, fisher_matrix, previous_parameters, args.lambda_ewc
                )
                loss += ewc_loss
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            with torch.no_grad():
                grad_norms = calculate_grad_norm(model.parameters())

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            if args.use_ema:
                ema_model.step(model.parameters())
            progress_bar.update(1)
            global_step += 1

            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [
                            d for d in checkpoints if d.startswith("checkpoint")
                        ]
                        checkpoints = sorted(
                            checkpoints, key=lambda x: int(x.split("-")[1])
                        )

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        # if len(checkpoints) >= args.checkpoints_total_limit:
                        #     num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        #     removing_checkpoints = checkpoints[0:num_to_remove]

                        #     logger.info(
                        #         f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        #     )
                        #     logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        #     for removing_checkpoint in removing_checkpoints:
                        #         removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                        #         shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}-{task_num}-{epoch}"
                    )
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
        if args.energy_based_training:
            if args.ewc_loss and ewc_loss is not None:
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "task": task_num,
                    "energy_norm": energy_norm,
                    "grad_norm": grad_norms,
                    "ewc_loss": ewc_loss.detach().item(),
                }
            else:
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "task": task_num,
                    "energy_norm": energy_norm,
                    "grad_norm": grad_norms,
                }
        else:
            if args.ewc_loss and ewc_loss is not None:
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "task": task_num,
                    "grad_norm": grad_norms,
                    "ewc_loss": ewc_loss.detach().item(),
                }
            else:
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "task": task_num,
                    "grad_norm": grad_norms,
                }

        if args.use_ema:
            logs["ema_decay"] = ema_model.cur_decay_value
        progress_bar.set_postfix(**logs)
        accelerator.log(logs)  # TODO: step = global_step FIX IT

    progress_bar.close()
