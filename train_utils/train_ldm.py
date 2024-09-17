"""
Training the latent diffusion model one epoch
"""

import os
from contextlib import nullcontext

import numpy as np
import torch
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline
from diffusers.utils import is_wandb_available

from tqdm import tqdm

from samplers.sd_pipeline import Energy_StableDiffusionPipeline

from .base_steps import base_train_step_ldm, base_train_step_ldm_energy
from .utils import calculate_grad_norm

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")


def log_validation(
    vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch
):
    logger.info("Running validation... ")
    if args.energy_based_training:
        pipeline_cls = Energy_StableDiffusionPipeline
    else:
        pipeline_cls = StableDiffusionPipeline

    pipeline = pipeline_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        cache_dir=args.cache_dir,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    images = []
    for i in range(len(args.validation_prompts)):
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(accelerator.device.type)

        with autocast_ctx:
            image = pipeline(
                args.validation_prompts[i], num_inference_steps=20, generator=generator
            ).images[0]

        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(
                "validation", np_images, epoch, dataformats="NHWC"
            )
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images


def train_one_epoch_ldm(
    args,
    accelerator,
    unet,
    vae,
    noise_scheduler,
    text_encoder,
    train_dataloader,
    optimizer,
    lr_scheduler,
    global_step,
    ema_unet,
    logger,
    weight_dtype,
    initial_global_step,
    epoch,
    tokenizer,
    task_num,
):
    """
    Training the latent diffusion model one epoch

    Args:
        args: The arguments containing training configurations.
        accelerator: The accelerator for mixed precision and distributed training.
        unet: The UNet model.
        vae: The VAE model.
        noise_scheduler: The noise scheduler.
        text_encoder: The text encoder.
        train_dataloader: The training dataloader.
        optimizer: The optimizer.
        lr_scheduler: The learning rate scheduler.
        global_step: The current global step.
        ema_unet: The exponential moving average of the UNet model.
        logger: The logger for logging information.
        weight_dtype: The data type for weights.
        initial_global_step: The initial global step for progress tracking.
        epoch: The current epoch.
        tokenizer: The tokenizer for text processing.
        task_num: Current Task Number for logging.
    
    """
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    if args.energy_based_training:
        train_step = base_train_step_ldm_energy
    else:
        train_step = base_train_step_ldm

    train_loss = 0.0

    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(unet):
            # Convert images to latent space
            latents = vae.encode(
                batch["pixel_values"].to(weight_dtype)
            ).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            if args.noise_offset:
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise += args.noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                )
            if args.input_perturbation:
                new_noise = noise + args.input_perturbation * torch.randn_like(noise)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            if args.input_perturbation:
                noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
            else:
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[
                0
            ]

            # Get the target for loss depending on the prediction type
            if args.prediction_type is not None:
                # set prediction_type of scheduler if defined
                noise_scheduler.register_to_config(prediction_type=args.prediction_type)

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )

            if args.energy_based_training:
                loss, energy_norm = train_step(
                    unet,
                    noise_scheduler,
                    timesteps,
                    noise,
                    noisy_latents,
                    target,
                    encoder_hidden_states,
                    args,
                )
            else:
                loss = train_step(
                    unet,
                    noise_scheduler,
                    timesteps,
                    noise,
                    noisy_latents,
                    target,
                    encoder_hidden_states,
                    args,
                )
            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps

            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)

            with torch.no_grad():
                grad_norms = calculate_grad_norm(unet.parameters())

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            if args.use_ema:
                if args.offload_ema:
                    ema_unet.to(device="cuda", non_blocking=True)
                ema_unet.step(unet.parameters())
                if args.offload_ema:
                    ema_unet.to(device="cpu", non_blocking=True)
            progress_bar.update(1)
            global_step += 1

            # TODO: step = global_step FIX IT
            if args.energy_based_training:
                accelerator.log(
                    {
                        "train_loss": train_loss,
                        "grad_norm": grad_norms,
                        "task_num": task_num,
                        "energy_norm": energy_norm,
                    }
                )
            else:
                accelerator.log(
                    {
                        "train_loss": train_loss,
                        "grad_norm": grad_norms,
                        "task_num": task_num,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                )

            train_loss = 0.0

            if global_step % args.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [
                            d for d in checkpoints if d.startswith("checkpoint")
                        ]
                        checkpoints = sorted(
                            checkpoints, key=lambda x: int(x.split("-")[1])
                        )

                  
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}-{task_num}-{epoch}"
                    )
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)

        if global_step >= args.max_train_steps:
            break

    if accelerator.is_main_process:
        if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
            if args.use_ema:
                # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                ema_unet.store(unet.parameters())
                ema_unet.copy_to(unet.parameters())
            log_validation(
                vae,
                text_encoder,
                tokenizer,
                unet,
                args,
                accelerator,
                weight_dtype,
                global_step,
            )
            if args.use_ema:
                # Switch back to the original UNet parameters.
                ema_unet.restore(unet.parameters())
