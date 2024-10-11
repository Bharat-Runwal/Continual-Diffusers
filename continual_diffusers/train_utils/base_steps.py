import torch
import torch.nn.functional as F
from diffusers.training_utils import (compute_dream_and_update_latents,
                                      compute_snr)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def base_train_step(
    model,
    noise_scheduler,
    timesteps,
    noise,
    noisy_images,
    clean_images,
    batch,
    encoder_hidden_states,
    args
):
    """
    Calculate the loss for training.

    Args:
        model: The model to be trained.
        noisy_images: The noisy images.
        timesteps: The timesteps for the diffusion process.
        noise_scheduler: The noise scheduler.
        noise: The noise tensor.
        clean_images: The clean images.
        batch: The batch of data.
        args: Additional arguments.
    
    Returns:
        loss: The calculated loss.
    """

    if "labels" in batch and any(batch["labels"] != -1):
        labels = batch["labels"]
    else:
        labels = None

    if "masks" in batch and any(batch["masks"] != -1):
        mask = batch["masks"]
    else:
        mask = None

    # Handling 3 cases: Classifier Free Guidance, Conditional, Unconditional
    if args.classifier_free_guidance:
        model_output = model(
            noisy_images, timesteps, class_labels=labels, mask=mask
        ).sample
    elif labels is not None and not args.classifier_free_guidance:
        # No classifier free guidance
        model_output = model(noisy_images, timesteps, class_labels=labels).sample
    else:
        # Unconditional
        model_output = model(noisy_images, timesteps).sample

    if args.prediction_type == "epsilon":
        loss = F.mse_loss(
            model_output.float(), noise.float()
        )  
    elif args.prediction_type == "sample":
        alpha_t = _extract_into_tensor(
            noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
        )
        snr_weights = alpha_t / (1 - alpha_t)
        # use SNR weighting from distillation paper
        loss = snr_weights * F.mse_loss(
            model_output.float(), clean_images.float(), reduction="none"
        )
        loss = loss.mean()
    else:
        raise ValueError(f"Unsupported prediction type: {args.prediction_type}")

    return loss


def base_train_step_energy(
    model,
    noise_scheduler,
    timesteps,
    noise,
    noisy_images,
    clean_images,
    batch,
    encoder_hidden_states,
    args,

    ):
    """
    Calculate the loss for energy-based diffusion training.

    Args:
        model: The model to be trained.s
        noisy_images: The noisy images.
        timesteps: The timesteps for the diffusion process.
        noise: The noise tensor.
        batch: The batch of data.
        args: Additional arguments.
    Returns:
        loss: The calculated loss.
        energy_norm: The energy norm.
    """

    if "labels" in batch and any(batch["labels"] != -1):
        labels = batch["labels"]
    else:
        labels = None

    if "masks" in batch:
        mask = batch["masks"]  
    else:
        mask = None

    # Choose model output based on the presence of labels and mask

    # TODO: aten::_scaled_dot_product_efficient_attention_backward is not implemented for the double backward pass so disable "fused" kernels (https://github.com/pytorch/pytorch/issues/116350)
    # Replace this context manger with : torch.nn.attention.sdpa_kernel
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=True, enable_mem_efficient=False
    ):
        if args.classifier_free_guidance:
            # CFG
            model_output, energy_norm = model(
                noisy_images,
                timesteps,
                class_labels=labels,
                mask=mask,
                return_dict=False,
            )
        elif (labels is not None or mask is None) and not args.classifier_free_guidance:
            # Conditional
            model_output, energy_norm = model(
                noisy_images, timesteps, class_labels=labels, return_dict=False
            )
        else:
            # Unconditional
            model_output, energy_norm = model(
                noisy_images, timesteps, return_dict=False
            )


    if args.prediction_type == "epsilon":
        loss = F.mse_loss(
            model_output.float(), noise.float()
        )  
    elif args.prediction_type == "sample":
        alpha_t = _extract_into_tensor(
            noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
        )
        snr_weights = alpha_t / (1 - alpha_t)
        # use SNR weighting from distillation paper
        loss = snr_weights * F.mse_loss(
            model_output.float(), clean_images.float(), reduction="none"
        )
        loss = loss.mean()
    else:
        raise ValueError(f"Unsupported prediction type: {args.prediction_type}")

    return loss, energy_norm.detach().item()


def base_train_step_ldm(
    unet,
    noise_scheduler,
    timesteps,
    noise,
    noisy_latents,
    clean_images,
    target,
    encoder_hidden_states,
    args,
):
    """
    Calculate the loss for LDM training.

    Args:
        unet: The UNet model.
        noise_scheduler: The noise scheduler.
        timesteps: The timesteps for the diffusion process.
        noise: The noise tensor.
        noisy_latents: The noisy latents.
        target: The target tensor.
        encoder_hidden_states: The encoder hidden states for conditioning.
        args: Additional arguments.

    Returns:
        loss: The calculated loss.
    """

    if args.dream_training:
        noisy_latents, target = compute_dream_and_update_latents(
            unet,
            noise_scheduler,
            timesteps,
            noise,
            noisy_latents,
            target,
            encoder_hidden_states,
            args.dream_detail_preservation,
        )

    # Predict the noise residual and compute loss
    model_pred = unet(
        noisy_latents, timesteps, encoder_hidden_states, return_dict=False
    )[0]

    if args.snr_gamma is None:
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    else:
        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
        # This is discussed in Section 4.2 of the same paper.
        snr = compute_snr(noise_scheduler, timesteps)
        mse_loss_weights = torch.stack(
            [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1
        ).min(dim=1)[0]
        if noise_scheduler.config.prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr
        elif noise_scheduler.config.prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        loss = loss.mean()

    return loss


def base_train_step_ldm_energy(
    unet,
    noise_scheduler,
    timesteps,
    noise,
    noisy_latents,
    clean_images,
    target,
    encoder_hidden_states,
    args,
):
    """
    Calculate the loss for energy-based diffusion training.

    Args:
        unet: The UNet model.
        noise_scheduler: The noise scheduler.
        timesteps: The timesteps for the diffusion process.
        noise: The noise tensor.
        noisy_latents: The noisy latents.
        target: The target tensor.
        encoder_hidden_states: The encoder hidden states.
        args: Additional arguments.
    
    Returns:
        loss: The calculated loss.
        energy_norm: The energy norm.
    """

    if args.dream_training:
        noisy_latents, target = compute_dream_and_update_latents(
            unet,
            noise_scheduler,
            timesteps,
            noise,
            noisy_latents,
            target,
            encoder_hidden_states,
            args.dream_detail_preservation,
        )

    # TODO: aten::_scaled_dot_product_efficient_attention_backward is not implemented for the double backward pass so disable "fused" kernels (https://github.com/pytorch/pytorch/issues/116350)
    # Replace this context manger with : torch.nn.attention.sdpa_kernel
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=True, enable_mem_efficient=False
    ):
        # Predict the noise residual and compute loss
        model_pred, energy_norm = unet(
            noisy_latents, timesteps, encoder_hidden_states, return_dict=False
        )

    if args.snr_gamma is None:
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    else:
        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
        # This is discussed in Section 4.2 of the same paper.
        snr = compute_snr(noise_scheduler, timesteps)
        mse_loss_weights = torch.stack(
            [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1
        ).min(dim=1)[0]
        if noise_scheduler.config.prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr
        elif noise_scheduler.config.prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        loss = loss.mean()

    return loss, energy_norm.detach().item()


def calculate_ewc_loss(model, fisher_information, prev_task_params, lambda_ewc):
    """
    Calculate the Elastic Weight Consolidation (EWC) loss.
    """
    ewc_loss = 0.0

    for name, param in model.named_parameters():
        if param.requires_grad and name in fisher_information:
            # Compute the EWC loss for each parameter with requires_grad=True
            ewc_loss += torch.sum(
                fisher_information[name] * (param - prev_task_params[name]) ** 2
            )

    return lambda_ewc * ewc_loss
