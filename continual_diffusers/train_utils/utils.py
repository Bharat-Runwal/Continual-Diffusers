import torch
from accelerate.logging import get_logger

from continual_diffusers.train_utils.base_steps import base_train_step, base_train_step_energy
import torch.nn as nn
from continual_diffusers.dataset.dataset_ldm import preprocess_and_get_hf_dataset_curr_task

logger = get_logger(__name__, log_level="INFO")


def collate_fn(examples):
    pixel_values = torch.stack(
        [example["images"] for example in examples]
    )
    pixel_values = pixel_values.to(
        memory_format=torch.contiguous_format
    ).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"images": pixel_values, "input_ids": input_ids}


def get_task_dataloader(
    task_num, args, accelerator, replay=None,train_transform=None ,data_structure= None, dataset=None,tokenizer=None
):
    """
    Get the dataloader for the current task

    Args:
        task_num: The current task number
        data_structure: Dictionary containing the data for all tasks
        tokenizer: The tokenizer to be used
        args: Additional arguments
        accelerator: Accelerator for distributed training
        replay: Optional replay buffer
        train_transform: The training transform to be used

    Returns:
        train_dataloader: The dataloader for the current task
    """
    
    # Set the current task with buffer, for the first task no buffer is added
    if task_num == 1:
        if args.text_conditioning:
            dataset = preprocess_and_get_hf_dataset_curr_task(
                data_structure,
                task_num,
                tokenizer,
                args,
                accelerator,
                args.caption_column,
                args.image_column,
                replay=None,        # No replay for the first task
                train_transform=train_transform
            )
            # DataLoaders creation:
            train_dataloader = torch.utils.data.DataLoader(
                dataset,
                shuffle=True,
                collate_fn=collate_fn,
                batch_size=args.train_batch_size,
                num_workers=args.dataloader_num_workers,
            ) 
        else:
            dataset.set_current_task(task_num, buffer=None)
            # Get the dataloader
            train_dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.train_batch_size,
                shuffle=True,
                num_workers=args.dataloader_num_workers,
            )

    else:
        if args.text_conditioning:
            dataset = preprocess_and_get_hf_dataset_curr_task(
                data_structure,
                task_num,
                tokenizer,
                args,
                accelerator,
                args.caption_column,
                args.image_column,
                replay=replay,
                train_transform=train_transform
            )
            # DataLoaders creation:
            train_dataloader = torch.utils.data.DataLoader(
                dataset,
                shuffle=True,
                collate_fn=collate_fn,
                batch_size=args.train_batch_size,
                num_workers=args.dataloader_num_workers,
            ) 
        else:
            dataset.set_current_task(task_num, buffer=replay)
            # Get the dataloader
            train_dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.train_batch_size,
                shuffle=True,
                num_workers=args.dataloader_num_workers,
            )
    logger.info(f"Task: {task_num} - Dataset size: {len(dataset)}")

    return train_dataloader




# calculate gradient norm given parameters
def calculate_grad_norm(parameters):
    total_norm = 0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def get_model_parameters(model):
    return {name: param.clone().detach() for name, param in model.named_parameters()}


def calculate_fisher_information(
    args, model, dataloader, accelerator, noise_scheduler, weight_dtype
):
    fisher_information = {}

    # Initialize Fisher Information matrix only for parameters with requires_grad=True
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_information[name] = torch.zeros_like(param)

    model.eval()

    # Limit the number of samples if num_samples is specified
    total_samples = 0
    for batch in dataloader:

        clean_images = batch["images"].to(weight_dtype)

        total_samples += clean_images.size(0)
        if args.num_samples_ewc and total_samples > args.num_samples_ewc:
            break

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

        if args.energy_based_training:
            loss, energy_norm = base_train_step_energy(
                model, noisy_images, timesteps, noise, batch, weight_dtype, args
            )
        else:
            loss = base_train_step(
                model,
                noisy_images,
                timesteps,
                noise_scheduler,
                noise,
                clean_images,
                batch,
                weight_dtype,
                args,
            )

        # Backward pass to calculate gradients
        accelerator.backward(loss)

        # Accumulate squared gradients (which represent the Fisher Information)
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_information[name] += param.grad**2

        # zero the grad again
        model.zero_grad()
    # Normalize by the number of samples used
    for name in fisher_information:
        fisher_information[name] /= min(
            total_samples, args.num_samples_ewc or total_samples
        )

    return fisher_information


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)


def initialize_except_conv_out(module_name, module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        if module_name != "conv_out":  # Skip the conv_out layer
            zero_module(module)
        else:
            print(f"Skipping {module_name}")
