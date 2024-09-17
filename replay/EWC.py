import torch


def get_model_parameters(model):
    return {name: param.clone().detach() for name, param in model.named_parameters()}


def compute_fisher_information(model, dataloader, device, diffusion, subset_size=1000):
    fisher_info = {
        name: torch.zeros_like(param) for name, param in model.named_parameters()
    }
    model.eval()

    total_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if total_samples >= subset_size:
                break
            labels, masks, reals = [x.to(device) for x in batch]
            model.zero_grad()

            timesteps = torch.randint(
                0, len(diffusion.betas) - 1, (reals.shape[0],), device=device
            )
            noise = torch.randn_like(reals, device=device)
            x_t = diffusion.q_sample(reals, timesteps, noise=noise)

            model_output = model(x_t, timesteps, y=labels, masks=masks)
            epsilon, _ = torch.split(model_output, model_output.shape[1] // 2, dim=1)

            loss = torch.nn.functional.mse_loss(epsilon, noise)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_info[name] += param.grad**2
            total_samples += reals.size(0)

    # Average Fisher Information
    for name in fisher_info:
        fisher_info[name] /= total_samples

    return fisher_info


import torch
import torch.nn.functional as F


def calculate_fisher_information(model, dataloader, accelerator, num_samples=None):
    fisher_information = {}

    # Initialize Fisher Information matrix only for parameters with requires_grad=True
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_information[name] = torch.zeros_like(param)

    model.eval()

    # Limit the number of samples if num_samples is specified
    total_samples = 0
    for batch in dataloader:
        inputs, targets = batch
        total_samples += inputs.size(0)

        if num_samples and total_samples > num_samples:
            break

        # Forward pass
        outputs = model(inputs)

        # Assuming a classification task with cross-entropy loss
        loss = F.cross_entropy(outputs, targets)

        # Backward pass to calculate gradients
        accelerator.backward(loss)

        # Accumulate squared gradients (which represent the Fisher Information)
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_information[name] += param.grad**2

    # Normalize by the number of samples used
    for name in fisher_information:
        fisher_information[name] /= min(total_samples, num_samples or total_samples)

    return fisher_information


# After finishing training on a task, calculate Fisher Information and save model parameters
if task_finished:
    fisher_information = calculate_fisher_information(model, dataloader, accelerator)
    prev_task_params = {
        name: param.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

# During training, add the EWC loss to the original loss
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, targets = batch

        # Forward pass
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)

        # Add EWC loss if applicable
        if task_num > 0:  # Assuming task_num tracks the current task index
            ewc_loss = calculate_ewc_loss(
                model, fisher_information, prev_task_params, lambda_ewc=args.lambda_ewc
            )
            loss += ewc_loss

        # Backward pass and optimization
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()


def calculate_ewc_loss(model, fisher_information, prev_task_params, lambda_ewc):
    ewc_loss = 0.0

    for name, param in model.named_parameters():
        if param.requires_grad and name in fisher_information:
            # Compute the EWC loss for each parameter with requires_grad=True
            ewc_loss += torch.sum(
                fisher_information[name] * (param - prev_task_params[name]) ** 2
            )

    return lambda_ewc * ewc_loss
