import torch
from continual_diffusers.samplers import get_mcmc_sampler


def get_grad_fn_mcmc_sampler(args,noise_scheduler):
    def gradient(x_t, ts, model, kwargs):
        # for classifier-free-guidance
        original_shape = x_t.shape
        x_t = torch.cat([x_t] * 2)
        # check if key "mask" exsits in kwargs
        if "mask" in kwargs:
            model_output = model(
                x_t, ts, mask=kwargs["mask"], class_labels=kwargs["class_labels"]
            ).sample
        else:
            model_output = model(
                x_t, ts, class_labels=kwargs["class_labels"]
            ).sample


        # split the output into conditional and unconditional
        out_size = model_output.size()  # (2*batch_size, channels, height, width)
        pred_noise = model_output.reshape(
            2, -1, *out_size[1:]
        )  # (2, batch_size, channels, height, width)
        noise_pred_eps_cond, noise_pred_eps_uncond = pred_noise[0], pred_noise[1]

        if model_output.shape[1] == original_shape[1] * 2 and kwargs[
            "variance_type"
        ] in ["learned", "learned_range"]:
            # get eps, variance
            noise_pred_eps_cond, noise_pred_var_cond = noise_pred_eps_cond.split(
                x_t.shape[1], dim=1
            )  # split the channels eps,var
            noise_pred_eps_uncond, _ = noise_pred_eps_uncond.split(
                x_t.shape[1], dim=1
            )

        noise_pred = noise_pred_eps_uncond + args.guidance_scale * (
            noise_pred_eps_cond - noise_pred_eps_uncond
        )

        if model_output.shape[1] == original_shape[1] * 2 and kwargs[
            "variance_type"
        ] in ["learned", "learned_range"]:
            noise_pred = torch.cat([noise_pred, noise_pred_var_cond], dim=1)

        # Need to scale the gradients by coefficient to properly account for normalization in DSM loss + data contraction
        # where we have replaced ε_θ(x, t) with -σ_t * ∇_x f_θ(x + σ_t * ε)
        return -1 * kwargs["scalar"] * noise_pred

    def gradient_compose(x_t, ts, model, kwargs):
        # for classifier-free-guidance
        original_shape = x_t.shape
        x_t = torch.cat([x_t] * 2)

        # check if key "mask" exsits in kwargs
        if "mask" in kwargs:
            mask = kwargs["mask"]
            labels = kwargs["class_labels"]

            model_output = []
            for label_, mask_ in zip(labels, mask):
                model_output.append(
                    model(
                        x_t[:1],
                        ts,
                        mask=mask_.unsqueeze(0),
                        class_labels=label_.unsqueeze(0),
                    ).sample
                )

            model_output = torch.cat(model_output, dim=0)
        else:
            labels = kwargs["class_labels"]
            model_output = []
            for label_ in labels:
                model_output.append(
                    model(x_t[:1], ts, class_labels=label_.unsqueeze(0)).sample
                )

            model_output = torch.cat(model_output, dim=0)

        noise_pred_eps_cond, noise_pred_eps_uncond = (
            model_output[:-1],
            model_output[-1:],
        )

        if model_output.shape[1] == original_shape[1] * 2 and kwargs[
            "variance_type"
        ] in ["learned", "learned_range"]:
            # get eps, variance
            noise_pred_eps_cond, noise_pred_var_cond = noise_pred_eps_cond.split(
                x_t.shape[1], dim=1
            )  # split the channels eps,var
            noise_pred_eps_uncond, _ = noise_pred_eps_uncond.split(
                x_t.shape[1], dim=1
            )

        noise_pred = noise_pred_eps_uncond + args.guidance_scale * (
            noise_pred_eps_cond - noise_pred_eps_uncond
        ).sum(dim=0, keepdim=True)

        if model_output.shape[1] == original_shape[1] * 2 and kwargs[
            "variance_type"
        ] in ["learned", "learned_range"]:
            noise_pred = torch.cat([noise_pred, noise_pred_var_cond], dim=1)

        # Need to scale the gradients by coefficient to properly account for normalization in DSM loss + data contraction
        # where we have replaced ε_θ(x, t) with -σ_t * ∇_x f_θ(x + σ_t * ε)

        return -1 * kwargs["scalar"] * noise_pred

    def gradient_adjusted_compose(x_t, ts, model, kwargs):
        # for classifier-free-guidance
        original_shape = x_t.shape
        x_t = torch.cat([x_t] * 2)

        # check if key "mask" exsits in kwargs
        if "mask" in kwargs:
            mask = kwargs["mask"]
            labels = kwargs["class_labels"]

            model_output = []
            energy = []
            for label_, mask_ in zip(labels, mask):
                model_output_, energy_ = model(
                    x_t[:1],
                    ts,
                    mask=mask_.unsqueeze(0),
                    class_labels=label_.unsqueeze(0),
                    return_dict=False,
                )
                model_output.append(model_output_)
                energy.append(energy_)

            model_output = torch.cat(model_output, dim=0)
            energy = torch.cat(energy, dim=0)
        else:
            labels = kwargs["class_labels"]

            model_output = []
            energy = []
            for label_ in labels:
                model_output_, energy_ = model(
                    x_t[:1], ts, class_labels=label_.unsqueeze(0), return_dict=False
                )
                model_output.append(model_output_)
                energy.append(energy_)

            model_output = torch.cat(model_output, dim=0)
            energy = torch.cat(energy, dim=0)

        # split the output into conditional and unconditional
        noise_pred_eps_cond, noise_pred_eps_uncond = (
            model_output[:-1],
            model_output[-1:],
        )
        energy_cond, energy_uncond = energy[:-1], energy[-1:]

        total_energy = energy_uncond.sum() + args.energy_guidance_scale * (
            energy_cond.sum() - energy_uncond.sum()
        )

        if model_output.shape[1] == original_shape[1] * 2 and kwargs[
            "variance_type"
        ] in ["learned", "learned_range"]:
            # get eps, variance
            noise_pred_eps_cond, noise_pred_var_cond = noise_pred_eps_cond.split(
                x_t.shape[1], dim=1
            )  # split the channels eps,var
            noise_pred_eps_uncond, _ = noise_pred_eps_uncond.split(
                x_t.shape[1], dim=1
            )

            # TODO: Better way of composing when variance present
            # In order to add variance back to the model_output , we average the variance across all conditionals in compositions
            noise_pred_var_cond = torch.mean(
                noise_pred_var_cond, dim=0, keepdim=True
            )

        noise_pred = noise_pred_eps_uncond + args.guidance_scale * (
            noise_pred_eps_cond - noise_pred_eps_uncond
        ).sum(dim=0, keepdim=True)

        if model_output.shape[1] == original_shape[1] * 2 and kwargs[
            "variance_type"
        ] in ["learned", "learned_range"]:
            noise_pred = torch.cat([noise_pred, noise_pred_var_cond], dim=1)

        # Need to scale the gradients by coefficient to properly account for normalization in DSM loss + data contraction
        # where we have replaced ε_θ(x, t) with -σ_t * ∇_x f_θ(x + σ_t * ε)

        return (
            -1 * kwargs["scalar"] * total_energy,
            -1 * kwargs["scalar"] * noise_pred,
        )

    def gradient_adjusted(x_t, ts, model, kwargs):
        # for classifier-free-guidance

        original_shape = x_t.shape
        x_t = torch.cat([x_t] * 2)

        # check if key "mask" exsits in kwargs
        if "mask" in kwargs:

            model_output, energy = model(
                x_t,
                ts,
                mask=kwargs["mask"],
                class_labels=kwargs["class_labels"],
                return_dict=False,
            )
        else:
            model_output, energy = model(
                x_t, ts, class_labels=kwargs["class_labels"], return_dict=False
            )


        # split the output into conditional and unconditional
        out_size = model_output.size()  # (2*batch_size, channels, height, width)
        pred_noise = model_output.reshape(
            2, -1, *out_size[1:]
        )  # (2, batch_size, channels, height, width)
        noise_pred_eps_cond, noise_pred_eps_uncond = pred_noise[0], pred_noise[1]

        # do the same for energy (This is used for Adjusted sampling)
        energy = energy.reshape(2, -1, *energy.size()[1:])
        energy_cond, energy_uncond = energy[0], energy[1]
        total_energy = energy_uncond.sum() + args.guidance_scale * (
            energy_cond.sum() - energy_uncond.sum()
        )

        if model_output.shape[1] == original_shape[1] * 2 and kwargs[
            "variance_type"
        ] in ["learned", "learned_range"]:
            # get eps, variance
            noise_pred_eps_cond, noise_pred_var_cond = noise_pred_eps_cond.split(
                x_t.shape[1], dim=1
            )  # split the channels eps,var
            noise_pred_eps_uncond, _ = noise_pred_eps_uncond.split(
                x_t.shape[1], dim=1
            )

        noise_pred = noise_pred_eps_uncond + args.guidance_scale * (
            noise_pred_eps_cond - noise_pred_eps_uncond
        )

        if model_output.shape[1] == original_shape[1] * 2 and kwargs[
            "variance_type"
        ] in ["learned", "learned_range"]:
            noise_pred = torch.cat([noise_pred, noise_pred_var_cond], dim=1)

        # Need to scale the gradients by coefficient to properly account for normalization in DSM loss + data contraction
        # where we have replaced ε_θ(x, t) with -σ_t * ∇_x f_θ(x + σ_t * ε)
        return (
            -1 * kwargs["scalar"] * total_energy,
            -1 * kwargs["scalar"] * noise_pred,
        )

    mass_diag_sqrt = noise_scheduler.betas

    if args.mcmc_sampler in ["MALA", "CHA"]:
        if args.composition_pipeline:
            grad_fn_ = gradient_adjusted_compose
        else:
            grad_fn_ = gradient_adjusted
    else:
        if args.composition_pipeline:
            grad_fn_ = gradient_compose
        else:
            grad_fn_ = gradient

    mcmc_sampler = get_mcmc_sampler(
        sampler_name=args.mcmc_sampler,
        grad_fn=grad_fn_,
        step_sizes=(noise_scheduler.betas) * (args.step_sizes_multiplier),
        num_steps=args.ddpm_num_steps,
        num_samples_per_step=args.num_samples_per_step,  # 10 or 20
        damping_coeff=args.damping_coeff,  # .9
        mass_diag_sqrt=mass_diag_sqrt,
        num_leapfrog_steps=args.num_leapfrog_steps,
    )
    return mcmc_sampler