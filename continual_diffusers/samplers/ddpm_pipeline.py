# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
from diffusers.pipelines.pipeline_utils import (DiffusionPipeline,
                                                ImagePipelineOutput)
from diffusers.utils.torch_utils import randn_tensor


class Custom_DDPMPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, mcmc_sampler=None):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, mcmc_sampler=mcmc_sampler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        class_labels=None,
        classifier_free_guidance: Optional[bool] = None,
        guidance_scale: Optional[float] = 5.0,
        mcmc_sampler_start_timestep: Optional[int] = 50,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                *self.unet.config.sample_size,
            )

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            if classifier_free_guidance:
                # TODO : for the text embeddings the class_labels should be adjusted
                assert (
                    guidance_scale > 1.0
                ), "Guidance scale must be greater than 1.0 for Classifier Free Guidance"
                # Classifier-free-guidance
                # prepare the null labels and concatenate it with the class_labels for CFG Inference
                null_labels = torch.zeros_like(
                    class_labels
                )  # Class labels shape (eval_batch_size,)
                final_labels = torch.cat(
                    [class_labels, null_labels], dim=-1
                )  # Size of final labels: (eval_batch_size * 2,)
                # similarly create the masks for CFG True of len (class labels) and False of len (null labels)

                masks = torch.cat(
                    [torch.ones_like(class_labels), torch.zeros_like(null_labels)],
                    dim=-1,
                )  # shape (eval_batch_size * 2,)
                # Convert the masks to boolean
                masks = masks.bool()

                image = torch.cat(
                    [image] * 2
                )  # Duplicate the image for the two class labels # Shape : (2*batch_size, channels, height, width)
                model_output = self.unet(
                    image, t, class_labels=final_labels, mask=masks
                ).sample
                # split the output into conditional and unconditional
                out_size = (
                    model_output.size()
                )  # (2*batch_size, channels, height, width)
                pred_noise = model_output.reshape(
                    2, -1, *out_size[1:]
                )  # (2, batch_size, channels, height, width)
                noise_pred_eps_cond, noise_pred_eps_uncond = (
                    pred_noise[0],
                    pred_noise[1],
                )

                if model_output.shape[1] == image_shape[
                    1
                ] * 2 and self.scheduler.variance_type in ["learned", "learned_range"]:
                    # get eps, variance
                    noise_pred_eps_cond, noise_pred_var_cond = (
                        noise_pred_eps_cond.split(image.shape[1], dim=1)
                    )  # split the channels eps,var
                    noise_pred_eps_uncond, _ = noise_pred_eps_uncond.split(
                        image.shape[1], dim=1
                    )

                noise_pred = noise_pred_eps_uncond + guidance_scale * (
                    noise_pred_eps_cond - noise_pred_eps_uncond
                )
                if model_output.shape[1] == image_shape[
                    1
                ] * 2 and self.scheduler.variance_type in ["learned", "learned_range"]:
                    model_output = torch.cat([noise_pred, noise_pred_var_cond], dim=1)
                else:
                    model_output = noise_pred

            elif class_labels is not None and not classifier_free_guidance:
                model_output = self.unet(image, t, class_labels=class_labels).sample
            else:
                model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image[:batch_size], generator=generator
            ).prev_sample

            # TODO : Add correct attributes to pass to mcmc_sampler
            if self.mcmc_sampler is not None:
                # t : 1000 .. .. 950 . .. 50
                if t >= mcmc_sampler_start_timestep:
                    scalar = torch.sqrt(1 / (1 - self.scheduler.alphas_cumprod[t]))
                    if masks is not None:
                        kwargs = {
                            "mask": masks,
                            "scalar": scalar,
                            "class_labels": final_labels,
                        }
                    else:
                        kwargs = {"scalar": scalar, "class_labels": class_labels}

                    image = self.mcmc_sampler.step(image, t, self.unet, kwargs)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


class Compose_DDPMPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, mcmc_sampler=None):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, mcmc_sampler=mcmc_sampler)

    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        class_labels=None,
        classifier_free_guidance: Optional[bool] = None,
        guidance_scale: Optional[float] = 5.0,
        mcmc_sampler_start_timestep: Optional[int] = 50,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                *self.unet.config.sample_size,
            )

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            if classifier_free_guidance:
                # TODO : for the text embeddings the class_labels should be adjusted
                assert (
                    guidance_scale > 1.0
                ), "Guidance scale must be greater than 1.0 for Classifier Free Guidance"
                # Classifier-free-guidance

                null_labels = torch.tensor([0]).to(self.device)

                # Class labels shape (eval_batch_size,)
                final_labels = torch.cat(
                    [class_labels, null_labels], dim=-1
                )  # Size of final labels: (len(class_labels)+1 ,)
                # similarly create the masks for CFG True of len (class labels) and False of len (null labels)

                masks = torch.cat(
                    [torch.ones_like(class_labels), torch.zeros_like(null_labels)],
                    dim=-1,
                )  # shape (eval_batch_size * 2,)
                # Convert the masks to boolean
                masks = masks.bool()

                image = torch.cat(
                    [image] * 2
                )  # Duplicate the image for the two class labels # Shape : (2*batch_size, channels, height, width)

                model_output = []
                for label_, mask_ in zip(final_labels, masks):
                    model_output.append(
                        self.unet(
                            image[:1],
                            t,
                            class_labels=label_.unsqueeze(0),
                            mask=mask_.unsqueeze(0),
                        ).sample
                    )

                model_output = torch.cat(model_output, dim=0)

                # split the output into conditional and unconditional

                noise_pred_eps_cond, noise_pred_eps_uncond = (
                    model_output[:-1],
                    model_output[-1:],
                )

                if model_output.shape[1] == image_shape[
                    1
                ] * 2 and self.scheduler.variance_type in ["learned", "learned_range"]:
                    # get eps, variance
                    noise_pred_eps_cond, noise_pred_var_cond = (
                        noise_pred_eps_cond.split(image.shape[1], dim=1)
                    )  # split the channels eps,var
                    noise_pred_eps_uncond, _ = noise_pred_eps_uncond.split(
                        image.shape[1], dim=1
                    )

                    # TODO: Better way of composing when variance present
                    # In order to add variance back to the model_output , we average the variance across all conditionals in compositions
                    noise_pred_var_cond = torch.mean(
                        noise_pred_var_cond, dim=0, keepdim=True
                    )

                noise_pred = noise_pred_eps_uncond + guidance_scale * (
                    noise_pred_eps_cond - noise_pred_eps_uncond
                ).sum(dim=0, keepdim=True)
                if model_output.shape[1] == image_shape[
                    1
                ] * 2 and self.scheduler.variance_type in ["learned", "learned_range"]:
                    model_output = torch.cat([noise_pred, noise_pred_var_cond], dim=1)
                else:
                    model_output = noise_pred

            elif class_labels is not None and not classifier_free_guidance:
                model_output = self.unet(image, t, class_labels=class_labels).sample
            else:
                model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image[:batch_size], generator=generator
            ).prev_sample

            # TODO : Add correct attributes to pass to mcmc_sampler
            if self.mcmc_sampler is not None:
                # t : 1000 .. .. 950 . .. 50(from here mcmc sampler step)
                if t >= mcmc_sampler_start_timestep:
                    scalar = torch.sqrt(1 / (1 - self.scheduler.alphas_cumprod[t]))
                    if masks is not None:
                        kwargs = {
                            "mask": masks,
                            "scalar": scalar,
                            "class_labels": final_labels,
                        }
                    else:
                        kwargs = {"scalar": scalar, "class_labels": class_labels}

                    image = self.mcmc_sampler.step(image, t, self.unet, kwargs)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
