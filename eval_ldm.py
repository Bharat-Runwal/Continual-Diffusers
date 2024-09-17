"""
Code for finetuning latent diffusion models 
"""

#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import logging
import os
from contextlib import nullcontext
from pathlib import Path

import accelerate
import datasets
import diffusers
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (AutoencoderKL, DDPMScheduler, StableDiffusionPipeline,
                       UNet2DConditionModel)
from diffusers.utils import deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.training_utils import EMAModel

from huggingface_hub import create_repo
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

from models.unets import Energy_Unet2DConditional
from samplers.mcmc_samplers import get_mcmc_sampler
from samplers.sd_pipeline import (Compose_MCMC_StableDiffusionPipeline,
                                  Custom_StableDiffusionPipeline)

if is_wandb_available():
    import wandb


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.

logger = get_logger(__name__, log_level="INFO")



def log_validation(
    pipeline_cls,
    vae,
    text_encoder,
    tokenizer,
    unet,
    args,
    accelerator,
    weight_dtype,
    mcmc_sampler,
):
    logger.info("Running validation... ")

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
                args.validation_prompts[i],
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                mcmc_sampler=mcmc_sampler,
                guidance_scale=args.guidance_scale,
                weights=args.weights_prompt,
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


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation",
        type=float,
        default=0,
        help="The scale of input perturbation. Recommended 0.1.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="labels",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--offload_ema",
        action="store_true",
        help="Offload EMA model to CPU during training step.",
    )
    parser.add_argument(
        "--foreach_ema",
        action="store_true",
        help="Use faster foreach implementation of EMAModel.",
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )

    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--noise_offset", type=float, default=0, help="The scale of noise offset."
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100,
        help="The scale of noise offset.",
    )

    parser.add_argument(
        "--energy_score_type",
        type=str,
        default="dae",
        help="Energy score type for energy based training.",
        choices=["dae", "l2"],
    )
    parser.add_argument(
        "--energy_based_inference",
        action="store_true",
        help="Whether to use energy based model inference.",
    )

    # Continual Learning Arguments for WANDB
    parser.add_argument(
        "--run_name", default=None, type=str, help="A name to identify the run."
    )
    parser.add_argument(
        "--project_name",
        default="continual_ddpm_testing",
        type=str,
        help="The project name for WANDB.",
    )
    parser.add_argument(
        "--save_code",
        default=True,
        type=bool,
        help="Whether to save the code to WANDB.",
    )
    parser.add_argument(
        "--text_conditioning",
        action="store_true",
        help="Whether to condition the replay buffer on text.",
    )
    parser.add_argument(
        "--model_saved_chkpt",
        type=str,
        default=None,
        help="The path to the saved model checkpoint.",
    )
    # MCMC Sampler arguments
    parser.add_argument(
        "--mcmc_sampler",
        type=str,
        default=None,
        help="MCMC sampler for Continual Generative Modeling .",
        choices=["ULA", "UHA"],
    )
    parser.add_argument(
        "--mcmc_sampler_start_timestep",
        type=float,
        default=50,
        help="MCMC Sampler start timstep .",
    )
    parser.add_argument(
        "--step_sizes_multiplier",
        type=float,
        default=0.10,
        help="Step sizes multiplier for MCMC sampler.",
    )
    parser.add_argument(
        "--num_samples_per_step",
        type=int,
        default=1,
        help="Number of samples per step for MCMC sampler.",
    )
    parser.add_argument(
        "--damping_coeff",
        type=float,
        default=0.9,
        help="Damping coefficient for MCMC sampler.",
    )
    parser.add_argument(
        "--num_leapfrog_steps",
        type=int,
        default=3,
        help="Number of leapfrog steps for MCMC sampler.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for CFG.",
    )
    parser.add_argument(
        "--weights_prompt",
        type=str,
        default=None,
        required=False,
        help="Prompt for the weights compositions.",
    )
    parser.add_argument(
        "--composition_pipeline",
        action="store_true",
        help="Whether to use composition pipeline.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = (
            AcceleratorState().deepspeed_plugin
            if accelerate.state.is_initialized()
            else None
        )
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
            variant=args.variant,
            cache_dir=args.cache_dir,
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
            variant=args.variant,
            cache_dir=args.cache_dir,
        )

    if args.energy_based_inference:

        unet = Energy_Unet2DConditional.from_pretrained(
            args.model_saved_chkpt,
            subfolder="unet",
            revision=args.non_ema_revision,
            cache_dir=args.cache_dir,
            energy_score_type=args.energy_score_type,
        )
    else:

        unet = UNet2DConditionModel.from_pretrained(
            args.model_saved_chkpt,
            subfolder="unet",
            revision=args.non_ema_revision,
            cache_dir=args.cache_dir,
        )

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)


    # Create EMA for the unet.
    if args.use_ema:
        if args.energy_based_inference:
            unet_cls = Energy_Unet2DConditional
        else:
            unet_cls = UNet2DConditionModel

        ema_unet = unet_cls.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.revision,
            variant=args.variant,
        )
        ema_unet = EMAModel(
            ema_unet.parameters(),
            model_cls=unet_cls,
            model_config=ema_unet.config,
            foreach=args.foreach_ema,
        )
    else:
        ema_unet = None

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Prepare everything with our `accelerator`.
    unet = accelerator.prepare(unet)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if args.use_ema:
        if args.offload_ema:
            ema_unet.pin_memory()
        else:
            ema_unet.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        if args.report_to == "wandb":
            accelerator.init_trackers(
                args.tracker_project_name,
                tracker_config,
                init_kwargs={
                    "wandb": {
                        "name": args.run_name,
                        "save_code": args.save_code,
                        "dir": args.cache_dir,
                    }
                },
            )

        else:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)

    if args.mcmc_sampler is not None:
        mass_diag_sqrt = noise_scheduler.betas
        step_sizes = (noise_scheduler.betas) * (args.step_sizes_multiplier)
        mcmc_sampler = get_mcmc_sampler(
            sampler_name=args.mcmc_sampler,
            grad_fn=None,
            step_sizes=step_sizes,
            num_steps=len(step_sizes),
            num_samples_per_step=args.num_samples_per_step,  # 10 or 20
            damping_coeff=args.damping_coeff,  # .9
            mass_diag_sqrt=mass_diag_sqrt,
            num_leapfrog_steps=args.num_leapfrog_steps,
        )
        # Sampler with compositions (Currently only support one image generation, no batch support)
        if args.composition_pipeline:
            pipeline_cls = Compose_MCMC_StableDiffusionPipeline
        else:
            pipeline_cls = Custom_StableDiffusionPipeline

    else:
        pipeline_cls = StableDiffusionPipeline
        mcmc_sampler = None

    if args.validation_prompts is not None:
        if accelerator.is_main_process:
            if args.use_ema:
                ema_unet.copy_to(unet.parameters())

            log_validation(
                pipeline_cls,
                vae,
                text_encoder,
                tokenizer,
                unet,
                args,
                accelerator,
                weight_dtype,
                mcmc_sampler,
            )


if __name__ == "__main__":
    main()
