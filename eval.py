import argparse
import inspect
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path

import accelerate
import diffusers
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.training_utils import EMAModel
from diffusers.utils import (check_min_version, is_accelerate_version,
                             is_tensorboard_available, is_wandb_available)
from diffusers.utils.import_utils import is_xformers_available
from packaging import version

from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from accelerate.state import AcceleratorState

from continual_diffusers.samplers.ddpm_pipeline import Compose_DDPMPipeline, Custom_DDPMPipeline,Custom_Text_DDPMPipeline
from continual_diffusers.samplers.energy_pipeline import Energy_DDPMPipeline, Energy_Text_DDPMPipeline

from continual_diffusers.models.utils import get_model_class
from eval_utils import get_grad_fn_mcmc_sampler
from safetensors.torch import load_file

logger = get_logger(__name__, log_level="INFO")

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


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-64",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="The number of images to generate for evaluation.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )

    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument(
        "--ema_inv_gamma",
        type=float,
        default=1.0,
        help="The inverse gamma value for the EMA decay.",
    )
    parser.add_argument(
        "--ema_power",
        type=float,
        default=3 / 4,
        help="The power value for the EMA decay.",
    )
    parser.add_argument(
        "--ema_max_decay",
        type=float,
        default=0.9999,
        help="The maximum decay magnitude for EMA.",
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
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo",
        action="store_true",
        help="Whether or not to create a private repository.",
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
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
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=10)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    # SAMPLER ARGUMENTS :
    parser.add_argument(
        "--mcmc_sampler",
        type=str,
        default=None,
        help="MCMC sampler for Continual Generative Modeling .",
        choices=["ULA", "UHA", "MALA", "CHA"],
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
        "--energy_score_type",
        type=str,
        default="dae",
        help="Energy score type for energy based training.",
        choices=["dae", "l2"],
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for Classifier-free-guidance.",
    )
    parser.add_argument(
        "--energy_guidance_scale",
        type=float,
        default=3,
        help="Energy guidance scale for Energy based inference for Adjusted Samplers MALA and CHA.",
    )
    parser.add_argument(
        "--energy_based_training",
        action="store_true",
        help="Whether to use energy based Inference.",
    )
    parser.add_argument(
        "--classifier_free_guidance",
        action="store_true",
        help="Whether to use classifier free guidance.",
    )
    parser.add_argument(
        "--variance_type",
        type=str,
        default="fixed_small",
        help="Variance type for Diffusion Model training.",
        choices=[
            "fixed_small",
            "fixed_small_log",
            "fixed_large",
            "fixed_large_log",
            "learned",
            "learned_range",
        ],
    )
    ## DATA ARGUMENTS
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=("The name of the Dataset to train on "),
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the data directory.",
        required=False,
    )
    parser.add_argument(
        "--uncond_p",
        type=float,
        default=0.05,
        help="Probability of Droppoing labels randomly for Classifier Free guidance Training.",
    )

    parser.add_argument(
        "--num_class_labels",
        type=int,
        default=None,
        help="Number of class labels for the entire Continual learning setup across tasks .",
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
        "--composition_pipeline",
        action="store_true",
        help="Whether to use composition pipeline for Gens.",
    )
    parser.add_argument(
        "--composition_labels",
        type=str,
        default=None,
        help="The labels for composition pipeline.",
    )
    parser.add_argument(
        "--text_conditioning",
        action="store_true",
        help="Whether to condition the replay buffer on text.",
    )

    parser.add_argument(
        "--clip_text_pretrained_path",
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
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=7200)
    )  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError(
                "Make sure to install tensorboard if you want to use it for logging during training."
            )

    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb


    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        # datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        # datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    if args.text_conditioning:
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            text_encoder = CLIPTextModel.from_pretrained(
                args.clip_text_pretrained_path,
                subfolder="text_encoder",
                revision=args.revision,
                variant=args.variant,
                cache_dir=args.cache_dir,
            )
        tokenizer = CLIPTokenizer.from_pretrained(
        args.clip_text_pretrained_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
        # Freeze the text encoder
        text_encoder.requires_grad_(False)
    else:
        text_encoder = None
        tokenizer = None

    model_cls = get_model_class(args)   
    config = model_cls.load_config(args.model_config_name_or_path)

    # update the config with energy_score_type value
    if args.energy_based_training:  # TODO: Better way of saving config
        config["energy_score_type"] = args.energy_score_type
        config["eval_mode"] = True

    model = model_cls.from_config(config)

    chkpt_model = load_file(
        args.model_config_name_or_path + "/diffusion_pytorch_model.safetensors"
    )
    model.load_state_dict(chkpt_model)

    print(model)
    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=model_cls,
            model_config=model.config,
        )
    else:
        ema_model = None

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )


    if args.text_conditioning:
        # Move text_encode to gpu and cast to weight_dtype
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(
        inspect.signature(DDPMScheduler.__init__).parameters.keys()
    )
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type,
            variance_type=args.variance_type,
        )
    else:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            variance_type=args.variance_type,
        )


    if args.mcmc_sampler:
        mcmc_sampler = get_grad_fn_mcmc_sampler(args,noise_scheduler)
    else:
        mcmc_sampler = None

    model = accelerator.prepare(model)


    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        if args.logger == "wandb":
            # TODO: Can add the cache dir hjere
            accelerator.init_trackers(
                args.project_name,
                init_kwargs={
                    "wandb": {
                        "name": args.run_name,
                        "save_code": args.save_code,
                        "dir": args.cache_dir,
                    }
                },
            )
        else:
            accelerator.init_trackers(run)


    # Generate sample images for visual inspection
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(model)

        if args.use_ema:
            ema_model.store(unet.parameters())
            ema_model.copy_to(unet.parameters())

        
        # Get the pipeline class based on the arguments
        if args.energy_based_training:
            if args.composition_pipeline and not args.text_conditioning:
                pipeline_cls = Compose_DDPMPipeline
            elif not args.composition_pipeline and args.text_conditioning:
                pipeline_cls = Energy_Text_DDPMPipeline
            elif args.composition_pipeline and args.text_conditioning:
                raise ValueError( "Composition pipeline and text conditioning are not supported together")
            else:
                pipeline_cls = Energy_DDPMPipeline
        else:
            if args.composition_pipeline and not args.text_conditioning:
                pipeline_cls = Compose_DDPMPipeline
            elif not args.composition_pipeline and args.text_conditioning:
                pipeline_cls = Custom_Text_DDPMPipeline
            elif args.composition_pipeline and args.text_conditioning:
                raise ValueError( "Composition pipeline and text conditioning are not supported together")
            else:
                pipeline_cls = Custom_DDPMPipeline

        if args.text_conditioning:
            pipeline = pipeline_cls(
                unet=unet,
                scheduler=noise_scheduler,
                tokenizer= tokenizer ,
                text_encoder=text_encoder,
                mcmc_sampler=mcmc_sampler,
        )
        else:
            pipeline = pipeline_cls(
                unet=unet,
                scheduler=noise_scheduler,
                mcmc_sampler=mcmc_sampler,  
            )

        generator = torch.Generator(device=pipeline.device).manual_seed(0)
        # run pipeline in inference (sample random noise and denoise)

        if args.num_class_labels is not None:
            if args.composition_pipeline:
                # args.composition_labels is like string "3,4,1", create sampled_class_labels torch tesnor

                sampled_class_labels = torch.tensor(
                    list(map(int, args.composition_labels.split(","))), dtype=torch.long
                ).to(pipeline.device)

            else:
                class_labels = torch.tensor(
                    torch.arange(args.num_class_labels), dtype=torch.long
                )
                sampled_class_labels = class_labels[
                    torch.randint(len(class_labels), (args.eval_batch_size,))
                ].to(pipeline.device)
        else:
            sampled_class_labels = None


        if args.text_conditioning:
            if args.validation_prompts:
                images = pipeline(
                    prompt=args.validation_prompts,
                    batch_size= args.eval_batch_size,
                    generator=  generator,
                    num_inference_steps=args.ddpm_num_inference_steps,
                    output_type="np",
                    classifier_free_guidance=args.classifier_free_guidance,
                    guidance_scale=args.guidance_scale,
                    mcmc_sampler_start_timestep=args.mcmc_sampler_start_timestep,
                ).images
        else:
            images = pipeline(
                generator=generator,
                batch_size=args.eval_batch_size,
                num_inference_steps=args.ddpm_num_inference_steps,
                output_type="np",
                class_labels=sampled_class_labels,
                classifier_free_guidance=args.classifier_free_guidance,
                guidance_scale=args.guidance_scale,
                mcmc_sampler_start_timestep=args.mcmc_sampler_start_timestep,
            ).images

        if args.use_ema:
            ema_model.restore(unet.parameters())

        # denormalize the images and save to tensorboard
        images_processed = (images * 255).round().astype("uint8")

        if args.logger == "tensorboard":
            if is_accelerate_version(">=", "0.17.0.dev0"):
                tracker = accelerator.get_tracker("tensorboard", unwrap=True)
            else:
                tracker = accelerator.get_tracker("tensorboard")
            tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2))
        elif args.logger == "wandb":
            # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
            accelerator.get_tracker("wandb").log(
                {"test_samples": [wandb.Image(img) for img in images_processed]},
                # TODO: step=global_step,
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
