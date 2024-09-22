import argparse
import inspect
import logging
import math
import os
from datetime import timedelta
from pathlib import Path

import accelerate
import diffusers
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import (check_min_version, is_accelerate_version,
                             is_tensorboard_available, is_wandb_available)
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

from dataset import get_dataset
from models import CFGUNet2DModel, Energy_CFGUNet2DModel, Energy_UNet2DModel
from replay.buffer import BufferReplay
from samplers.ddpm_pipeline import Custom_DDPMPipeline
from samplers.energy_pipeline import Energy_DDPMPipeline
from train_utils import train_one_epoch
from train_utils.utils import (calculate_fisher_information,
                               get_model_parameters,initialize_except_conv_out)


logger = get_logger(__name__, log_level="INFO")


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
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=2,
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
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument(
        "--save_images_epochs",
        type=int,
        default=10,
        help="How often to save images during training.",
    )
    parser.add_argument(
        "--save_model_epochs",
        type=int,
        default=10,
        help="How often to save the model during training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.95,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-6,
        help="Weight decay magnitude for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer.",
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
        default=50,
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
        "--clip_grad_norm", type=float, default=10.0, help="Gradient clipping norm."
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
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
        "--energy_based_training",
        action="store_true",
        help="Whether to use energy based training.",
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
        required=True,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the data directory.",
        required=True,
    )
    parser.add_argument(
        "--uncond_p",
        type=float,
        default=0.05,
        help="Probability of Droppoing labels randomly for Classifier Free guidance Training.",
    )
    # Replay arguments
    parser.add_argument(
        "--buffer_type",
        type=str,
        default="reservoir",
        choices=["reservoir", "all_data", "no_data"],
        help="Type of replay buffer to use.",
    )
    parser.add_argument(
        "--buffer_size", type=int, default=1000, help="Size of the replay buffer."
    )
    parser.add_argument(
        "--retain_label_uniform_sampling",
        type=bool,
        default=True,
        help="Whether to retain label uniform sampling during Buffer Addition.",
    )
    parser.add_argument(
        "--num_class_labels",
        type=int,
        default=None,
        help="Number of class labels for the entire Continual learning setup across tasks .",
    )

    ## EWC Arguments
    parser.add_argument(
        "--ewc_loss",
        action="store_true",
        help="Whether to use Elastic Weight Consolidation for continual learning.",
    )
    parser.add_argument(
        "--lambda_ewc",
        type=float,
        default=0.1,
        help="Regularization strength for EWC penalty",
    )
    parser.add_argument(
        "--num_samples_ewc",
        type=int,
        default=100,
        help="Number of samples to use for EWC calculation.",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed that will be set in the training script.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.data_path is None:
        raise ValueError(
            "You must specify either a dataset name from the hub or a train data directory."
        )

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
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"),
                    UNet2DModel,
                    CFGUNet2DModel,
                    Energy_UNet2DModel,
                    Energy_CFGUNet2DModel,
                )
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            if args.classifier_free_guidance:
                if args.energy_based_training:
                    model_cls = Energy_CFGUNet2DModel
                else:
                    model_cls = CFGUNet2DModel
            else:
                if args.energy_based_training:
                    model_cls = Energy_UNet2DModel
                else:
                    model_cls = UNet2DModel

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = model_cls.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

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

    # Set the random seed for reproducibility
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

    if args.classifier_free_guidance:
        if args.energy_based_training:
            model_cls = Energy_CFGUNet2DModel
        else:
            model_cls = CFGUNet2DModel
    else:
        if args.energy_based_training:
            model_cls = Energy_UNet2DModel
        else:
            model_cls = UNet2DModel

    # Initialize the model
    if args.model_config_name_or_path is None:
        if args.energy_based_training:
            model = model_cls(
                sample_size=args.resolution,
                in_channels=3,
                num_class_embeds=(
                    args.num_class_labels if args.num_class_labels else None
                ),
                out_channels=(
                    3
                    if args.variance_type in ["fixed_small", "fixed_small_log"]
                    else 3 * 2
                ),
                layers_per_block=2,
                block_out_channels=(128, 128, 256, 256, 512, 512),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
                energy_score_type=args.energy_score_type,
            )
        else:
            model = model_cls(
                sample_size=args.resolution,
                in_channels=3,
                num_class_embeds=(
                    args.num_class_labels if args.num_class_labels else None
                ),
                out_channels=(
                    3
                    if args.variance_type in ["fixed_small", "fixed_small_log"]
                    else 3 * 2
                ),
                layers_per_block=2,
                block_out_channels=(128, 128, 256, 256, 512, 512),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            )
    else:
        config = model_cls.load_config(args.model_config_name_or_path)
        model = model_cls.from_config(config)

    # if args.energy_based_training:
    #     # Zero initialize all conv layers except the conv_out layer
    #     for name, module in model.named_modules():
    #         initialize_except_conv_out(name, module)

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

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Preprocessing the datasets and DataLoaders creation.

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    # Prepare everything with our `accelerator`.

    (
        model,
        optimizer,
    ) = accelerator.prepare(model, optimizer)

    if args.use_ema:
        ema_model.to(accelerator.device)

    dataset, num_tasks = get_dataset(args, transform)

    logger.info(f"Number of tasks for the dataset {args.dataset_name}: {num_tasks}")

    # Load the Replay Mechanism
    replay = BufferReplay(args, device=accelerator.device)

    # Define current task
    current_task = 1

    if args.resume_from_checkpoint:
        # Saved checkpoint : checkpoint-{global_step_for_task}-{task_num}-{epoch}
        path = args.resume_from_checkpoint
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(path)
        
        
        global_step = int(path.split("-")[1])
        current_task  = int(path.split("-")[2])
        first_epoch = int(path.split("-")[3])

        resume_step = global_step
    else:
        resume_step = None

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

    if args.ewc_loss:
        fisher_matrices = {}
        previous_parameters = {}

        # Set the fisher matrix and prev_parameter for the first task to be None
        fisher_matrices[1] = None
        previous_parameters[1] = None

    resume_once = False

    for task_num in range(current_task, num_tasks + 1):
        if resume_once:
            global_step = 0
            first_epoch = 0
        else:
            if not args.resume_from_checkpoint: 
                global_step = 0
                first_epoch = 0
            else:
                resume_once = True

        # Set the current task
        logger.info(
            f" Starting with : Task: {task_num} ,first_epoch: {first_epoch}, global_step: {global_step}"
        )

        # Set the current task with buffer, for the first task no buffer is added
        if task_num == 1:
            dataset.set_current_task(task_num, buffer=None)
        else:
            dataset.set_current_task(task_num, buffer=replay)

        logger.info(f"Task: {task_num} - Dataset size: {len(dataset)}")

        # Get the dataloader
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.dataloader_num_workers,
        )

        # Initialize the learning rate scheduler
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=(len(train_dataloader) * args.num_epochs),
        )

        # Prepare dataset, lr_scheduler with our `accelerator`.
        train_dataloader, lr_scheduler = accelerator.prepare(
            train_dataloader, lr_scheduler
        )

        total_batch_size = (
            args.train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        )
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        max_train_steps = args.num_epochs * num_update_steps_per_epoch

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(dataset)}")
        logger.info(f"  Num Epochs = {args.num_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_train_steps}")

        # Train
        for epoch in range(first_epoch, args.num_epochs):
            train_one_epoch(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                train_dataloader=train_dataloader,
                noise_scheduler=noise_scheduler,
                accelerator=accelerator,
                ema_model=ema_model,
                global_step=global_step,
                args=args,
                resume_step=resume_step,
                weight_dtype=weight_dtype,
                num_update_steps_per_epoch=num_update_steps_per_epoch,
                first_epoch=first_epoch,
                epoch=epoch,
                task_num=task_num,
                fisher_matrix=(
                    fisher_matrices[task_num - 1]
                    if args.ewc_loss and task_num > 1
                    else None
                ),
                previous_parameters=(
                    previous_parameters[task_num - 1]
                    if args.ewc_loss and task_num > 1
                    else None
                ),
            )

            accelerator.wait_for_everyone()

            # Generate sample images for visual inspection
            if accelerator.is_main_process:
                if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
                    unet = accelerator.unwrap_model(model)

                    if args.use_ema:
                        ema_model.store(unet.parameters())
                        ema_model.copy_to(unet.parameters())
                    if args.energy_based_training:
                        pipeline = Energy_DDPMPipeline(
                            unet=unet,
                            scheduler=noise_scheduler,
                        )
                    else:
                        pipeline = Custom_DDPMPipeline(
                            unet=unet,
                            scheduler=noise_scheduler,
                        )

                    generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)
                    # run pipeline in inference (sample random noise and denoise)

                    # Get the unique class labels for the task
                    if args.num_class_labels is not None:
                        class_labels = dataset.get_class_labels(task_num)
                        class_labels = torch.tensor(class_labels, dtype=torch.long)
                        sampled_class_labels = class_labels[
                            torch.randint(len(class_labels), (args.eval_batch_size,))
                        ].to(pipeline.device)
                    else:
                        sampled_class_labels = None

                    images = pipeline(
                        generator=generator,
                        batch_size=args.eval_batch_size,
                        num_inference_steps=args.ddpm_num_inference_steps,
                        output_type="np",
                        class_labels=sampled_class_labels,
                        classifier_free_guidance=args.classifier_free_guidance,
                        guidance_scale=args.guidance_scale,
                    ).images

                    if args.use_ema:
                        ema_model.restore(unet.parameters())

                    # denormalize the images and save to tensorboard
                    images_processed = (images * 255).round().astype("uint8")

                    if args.logger == "tensorboard":
                        if is_accelerate_version(">=", "0.17.0.dev0"):
                            tracker = accelerator.get_tracker(
                                "tensorboard", unwrap=True
                            )
                        else:
                            tracker = accelerator.get_tracker("tensorboard")
                        tracker.add_images(
                            "test_samples",
                            images_processed.transpose(0, 3, 1, 2),
                            epoch,
                        )
                    elif args.logger == "wandb":
                        # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
                        accelerator.get_tracker("wandb").log(
                            {
                                "test_samples": [
                                    wandb.Image(img) for img in images_processed
                                ],
                                "epoch": epoch,
                            },
                            # TODO: step=global_step,
                        )

                if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                    # save the model
                    unet = accelerator.unwrap_model(model)

                    if args.use_ema:
                        ema_model.store(unet.parameters())
                        ema_model.copy_to(unet.parameters())

                    if args.energy_based_training:
                        pipeline = Energy_DDPMPipeline(
                            unet=unet,
                            scheduler=noise_scheduler,
                        )
                    else:
                        pipeline = Custom_DDPMPipeline(
                            unet=unet,
                            scheduler=noise_scheduler,
                        )
                    pipeline.save_pretrained(args.output_dir)

                    if args.use_ema:
                        ema_model.restore(unet.parameters())

                    if args.push_to_hub:
                        upload_folder(
                            repo_id=repo_id,
                            folder_path=args.output_dir,
                            commit_message=f"Epoch {epoch}",
                            ignore_patterns=["step_*", "epoch_*"],
                        )

        if (
            task_num < num_tasks
        ):  # Add the task data to the buffer replay, no need to add for the last task
            if args.ewc_loss:
                # get fisher information for the task

                fisher_matrices[task_num] = calculate_fisher_information(
                    args,
                    model,
                    train_dataloader,
                    accelerator,
                    noise_scheduler,
                    weight_dtype,
                )
                previous_parameters[task_num] = get_model_parameters(model)

            replay.add_task_data(dataset.task_data[task_num])

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
