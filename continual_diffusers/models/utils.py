from .unets import CFGUNet2DModel, Energy_CFGUNet2DModel, Energy_UNet2DModel,Energy_Unet2DConditional
from diffusers.models.unets import UNet2DConditionModel, UNet2DModel


def get_model_class(args):
    """Determine the appropriate model class based on args."""
    if args.classifier_free_guidance:
        if args.energy_based_training:
            return Energy_Unet2DConditional if args.text_conditioning else Energy_CFGUNet2DModel
        else:
            return UNet2DConditionModel if args.text_conditioning else CFGUNet2DModel
    else:
        if args.energy_based_training:
            return Energy_Unet2DConditional if args.text_conditioning else Energy_UNet2DModel
        else:
            return UNet2DConditionModel if args.text_conditioning else UNet2DModel


def initialize_model(args, model_cls):
    """Initialize the model with the appropriate parameters based on args."""
    common_params = {
        'sample_size': args.resolution,
        'in_channels': 3,
        'out_channels': 3 if args.variance_type in ["fixed_small", "fixed_small_log"] else 6,
    }

    if args.model_config_name_or_path is not None:
        # Load model from config if provided
        config = model_cls.load_config(args.model_config_name_or_path)
        return model_cls.from_config(config)

    if args.energy_based_training and not args.text_conditioning:
        return model_cls(
            **common_params,
            num_class_embeds=args.num_class_labels if args.num_class_labels else None,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"
            ),
            energy_score_type=args.energy_score_type,
        )
    elif args.text_conditioning:
        if args.energy_based_training:
            return model_cls(
                **common_params,
                attention_head_dim=8,
                act_fn="silu",
                downsample_padding=1,
                flip_sin_to_cos=True,
                freq_shift=0,
                layers_per_block=2,
                mid_block_scale_factor=1,
                norm_eps=1e-05,
                norm_num_groups=32,
                cross_attention_dim=768,
                block_out_channels=(320, 640, 1280, 1280),
                down_block_types=(
                    "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"
                ),
                up_block_types=(
                    "UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"
                ),
                energy_score_type=args.energy_score_type ,
            )
        else:
            return model_cls(
                **common_params,
                attention_head_dim=8,
                act_fn="silu",
                downsample_padding=1,
                flip_sin_to_cos=True,
                freq_shift=0,
                layers_per_block=2,
                mid_block_scale_factor=1,
                norm_eps=1e-05,
                norm_num_groups=32,
                cross_attention_dim=768,
                block_out_channels=(320, 640, 1280, 1280),
                down_block_types=(
                    "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"
                ),
                up_block_types=(
                    "UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"
                ),
            )
    else:
        return model_cls(
            **common_params,
            num_class_embeds=args.num_class_labels if args.num_class_labels else None,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"
            ),
        )


def load_model(args):
    """Main function to select and initialize the model based on args."""
    model_cls = get_model_class(args)
    model = initialize_model(args, model_cls)
    return model,model_cls
