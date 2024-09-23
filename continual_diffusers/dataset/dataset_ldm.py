import logging
import pickle
import random

import numpy as np
import torch as th
from datasets import Dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_task_to_dataset(task):
    """
    convert a task given as a dictionary to Dataset class object

    Args:
        task: dictionary containing images, labels
    """
    return Dataset.from_dict(task)


def get_datasets(args):
    """
    Initialize the dataset based on the provided arguments and return the dataset instance and total number of tasks.

    Args:
        args: The arguments containing data path.

    Returns:
        tuple: A tuple containing the dataset instance Dict and the number of tasks.
    """
    try:
        data = np.load(args.data_path, allow_pickle=True)
        logger.info("Loading data from %s", args.data_path)

        if "data_structure" in data:
            # Assuming 'data_structure' is stored as a pickled dictionary
            data_structure = pickle.loads(data["data_structure"].item())
            total_tasks = len(data_structure)

            logger.info(f"Dataset successfully loaded with {total_tasks} tasks.")
            return data_structure, total_tasks
        else:
            logger.error("Data structure not found in the file.")
            return None, 0
    except FileNotFoundError:
        logger.error(f"File not found: {args.data_path}")
        return None, 0
    except KeyError:
        logger.error("Error in data structure keys.")
        return None, 0
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return None, 0


def preprocess(tokenizer, args, accelerator, dataset, caption_column, image_column):
    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            (
                transforms.CenterCrop(args.resolution)
                if args.center_crop
                else transforms.RandomCrop(args.resolution)
            ),
            (
                transforms.RandomHorizontalFlip()
                if args.random_flip
                else transforms.Lambda(lambda x: x)
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [
            Image.fromarray(np.array(image, dtype=np.uint8)).convert("RGB")
            for image in examples[image_column]
        ]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset = dataset.shuffle(seed=args.seed).select(
                range(args.max_train_samples)
            )
        # Set the training transforms
        train_dataset = dataset.with_transform(preprocess_train)

    return train_dataset


def preprocess_and_get_hf_dataset_curr_task(
    data_structure,
    task_id,
    tokenizer,
    args,
    accelerator,
    caption_column,
    image_column,
    replay=None,
):
    """
    Get the dataset for the current task

    Args:
        data_structure: Dictionary containing the data for all tasks
        task_id: The current task ID
        tokenizer: The tokenizer to be used
        args: Additional arguments
        accelerator: Accelerator for distributed training
        caption_column: The column containing captions
        image_column: The column containing images
        replay: Optional replay buffer

    Returns:
        dataset: Dataset class object
    """
    try:
        if replay is not None:
            combined_data = replay.get_combined_data(data_structure[task_id])
        else:
            combined_data = data_structure[task_id]

        dataset = convert_task_to_dataset(combined_data)
        dataset = preprocess(
            tokenizer, args, accelerator, dataset, caption_column, image_column
        )
    except:
        logger.error(
            f"Error in getting dataset for current task: {task_id} not in data_structure"
        )
        return None
    return dataset


# TEST
# data_structure = {
#     1: {'images':  np.random.randint(0, 256, size=(100, 32, 32, 3), dtype=np.uint8), 'labels': np.random.randint(0, 5, 100)},
#     2: {'images':  np.random.randint(0, 256, size=(100, 32, 32, 3), dtype=np.uint8),'labels': np.random.randint(5, 10, 100)}  # Example state with no labels provided
# }

