import logging
import pickle
import random

import numpy as np
from datasets import Dataset as Dataset_hf
from PIL import Image
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
    return Dataset_hf.from_dict(task)


def get_datasets_ldm(args):
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
            raise KeyError("Key 'data_structure' not found in the file.")

    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")



def get_tokenized_text(captions, tokenizer):
    """
    Tokenize the input text using the provided tokenizer
    """
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def preprocess(tokenizer, args, accelerator, dataset, caption_column, image_column,train_transforms):
    # Preprocessing the datasets.
    if train_transforms is None:
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        
        logger.info(f"Using default training transforms: {train_transforms}")

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

        inputs = get_tokenized_text(captions, tokenizer) 
        return inputs.input_ids


    def preprocess_train(examples):
        images = [
            Image.fromarray(np.array(image, dtype=np.uint8)).convert("RGB")
            for image in examples[image_column]
        ]
        examples["images"] = [train_transforms(image) for image in images]
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
    train_transform=None,
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
        train_transform: The transforms to be applied to the training images

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
            tokenizer, args, accelerator, dataset, caption_column, image_column,train_transform
        )
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")
    
    return dataset