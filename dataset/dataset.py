import logging
import pickle
import random

import numpy as np
import torch as th
from PIL import Image
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sanity_check(labels):
    return np.squeeze(labels)

class ContinualDataset(Dataset):
    def __init__(self, uncond_p=0.05, data_structure=None, transform=None):

        self.data_structure = data_structure
        self.uncond_p = uncond_p
        self.current_task = None
        self.transform = transform
        self.task_data = {}

    def set_current_task(self, task_num, buffer=None):
        """
        Sets the current task and integrates buffer data if provided.

        Args:
            task_num (int): The task number to set as current.
            buffer (BufferReplay, optional): The buffer to integrate data from previous tasks.
        """
        self.current_task = task_num
        if buffer:
            # Fetches the current task data (could be empty if not loaded)
            self.load_task_data(task_num)
            # Combine it with the buffered data
            combined_data = buffer.get_combined_data(self.task_data[task_num])
            self.task_data[task_num] = combined_data
            logger.info(f"Set current task to {task_num} with buffer integration.")
        else:
            self.load_task_data(task_num)
            logger.info(f"Set current task to {task_num} without buffer integration.")

    def load_task_data(self, task_num):
        if task_num in self.data_structure:
            task_info = self.data_structure[task_num]
            self.task_data[task_num] = {
                "images": task_info["images"],
                "labels": task_info.get("labels", np.array([])),
            }
            logger.info(f"Task data for task {task_num} loaded.")
        else:
            logger.error(f"Task {task_num} not found in data structure.")

    def get_class_labels(self, task_num):
        # returns the unique class labels for the given task_num Task
        return th.unique(th.tensor(self.task_data[task_num]["labels"])).tolist()

    # TODO : Handle the case where the images are presnt in the native format
    def __getitem__(self, index):
        """
        Fetches the item at the given index from the current task data.
        """
        im = Image.fromarray(self.task_data[self.current_task]["images"][index])
        if self.transform is not None:
            base_tensor = self.transform(im)

        labels = self.task_data[self.current_task].get("labels", None)

        label = (
            labels[index] if labels is not None and len(labels) > 0 else -1
        )  # Handle empty or None labels

        mask = th.rand(1).item() > self.uncond_p
        return {
            "images": base_tensor,
            "labels": th.tensor(label, dtype=th.long),
            "masks": th.tensor(mask, dtype=th.bool),
        }

    def __len__(self):
        """
        Returns the number of items in the current task.
        """
        return (
            len(self.task_data[self.current_task]["images"])
            if self.current_task in self.task_data
            else 0
        )


def get_dataset(args, transform=None):
    """
    Initialize the dataset based on the provided arguments and return the dataset instance and total number of tasks.

    Args:
        args: The arguments containing data path and unconditional probability.

    Returns:
        tuple: A tuple containing the dataset instance and the number of tasks.
    """
    try:
        data = np.load(args.data_path, allow_pickle=True)
        logger.info("Loading data from %s", args.data_path)

        if "data_structure" in data:
            # Assuming 'data_structure' is stored as a pickled dictionary
            data_structure = pickle.loads(data["data_structure"].item())
            total_tasks = len(data_structure)

            dataset_stats = {}
            for task_num in data_structure:
                # Check if 'labels' key is present in the task dictionary
                if 'labels' in data_structure[task_num]:
                    data_structure[task_num]['labels'] = sanity_check(data_structure[task_num]['labels'])
                    dataset_stats[task_num] = {
                        "images": data_structure[task_num]['images'].shape,
                        "labels": data_structure[task_num]['labels'].shape
                    }
                else:
                    dataset_stats[task_num] = {
                        "images": data_structure[task_num]['images'].shape,
                    }

            logger.info(f"Dataset Statistics: {dataset_stats}")
            # Initialize the dataset with the retrieved structure
            dataset = ContinualDataset(
                uncond_p=args.uncond_p,
                data_structure=data_structure,
                transform=transform,
            )
            logger.info(f"Dataset successfully loaded with {total_tasks} tasks.")
            return dataset, total_tasks
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


# TEST NEW


# Example data structure with fake images and labels as numpy

# data_structure = {
#     1: {'images':  np.random.randint(0, 256, size=(100, 32, 32, 3), dtype=np.uint8), 'labels': np.random.randint(0, 5, 100)},
#     2: {'images':  np.random.randint(0, 256, size=(100, 32, 32, 3), dtype=np.uint8),'labels': np.random.randint(5, 10, 100)}  # Example state with no labels provided
# }

# data_structure = {
#     1: {'images':  np.random.randint(0, 256, size=(100, 32, 32, 3), dtype=np.uint8), 'labels': np.random.randint(0, 10, 100)},
#     2: {'images':  np.random.randint(0, 256, size=(100, 32, 32, 3), dtype=np.uint8),}  # Example state with no labels provided
# }
# # save this data structure to a file
# np.savez("example_data.npz", data_structure=pickle.dumps(data_structure))

# dataset = ContinualDataset(uncond_p=0.05, data_structure=data_structure)

# dataset.set_current_task(1)
# print(dataset.task_data[1]['images'].shape)
# print(dataset.task_data[1]['labels'].shape)


# # GET item
# item = dataset.__getitem__(0)
# print(item['images'].shape,item['images'])
# print(item['labels'].shape,item['labels'])
# print(item['masks'])

# TEST
# Example data structure
# data_structure = {
#     1: {'images': 'task1_img', 'labels': 'task1_lab'},
#     2: {'images': 'task2_img'}  # Example state with no labels provided
# }

# # Example dataset instantiation
# dataset = ContinualDataset(data_path="./example_data.npz", state=1, uncond_p=0.05, buffer=True, data_structure=data_structure)
