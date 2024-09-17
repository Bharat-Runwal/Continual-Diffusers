import logging

import numpy as np
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BufferReplay:
    def __init__(self, args, device):
        """
        This class is responsible for managing the replay buffer for the reservoir, all_data, no_data, and generative strategies.
        args : dict - the arguments to pass to the replay
        """
        assert args.buffer_type in [
            "reservoir",
            "all_data",
            "no_data",
            "generative",
        ], "Buffer type not recognized."
        self.buffer = {}
        self.device = device
        self.args = args
        self.buffer_type = args.buffer_type
        self.buffer_size = args.buffer_size
        self.label_counts = {}
        self.task_count = 0

        logger.info(
            f"Initialized BufferReplay with type {args.buffer_type} on device {device}"
        )

    def add_task_data(self, task_data):
        """Add data from a new task to the buffer, adjusting the buffer to maintain a balanced representation across all tasks."""
        self.task_count += 1
        task_id = self.task_count

        if self.buffer_type == "reservoir":
            if self.args.num_class_labels is not None or self.args.text_conditioning:
                self.buffer[task_id] = {
                    "images": np.array(task_data["images"]),
                    "labels": np.array(task_data["labels"]),
                }
            else:
                self.buffer[task_id] = {"images": np.array(task_data["images"])}
            # Adjust the buffer to evenly distribute across all tasks
            self._rebalance_buffer()
        elif self.buffer_type == "all_data":
            # No rebalancing is needed as we use full data for all tasks
            self.buffer[task_id] = {
                "images": np.array(task_data["images"]),
                "labels": np.array(
                    task_data.get("labels", [None] * len(task_data["images"]))
                ),
            }
        elif self.buffer_type == "no_data":
            # No operation needed as we don't store any data
            self.buffer[task_id] = {"images": np.array([]), "labels": np.array([])}
        elif self.buffer_type == "generative":
            # Generate data using a generative model
            raise NotImplementedError("Generative Replay not implemented yet.")

    def _rebalance_buffer(self):
        """Rebalances the buffer to ensure uniform distribution across all tasks and optionally across labels within tasks."""
        if self.task_count == 0:
            return

        per_task_limit = self.buffer_size // self.task_count

        # Truncate or sample each task's data to fit the new per task limit
        for task_id, data in self.buffer.items():
            if (
                self.args.retain_label_uniform_sampling
                and self.args.num_class_labels is not None
            ):
                # Ensure uniform label distribution within the task
                labels, counts = np.unique(data["labels"], return_counts=True)
                num_samples_per_label = min(
                    per_task_limit // len(labels), min(counts)
                )  # Avoid exceeding the number of available labels

                indices = np.concatenate(
                    [
                        np.random.choice(
                            np.where(data["labels"] == label)[0],
                            num_samples_per_label,
                            replace=False,
                        )
                        for label in labels
                    ]
                )
            else:
                # Randomly sample without considering labels
                total_random_samples = min(per_task_limit, len(data["images"]))
                indices = np.random.choice(
                    len(data["images"]), total_random_samples, replace=False
                )

            # Replace the self.buffer
            if self.args.num_class_labels is not None or self.args.text_conditioning:
                self.buffer[task_id] = {
                    "images": data["images"][indices],
                    "labels": data["labels"][indices],
                }
            else:
                self.buffer[task_id] = {"images": data["images"][indices]}
            logger.info(f"Buffer replay indices choose for task {task_id} : {indices}")
            # save the buffer idx and data for reproducibility
            np.savez(
                f"{self.args.output_dir}/buffer_indices_task_{task_id}.npz",
                indices=indices,
                buffer_data=self.buffer[task_id],
            )

        logger.info(
            f"Buffer rebalanced to maintain a limit of {per_task_limit} items per task with uniform label sampling: {self.args.retain_label_uniform_sampling}"
        )

    def generate_data(self, data):
        """Generates data using a generative model (placeholder for real generative logic)."""
        return data  # Placeholder for actual generation logic

    def get_combined_data(self, current_task_data):
        """Combines the buffer with the current task data, handling potentially missing labels."""
        combined_images = []
        combined_labels = []

        # Collect data from buffer
        for task_id, data in self.buffer.items():
            combined_images.extend(data["images"])
            if "labels" in data:
                combined_labels.extend(data["labels"])

        # Add current task data
        combined_images.extend(current_task_data["images"])
        if "labels" in current_task_data:
            combined_labels.extend(current_task_data["labels"])

        if combined_labels:
            return {"images": combined_images, "labels": combined_labels}
        else:

            return {
                "images": combined_images
            }  # Return without labels if none were provided
