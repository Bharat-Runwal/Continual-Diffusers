import numpy as np
from torchvision import datasets
from torch.utils.data import Subset
import matplotlib.pyplot as plt

def get_cifar10_tasks():

    # Load the CIFAR-10 training and test datasets
    cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True)
    cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True)
    
    # Dictionary to store tasks
    tasks = {}
    
    # Split the dataset into 5 tasks, each with 2 classes
    for task_id in range(5):
        # Calculate the class indices for this task
        start_class = task_id * 2
        end_class = start_class + 2
        
        # Get indices of images belonging to the current task's classes
        train_indices = [i for i, label in enumerate(cifar10_train.targets) if label in range(start_class, end_class)]
        test_indices = [i for i, label in enumerate(cifar10_test.targets) if label in range(start_class, end_class)]
        
        # Create subsets for training and testing
        train_subset = Subset(cifar10_train, train_indices)
        test_subset = Subset(cifar10_test, test_indices)
        
        # Convert the training subset to NumPy arrays without using DataLoader
        images = []
        labels = []
        
        for i in range(len(train_subset)):
            image, label = train_subset[i]
            images.append(image)
            labels.append(label)
        
        # Stack the images and labels lists into numpy arrays
        images_np = np.stack(images)
        labels_np = np.array(labels)
        
        # Save the arrays in the dictionary
        tasks[task_id+1] = {"images": images_np, "labels": labels_np}
    
    import pickle 

    # Save the entire dictionary to an npz file
    np.savez('cifar10_tasks_train.npz',data_structure=pickle.dumps(tasks))
    
    # Plot the tasks
    # plot_tasks(tasks)

    return tasks

# def plot_tasks(tasks, samples_per_task=5):
#     # Define class names (assuming standard CIFAR-10 classes)
#     class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
#                    'dog', 'frog', 'horse', 'ship', 'truck']
    
#     # Create a plot for each task
#     for task_id, data in tasks.items():
#         images = data['images']
#         labels = data['labels']
        
#         fig, axs = plt.subplots(1, samples_per_task, figsize=(15, 3))
#         fig.suptitle(f'Task {task_id} (Classes: {class_names[task_id*2]} & {class_names[task_id*2+1]})', fontsize=16)
        
#         for i in range(samples_per_task):
#             img = images[i].transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
#             img = img * 0.5 + 0.5  # Unnormalize
#             axs[i].imshow(img)
#             axs[i].axis('off')
#             axs[i].set_title(class_names[labels[i]])
        
#         plt.show()

# Call the function to create, save, and plot the tasks
tasks_dict = get_cifar10_tasks()
