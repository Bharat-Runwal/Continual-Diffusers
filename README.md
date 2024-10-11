# Continual-Diffusers  : A PyTorch library for continual learning with diffusion models

A pytorch library for continual learning with diffusion models. This library is based on the hugginface [diffusers](https://github.com/huggingface/diffusers).

<p align="center">
  <img src ="misc/Continual_Diffusers.png" height=240 , width=350 >
</p>

<p align="center"> <a href="https://bharat-runwal.github.io/" target="_blank id="website">Bharat Runwal</a><sup>1</sup>, <a href="https://yilundu.github.io/" target="_blank id="website">Yilun Du</a><sup>2</sup></p>
<p align="center"><sup>1</sup> Independent Researcher,<sup>2</sup> MIT</p>

---

## Features

- **Continual Learning**: Train diffusion models incrementally on new data without forgetting previous tasks.
- **Energy-Based and Score-Based Training**: Support for different training paradigms, including energy-based and score-based methods.
- **Conditioning Mechanisms**: Compatible with class and text conditioning.
- **Evaluation**: Supports multiple evaluation types such as ancestral sampling, MCMC sampling, and compositional generation.
- **Training**: Includes support for Multi-GPU training, gradient accumulation, and other performance optimizations provided by the Diffusers library.

Comming Soon:
- **LoRA**: Support for LoRA finetuning with Latent Diffusion Models.
- **Slot Attention**: Conditioning with a slot attention module for training diffusion models.
---

## Setting up the environment
```
conda create -n continual-diffusers python=3.8
conda activate continual-diffusers
pip install -e .
```
For setting up `bitsandbytes`,  please follow the official installation instructions provided [here](https://huggingface.co/docs/bitsandbytes/main/en/installation).

## Supported Training Strategies

| Model Type         | Energy-Based Training | Score-Based Training | Conditioning                      |
|--------------------|-----------------------|----------------------|-----------------------------------|
| DDPM Unet   | ✔️                      |✔️                    |`Text`, `class-conditional`, `unconditional`                   |
| Latent Diffusion Model   | ✔️                     | ✔️                    | `text`   |

**Note**: The Energy-Based and Score-Based Training both supports `Classifier-Free Guidance`. 


## Supported Techniques for continual-learning
We mainly support following techniques for continual learning with diffusion models to mitigate catastrophic forgetting.

| Model Type         | Regularization Techniques | 
|--------------------|---------------------------|
|DDPM Unet            | `Elastic Weight Consolidation`, `Buffer Replay` |
|Latent Diffusion Model | `Buffer Replay` | 

`Buffer Replay` is a simple technique that stores a small number of samples from previous tasks and replays them during training. We have `Full Replay`, `No Replay`, and `Fixed Random Replay` (Uniform label support available).  

`Elastic Weight Consolidation` is a regularization technique that constrains the model's weights to stay close to their values at the end of training on previous tasks.


### Training DDPM Model 

#### Score-Based Training
The following commands trains a DDPM model with score-based appraoch Please refer to the `scripts/train_ddpm_score.sh` for more details.
 
```
bash scripts/train_ddpm_score.sh
```

### Energy-Based Training
The following commands trains a DDPM model with energy-based appraoch. We support two energy-score Types : `Denoising Auto-Encoder (DAE)` and `L2`. 
Please refer to the `scripts/train_ddpm_energy.sh` for more details.
```
bash scripts/train_ddpm_energy.sh
```

**Note :** For `text` based conditioning, please refer to the script `scripts/train_clip_cond.sh` for more details.


### Training Latent Diffusion Model

#### Score-Based Training

We support training the unet model from scratch and finetuning from a pretrained Unet model (eg. From Stable-Diffusion). Please refer to the `scripts/train_latent_diffusion_score.sh` for more details.

```
bash scripts/train_latent_diffusion_score.sh
```

#### Energy-Based Training : 

We experimented with fine-tuning the pretrained U-Net from the Stable Diffusion checkpoint using energy-based training. However, the results were not satisfactory, so we currently do not provide a dedicated script for energy-based training of Latent Diffusion Models (LDMs). That said, if you wish to explore this further, the framework allows you to train the U-Net either from a pretrained checkpoint or from scratch using the `--unet_scratch` argument in `main_ldm.py` with different energy score type supported.
       


## Dataset Structure

The dataset used for continual learning in **Continual-Diffusers** should follow a structured format, where data is divided into different tasks, each containing a set of images and their corresponding labels. This structure allows for task-based training and evaluation.

Each task is represented as a dictionary where:

- The **key** represents the task number (e.g., Task 1, Task 2, etc.). Starting from 1.
- The **value** is another dictionary that contains:
  - `images`: A collection of images, typically in a 4D format where the dimensions represent the number of images, image height, width, and channels (e.g., RGB channels).
  - `labels`: A set of labels associated with each image, corresponding to specific classes for that task. The labels can be class indices, and each task can have its own range of labels. This can also be a list of text strings for text-conditioning.

### Example Structure

Here's an example of how the data might be structured:

```python
import pickle 

data_structure = {
    1: {
        'images': <images for Task 1>, 
        'labels': <labels for Task 1> 
    },
    2: {
        'images': <images for Task 2>, 
        'labels': <labels for Task 2> 
    }
}

# Please save the above data in following format : 
np.savez("example_data.npz", data_structure=pickle.dumps(data_structure))

```

Please look at the example script for creating CIFAR10 dataset `create_cifar10_data_example.py`. 

## Generating Samples

We support the following evaluation pipelines for generating samples from the trained models:

| Model Type               | Ancestral Sampling (Default) | Ancestral + MCMC Sampling | Compositional Generation |
|--------------------------|-----------------------------|---------------------------|--------------------------|
| DDPM Unet                | ✔️                          | - `Score Based: UHA, ULA`  <br> - `Energy-Based: UHA, ULA, CHA, MALA`                        | - Supported with Ancestral and MCMC Samplers Both  <br> - `Score Based: UHA, ULA`  <br> - `Energy-Based: UHA, ULA, CHA, MALA` |
| Latent Diffusion Model                | ✔️                          | - `Score Based: UHA, ULA`  <br> - `Energy-Based`: ❌                        | - Supported with Ancestral and MCMC Samplers Both  <br> - `Score Based: UHA, ULA`  <br> - `Energy-Based`: ❌ |


### Ancestral Sampling

Following command generates samples using ancestral sampling from the trained model. Please refer to the `scripts/generate_samples.sh` for more details.

```bash 
bash scripts/generate_samples_ancestral.sh
```

### Ancestral + MCMC Sampling

Following command generates samples using ancestral + MCMC sampling from the trained model. Please refer to the `scripts/generate_samples_mcmc.sh` for more details.

```bash
bash scripts/generate_samples_mcmc.sh
```

We currently support the following MCMC samplers:

- `UHA` : Unadjusted Hamiltonian Monte-Carlo Algorithm
- `ULA` : Unadjusted Langevin Algorithm
- `CHA` : Hamiltonian Monte-Carlo Algorithm
- `MALA` : Metropolis-Adjusted Langevin Algorithm


**Note**:  We use MCMC sampling for only few k steps, this can be controlled by argument `mcmc_sampler_start_timestep` . Please refer to the evaluation code for more details on arguments.

### Compositional Generation

We also allow composing multiple concepts (text or class labels) to generate novel images. Please refer to the `scripts/compositional_generation.sh` for more details.

```bash
bash scripts/compositional_generation.sh
```

**Note**:
- We currently support only one image generation at a time for compositional generation.
-  We currently support energy-based compositional generation for DDPM Unet model only.


## Acknowledgements

- [Huggingface Diffusers](https://github.com/huggingface/diffusers)
- [Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion
Models and MCMC](https://github.com/yilundu/reduce_reuse_recycle)



