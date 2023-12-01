# Big-data-Machine-learning-Last-experiment


## Prerequisites

The following packages are required to run the scripts:

- [PyTorch-2.0 and torchvision](https://pytorch.org)

## Dataset

PACS is an image dataset for domain generalization. It consists of four domains, namely 
Photo (1,670 images), Art Painting (2,048 images), Cartoon (2,344 images) and Sketch (3,929 
images). Each domain contains seven categories.

PACS dataset link: https://drive.google.com/file/d/1mcrwg5sYXgzweDKmeKLMUpiXX6hHCC7Y/view?usp=drive_link

## Code Structures
To reproduce our experiments with UNICORN-MAML, please use **train_fsl.py**. There are four parts in the code.
 - `model`: It contains the main files of the code, including the few-shot learning trainer, the dataloader, the network architectures, and baseline and comparison models.
 - `data`: Images and splits for the data sets.
 - `saves`: The pre-trained weights of different networks.
 - `checkpoints`: To save the trained models.

## Model Training and Evaluation
Please use **train_fsl.py** and follow the instructions below. The file will automatically evaluate the model on the meta-test set with 10,000 tasks after given epochs.

## Arguments
The train_fsl.py takes the following command line options (details are in the `model/utils.py`):

**Task Related Arguments**
- `dataset`: Option for the dataset (`MiniImageNet`, `TieredImageNet`, or `CUB`), default to `MiniImageNet`

- `way`: The number of classes in a few-shot task during meta-training, default to `5`

- `eval_way`: The number of classes in a few-shot task during meta-test, default to `5`

- `shot`: Number of instances in each class in a few-shot task during meta-training, default to `1`

- `eval_shot`: Number of instances in each class in a few-shot task during meta-test, default to `1`

- `query`: Number of instances in each class to evaluate the performance during meta-training, default to `15`

- `eval_query`: Number of instances in each class to evaluate the performance during meta-test, default to `15`

## Training scripts for UNICORN-MAML

For example, to train the 1-shot/5-shot 5-way MAML/UNICORN-MAML model with ResNet-12 backbone on MiniImageNet:

    $ python train_fsl.py --max_epoch 100 --way 5 --eval_way 5 --lr_scheduler step --model_class MAML --lr_mul 10 --backbone_class Res12 --dataset MiniImageNet --gpu 0 --query 15 --step_size 20 --gamma 0.1 --para_init './saves/initialization/miniimagenet/Res12-pre.pth' --lr 0.001 --shot 1 --eval_shot 1  --temperature 0.5 --gd_lr 0.05 --inner_iters 15
	$ python train_fsl.py --max_epoch 100 --way 5 --eval_way 5 --lr_scheduler step --model_class MAML --lr_mul 10 --backbone_class Res12 --dataset MiniImageNet --gpu 0 --query 15 --step_size 20 --gamma 0.1 --para_init './saves/initialization/miniimagenet/Res12-pre.pth' --lr 0.001 --shot 5 --eval_shot 5  --temperature 0.5 --gd_lr 0.1 --inner_iters 20 
	$ python train_fsl.py --max_epoch 100 --way 5 --eval_way 5 --lr_scheduler step --model_class MAMLUnicorn --lr_mul 10 --backbone_class Res12 --dataset MiniImageNet --gpu 0 --query 15 --step_size 20 --gamma 0.1 --para_init './saves/initialization/miniimagenet/Res12-pre.pth' --lr 0.001 --shot 1 --eval_shot 1  --temperature 0.5 --gd_lr 0.1 --inner_iters 5 
	$ python train_fsl.py --max_epoch 100 --way 5 --eval_way 5 --lr_scheduler step --model_class MAMLUnicorn --lr_mul 10 --backbone_class Res12 --dataset MiniImageNet --gpu 0 --query 15 --step_size 20 --gamma 0.1 --para_init './saves/initialization/miniimagenet/Res12-pre.pth' --lr 0.001 --shot 5 --eval_shot 5  --temperature 0.5 --gd_lr 0.1 --inner_iters 20 
