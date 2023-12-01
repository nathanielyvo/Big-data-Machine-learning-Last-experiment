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
To reproduce the demo experimentsfor DG, please use **train.py** & **test.py**.
 - `PACS`: Folder to store PACS data set.
 - `datasets.py`: Contains 3 different dataset classes. Suitable for different methods of training and testing
 - `utils.py`: Contains some usage code for building dataloader and setting seeds, etc.
 - `train.py`: Code used for trainings.
 - `test.py` : Code used for testings.
 - `model.py`: Contains resnet network structure code for training and testing。

## Model Training and Evaluation
Please use **train.py** & **test.py** and follow the instructions below. 

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

    $ python train.py --epoch 200 --lr 0.001 --weight_decay 0.0005 --num_workers 32 --batchsize 32
    
For example, to train the 1-shot/5-shot 5-way MAML/UNICORN-MAML model with ResNet-12 backbone on MiniImageNet:

    $ python test.py --num_workers 32 --batchsize 32

