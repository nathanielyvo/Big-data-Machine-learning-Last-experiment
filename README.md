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
To reproduce the Demonstration experiment for DG, please use **train.py** & **test.py**.
 - `PACS`: Folder to store PACS data set.
 - `datasets.py`: Contains 3 different dataset classes. Suitable for different methods of training and testing
 - `utils.py`: Contains some usage code for building dataloader and setting seeds, etc.
 - `train.py`: Code used for trainings.
 - `test.py`: Code used for testings.
 - `model.py`: Contains resnet network structure code for training and testing.
 - `config.py`: Contains arguments for training and testing.

## Model Training and Evaluation
Please use **train.py** & **test.py** and follow the instructions below. 

## Arguments
The **train.py** & **test.py** takes the following command line options (details are in the `config.py`):

**Related Arguments**
- `epoch`: the number of training epoch

- `lr`: The learning rate of training

- `weight_decay`: The weight_decay of training

- `num_workers`: the num_workers of dataloader

- `batchsize`: the batchsize of dataloader

## Training and testing scripts for Demonstration experiment

For example, to train the model with batchsize 32:

    $ python train.py --epoch 200 --lr 0.001 --weight_decay 0.0005 --num_workers 32 --batchsize 32
    
For example, to test the model and get the result.csv

    $ python test.py --num_workers 32 --batchsize 32

