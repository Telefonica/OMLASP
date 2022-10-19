# Tool to audit Fast Gradient Sign Method (FGSM) in Machine Learning algorithms

## Setting up the environment

Import conda environment and activate it with the following commands:

```
conda env create -n attackdefend_fgsm --file attackdefend_fgsm.yml
conda activate attackdefend_fgsm
```

## Description

Program name: **fgsm.py**

You can do the following tasks:
- Generate a dataset to hack this model (Task 1).
- Check the robustness of your model (Task 2).
- Train your model to avoid FGSM attacks (Task 3).

The arguments received by the program are the following (you can run **python fgsm.py -h** for a deeper explanation):

```
model_file_path: The path of the file that contains the model.

dataset_path: The path of the dataset. Each of the images must be in a folder that indicates its label.

task: ['gen_data', 'check_loss', 'train', 'all']. You must choose one of the following options. Generate modified images, check the loss of your model when images are modified or train your model.

-s  or --image-size: The target size of the images. The images will be pre-processed and resized to that size.

-p  or --results-path: The path where you want to save the results. Default='./results/'

-e  or --epsilon: Enter how much you want to modify the images. If epsilon is small, the modifications of images will be small too. This argument is only needed for task 1 and 2. Default=0.1

-b  or --batch-size: The batch size. For efficiency reasons it should be a multiple of 2. For example: 16, 32, 64, 128. Default=1

-n  or --n-epochs: The number of epochs you want to train the neural network. This argument is only needed for task 3. Default=15

-v  or --epsilon-values: How many epsilons you want to generate to train the model. This argument is only needed for task 3. Default=10
```

## How it works

We have a model trained on cifar10. We apply an fgsm algorithm that makes it generate this same dataset but poisoned with fgsm attacks, and saves it in another folder. Then it generates model error rate on real dataset and on modified dataset.  Afterwards we retrain the model to reduce its loss with respect to fgsm attacks. This way we reduce the error rates, creating a more robust model, with better generalisation capability.


https://www.cs.toronto.edu/~kriz/cifar.html


## Example of use

In order to generate poisoned dataset, check loss rate and train a new model:

```
python fgsm.py cifar10-model.h5 datasetCifar/train all -s 32 32 -b 128
```

In order to check loss rate with the new trained model:

```
python fgsm.py results/new_model.h5 datasetCifar/train check_loss -s 32 32 -b 128
```
