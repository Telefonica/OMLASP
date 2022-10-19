# Tool that allows obtaining the parameters of a neural network

## Setting up the environment

Import conda environment and activate it with the following commands:

```
conda env create -n attackdefend_reverseneuralnetwork --file attackdefend_reverseneuralnetwork.yml

conda activate attackdefend_reverseneuralnetwork
```

## Description

Program name: **neuralnetworkreversing.py**


Given the inputs and outputs of a neural network, the aim is to infer the parameters of the network, saving the new model with the found parameters.


The arguments received by the program are the following (you can run **python neuralnetworkreversing.py -h** for a deeper explanation):

```
content_file_path: The path of the file that contains the inputs and outputs of the target model. It must be a CSV file with values separated using a comma.

-l  or  --layers: The number of layers. Default=4.

-n  or --neurons: The number of neurons per layer. Enter a list of values.

-i  or --inputdim: The input dimension.

-o  or --outputdim: The output dimension.

-a  or --activation: The activation function of each layer. It will be the same for all layers, except the last one which will have a softmax.

-b  or --batchsize: The training batch size. Default=32.

-e  or --epochs: The number of epochs. Default=50.

-s  or --savepath: The path where the parameters of the model are going to be saved. Default = '\model_params.h5'.
```

## How it works


From the parameters passed as arguments, a neural network is created and trained on the data to try to approximate the model that has actually been used.


## Example of use

```
python neuralnetworkreversing.py -l 5 -n 16 32 64 32 16 -i 784 -o 10 -a relu -b 32 -e 10 ./data.csv
```