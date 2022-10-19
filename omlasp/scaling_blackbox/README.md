# Scaling   black-box attack

## Setting up the environment

Install the requirements (it is recommended to use Python 3.8):

```
pip install -r requirements.txt

pip install git+https://github.com/google-research/tensorflow_constrained_optimization.git
```

## Description

Notebook name: **scaling_blackbox.ipynb**

In this notebook you can upload two images and a trained model  (it simulates an example AI model that a company may have trained, we don't know what size image it has been trained with and the interpolation method used in order to resize the images).

One of the images can be an example of an image that might be close to what a company's AI model might be predicting, but in reality we can use any image. This image is defined in the code as **Source image (S)**.

The other image is the important one, this is the one that will appear when we have rescaled the Source image (S). This image is defined in the code as the **Target image (T)**. It is the one we will use to alter the model prediction.

## How it works

We rescale the Source image (S) and try to make a prediction with the loaded AI model, if it does not give any result it means that the image size is not correct, that is, it is not the image size with which the possible company has trained its model, until a moment comes when we manage to obtain the prediction data, this means that we have discovered the correct image size.

Then we can perform the scaling attack with different interpolation methods and observe which methods the model is vulnerable to.

## Example of use

Just follow the steps defined in the notebook.