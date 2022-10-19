#  Tool to audit scaling methods

## Setting up the environment

Install the requirements (it is recommended to use Python 3.8):

```
pip install -r requirements.txt

pip install git+https://github.com/google-research/tensorflow_constrained_optimization.git
```


## Description

Program name: **scaling.py**

You can do the following task:

- Generating an image that if reduced to a certain size with a specific algorithm produces a totally different output.

Requirements:

- You have to introduce 2 images: one called source image and another one called target image.

- The larger the source image and the smaller the target image, the more likely the attack will be successful.

- The size of the target image should be the size of the model input image.

The arguments received by the program are the following (you can run **python scaling.py -h** for a deeper explanation):

```
source_img_path: The path of the file that contains the source image.

target_img_path: The path of the file that contains the target image.

-m  or --method: ['bilinear', 'lanczos3', 'lanczos5', 'bicubic', 'gaussian', 'nearest', 'area', 'mitchellcubic']. You must choose one of the following scaling algorithms.
    
-s  or --target-size: The size of the target image. The target image will be pre-processed and resized to that size.

-p  or --results-path: The path where you want to save the results. Default='./results/'

-e  or --epsilon: How much you want to modify the images. If epsilon is small, the modifications of images will be small too. If epsilon is too small maybe the attack will not have success. Default=0.1

-n  or --niterations: The number of iterations of the algorithm. The more iterations the algorithm will take, but the better the results will be. It is recommended to use a GPU. Default=20000
```
        
        
## How it works


In a deep learning model, when you are going to predict a new image, it is rescaled and a preprocessing is applied to it to adjust it to the input of your model. It is in this rescaling of the image that an attack can be applied, so that when the original image is rescaled, a completely different one chosen by the attacker appears. 


## Example of use

```
python scaling.py obama_grande.jpg trump.jpg -m bilinear -p results/ -n 20000
```


