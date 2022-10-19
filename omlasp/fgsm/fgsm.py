import os
# Delete debug TF messages
"""
    0 = all messages are logged (default behavior)
    1 = INFO messages are not printed
    2 = INFO and WARNING messages are not printed
    3 = INFO, WARNING, and ERROR messages are not printed
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import cv2
import glob
from matplotlib import pyplot as plt
import numpy as np
import argparse
import textwrap
import sys

class TFUtils:

    @staticmethod
    def check_gpu():
        return tf.test.is_gpu_available()

class ImageUtils:
    @staticmethod
    def load_dataset(dataset_path, img_size, class_mode='categorical', train_batch_size=32, test_batch_size=32, validation_split=False, validation_split_per=0.1):

        if validation_split:
            generator = ImageDataGenerator(rescale=1./255, validation_split=validation_split_per)
            train_generator = generator.flow_from_directory(
                dataset_path,
                target_size=(img_size[0],img_size[1]),
                batch_size=train_batch_size,
                shuffle=True,
                class_mode='categorical',
                subset='training'
            )

            val_generator = generator.flow_from_directory(
                dataset_path,
                target_size=(img_size[0],img_size[1]),
                batch_size=test_batch_size,
                shuffle=True,
                class_mode='categorical',
                subset='validation'
            )
        else:
            generator = ImageDataGenerator(rescale=1./255)
            train_generator = generator.flow_from_directory(
                dataset_path+"/train",
                target_size=(img_size[0],img_size[1]),
                batch_size=train_batch_size,
                shuffle=True,
                class_mode='categorical'
            )

            generator = ImageDataGenerator(rescale=1./255)
            val_generator = generator.flow_from_directory(
                dataset_path+"/test",
                target_size=(img_size[0],img_size[1]),
                batch_size=test_batch_size,
                shuffle=True,
                class_mode='categorical'
            )

        return train_generator, val_generator

    @staticmethod
    def generate_dataset(dataset_path, img_size, class_mode='categorical'):
        # No hace falta poner el batch size
        batch_size = 128
        generator, _ = ImageUtils.load_dataset(dataset_path, img_size, class_mode='categorical', train_batch_size=batch_size, test_batch_size=batch_size, 
            validation_split=True, validation_split_per=0.0)

        number_imgs = ImageUtils.calculate_recursively_number_imgs(dataset_path)
        max_step = (number_imgs // batch_size) + 1

        first_time = True
        for step, (imgs, labels) in enumerate(generator):
            if step == max_step:
                break

            imgs = tf.convert_to_tensor(imgs)
            labels = tf.convert_to_tensor(labels)

            if first_time:
                x = imgs
                y = labels
                first_time = False
            else:
                x = tf.concat([x, imgs], axis=0)
                y = tf.concat([y, labels], axis=0)

        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        return dataset

        

    @staticmethod
    def calculate_recursively_number_imgs(path):
        n_imgs = 0

        for path in Path(path).rglob('*.jpg'):
            n_imgs += 1

        for path in Path(path).rglob('*.png'):
            n_imgs += 1

        for path in Path(path).rglob('*.jpeg'):
            n_imgs += 1

        return n_imgs


    @staticmethod
    def load_images_recursively(directories):
        # directories is a list of directories. For example: ['buttons/', 'checkbuttons/', 'desplegables/', 'images/', 'inputs/', 'labels/', 'paragraphs/', 'radiobuttons/']
        images = []

        for directory in directories:
            for path in Path(directory).rglob('*.jpg'):
                path = os.path.abspath(path)
                images.append(cv2.imread(path,1))
            for path in Path(directory).rglob('*.png'):
                path = os.path.abspath(path)
                images.append(cv2.imread(path,1))
            for path in Path(directory).rglob('*.jpeg'):
                path = os.path.abspath(path)
                images.append(cv2.imread(path,1))
            
        return images

    @staticmethod
    def save_images(imgs, path_to_save):
        index=0
        
        for img in imgs:
            #plt.imshow(img.numpy()[:,:,::-1])
            #plt.show()

            img = tf.image.convert_image_dtype(img, tf.uint8, saturate=True).numpy()

            cv2.imwrite(path_to_save+'/'+str(index)+'.png', img)
            index += 1
        

class ModelUtils:

    @staticmethod
    def load_model(file_model_path):
        """
            Si el usuario ha construido un modelo funcional con TF debe guardarlo como model.save('modelo.h5')
            Si el usuario ha construido un modelo que no cumple con la API funcional de TF debe guardarlo como model.save('modelo.model', save_format="tf")
        """

        file_model = Path(file_model_path)

        if not file_model.is_file():
            return None

        # Loads the model regardless of the format (HDF5 or SavedModel format)
        # HDF5 -> Keras model format. To sve the model in this format the file must be in format .h5
        # SavedModel -> TensorFlow model format (it is used by default)
        model = tf.keras.models.load_model(file_model)

        return model

    
class FSGMAttack:

    def __init__(self, model_path, loss_function, optimizer=tf.keras.optimizers.Adam()):
        self.model = ModelUtils.load_model(model_path)
        self.loss_function = loss_function
        self.optimizer = optimizer

    def build_imgs(self, x, y, epsilon):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y_hat = self.model(x)
            loss = self.loss_function(y, y_hat)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, x)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)

        modified_imgs = x + epsilon*signed_grad

        return modified_imgs

    def FSGM_loss(self, y_true, y_init_img, y_modified_img):
        alpha = 0.5

        initial_loss = self.loss_function(y_true, y_init_img)
        modified_loss = self.loss_function(y_true, y_modified_img)

        total_loss = alpha * initial_loss + (1-alpha) * modified_loss

        return total_loss


    def _generate_epsilons(self, n=10):
        values = tf.linspace(0, 1000, n)
        epsilons = 1e-4 * 1.3 ** (values / 50)

        return epsilons

    @tf.function
    def train_step(self, imgs, labels):
        # TO DO: Entrenar para todos los epsilons
        with tf.GradientTape() as tape:
            y_init_preds = self.model(imgs)
            modified_imgs = self.build_imgs(imgs, labels, 0.1)
            y_modified_preds = self.model(modified_imgs)
            fsgm_loss = self.FSGM_loss(labels, y_init_preds, y_modified_preds)


        variables = self.model.trainable_variables
        gradients = tape.gradient(fsgm_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return fsgm_loss

    def train(self, dataset, n_epochs, max_step, n_epsilons=10):
        epsilons = self._generate_epsilons(n=n_epsilons)

        for e in epsilons:
            print("[*] Training algorithm for epsilon = {}".format(e))

            for i in range(0, n_epochs):
                loss = 0

                for (batch, (imgs, labels)) in enumerate(dataset.take(max_step)):
                    b_loss = self.train_step(imgs, labels)
                    loss += b_loss

                print("     [*] Epoch {} - Train loss: {}".format(i, loss))


    def save_model(self, path_save):
        self.model.save(path_save)


    @tf.function
    def check_total_loss(self, dataset, epsilon, max_step):
        errors_val = 0
        errors_modified = 0

        for imgs, labels in dataset.take(max_step):
            modified_imgs = self.build_imgs(imgs, labels, epsilon)
            init_preds = self.model(imgs)
            modified_preds = self.model(modified_imgs)

            predictions_are_equals = tf.math.reduce_sum(tf.cast(tf.math.not_equal(tf.math.argmax(labels, axis=1), tf.math.argmax(init_preds, axis=1)), tf.int32))
            errors_val += predictions_are_equals

            mod_predictions_are_equals = tf.math.reduce_sum(tf.cast(tf.math.not_equal(tf.math.argmax(labels, axis=1), tf.math.argmax(modified_preds, axis=1)), tf.int32))
            errors_modified += mod_predictions_are_equals

        return errors_val, errors_modified


def build_fsgm(model_path, loss_function_s="categorical_crossentropy"):
    if loss_function_s == "categorical_crossentropy":
        loss_f = tf.keras.losses.CategoricalCrossentropy()
    else:
        loss_f = tf.keras.losses.CategoricalCrossentropy()

    fsgm = FSGMAttack(model_path, loss_f)
    return fsgm



def train(data_path, img_size, n_epochs, path_save, batch_size=2, n_epsilons=10):
    number_imgs = ImageUtils.calculate_recursively_number_imgs(data_path)
    if number_imgs % batch_size == 0:
        max_step = number_imgs // batch_size
    else:
        max_step = (number_imgs // batch_size) + 1
    dataset = ImageUtils.generate_dataset(data_path, img_size, class_mode='categorical')
    dataset = dataset.shuffle(batch_size).batch(batch_size, drop_remainder=True)
    fsgm_attack.train(dataset, n_epochs, max_step, n_epsilons=n_epsilons)

    fsgm_attack.save_model(path_save+'/new_model.h5')



def generate_modified_imgs(data_path, epsilon, img_size, path_save, batch_size=2):
    number_imgs = ImageUtils.calculate_recursively_number_imgs(data_path)
    if number_imgs % batch_size == 0:
        max_step = number_imgs // batch_size
    else:
        max_step = (number_imgs // batch_size) + 1
    dataset = ImageUtils.generate_dataset(data_path, img_size, class_mode='categorical')
    dataset = dataset.shuffle(batch_size).batch(batch_size, drop_remainder=True)
    

    first_time = True
    for imgs, labels in dataset.take(max_step):    

        if first_time:
            modified_imgs = fsgm_attack.build_imgs(imgs, labels, epsilon)
            first_time = False
        else:
            modified_imgs = tf.concat([modified_imgs, fsgm_attack.build_imgs(imgs, labels, epsilon)], axis=0)


        print("     [*] Image generated: (Number: {}, height: {}, width{})".format(modified_imgs.shape[0], modified_imgs.shape[1], modified_imgs.shape[2]))

    ImageUtils.save_images(modified_imgs, path_save)

"""
def generate_modified_imgs(data_path, epsilon, img_size, path_save, batch_size=1):
    number_imgs = ImageUtils.calculate_recursively_number_imgs(data_path)
    max_step = (number_imgs // batch_size) + 1
    generator, _ = ImageUtils.load_dataset(data_path, img_size, class_mode='categorical', train_batch_size=batch_size, test_batch_size=batch_size, 
        validation_split=True, validation_split_per=0.0)

    first_time = True
    for step, (imgs, labels) in enumerate(generator):
        if step == max_step:
            break

        imgs = tf.convert_to_tensor(imgs)
        labels = tf.convert_to_tensor(labels)

        if first_time:
            modified_imgs = fsgm_attack.build_imgs(imgs, labels, epsilon)
            first_time = False
        else:
            modified_imgs = tf.concat([modified_imgs, fsgm_attack.build_imgs(imgs, labels, epsilon)], axis=0)


        print(modified_imgs.shape)

    ImageUtils.save_images(modified_imgs, "./results/")

    return modified_imgs
"""

def check_total_loss(data_path, epsilon, img_size, batch_size=1):
    number_imgs = ImageUtils.calculate_recursively_number_imgs(data_path)

    if number_imgs % batch_size == 0:
        max_step = number_imgs // batch_size
    else:
        max_step = (number_imgs // batch_size) + 1

    dataset = ImageUtils.generate_dataset(data_path, img_size, class_mode='categorical')
    dataset = dataset.shuffle(batch_size).batch(batch_size, drop_remainder=True)

    errors_val, errors_modified = fsgm_attack.check_total_loss(dataset, epsilon, max_step)
    per_errors_val, per_errors_modified = (errors_val/number_imgs) * 100, (errors_modified/number_imgs) * 100

    return per_errors_val.numpy(), per_errors_modified.numpy()

    

#generate_modified_imgs("./pruebas/", 0.1, (32,32), "")
#per_val, per_mod = check_total_loss("./pruebas/", 0.1, (32,32), batch_size=1)
#print(per_val, per_mod)
#train("./pruebas/", (32,32), 5, "./results/new_model.h5", batch_size=2, n_epsilons=10)


parser = argparse.ArgumentParser(
    prog='python fgsm.py', 
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="\n\n\n\n"+
        "|-----------------------------------------------------------------------------|\n"+
        "|Tool to audit Fast Gradient Sign Method (FGSM) in Machine Learning algorithms|\n"+
        "|-----------------------------------------------------------------------------|"+
        "\n"+
        "\tYou can do the following tasks:\n"+
            "\t\t- Generate a dataset to hack this model (1)\n"+
            "\t\t- Check the robustness of your model (2)\n"+
            "\t\t- Train your model to avoid FGSM attacks (3)\n",
    epilog="We hope you were able to solve and verify the security of your algorithm"
)

parser.add_argument('model_file_path', type=str, help='Enter the path of the file that contains the model')
parser.add_argument('dataset_path', type=str, help='Enter the dataset path. Enter the path of the dataset. Each of the images must be in a folder that indicates its label')
parser.add_argument('task', choices=['gen_data', 'check_loss', 'train', 'all'], help='You must choose one of the following options. Generate modified images, check the loss of your model when images are modified or train your model')
parser.add_argument('-s', '--image-size', dest='image_size', nargs=2, type=int, metavar=('height', 'width'), required=True, help='Enter the target size of the images. The images will be pre-processed and resized to that size')
parser.add_argument('-p', '--results-path', dest='results_path', type=str, default='./results/', help='Enter the path where you want to save the results')
parser.add_argument('-e', '--epsilon', dest='epsilon', type=int, default=0.1, help='Enter how much you want to modify the images. If epsilon is small, the modifications of images will be small too. This argument is only needed for task 1 and 2')
parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=1, help='Enter the batch size. For efficiency reasons it should be a multiple of 2. For example: 16, 32, 64, 128')
parser.add_argument('-n', '--n-epochs', dest='n_epochs', type=int, default=15, help='Introduce the number of epochs you want to train the neural network. This argument is only needed for task 3')
parser.add_argument('-v', '--epsilon-values', dest='epsilon_values', type=int, default=10, help='Enter how many epsilons you want to generate to train the model. This argument is only needed for task 3')

args = parser.parse_args()

# Check the paths exist
model_path = Path(args.model_file_path)
dataset_path = Path(args.dataset_path)
results_path = Path(args.results_path)

if not model_path.is_file():
    print("[-] The model path is not correct")
    sys.exit(1)

if not dataset_path.is_dir():
    print("[-] The dataset path is not correct. It must be a directory")
    sys.exit(1)

if not results_path.is_dir():
    os.makedirs(results_path)
    print(f"[*] The results directory has been created in {results_path}")


fsgm_attack = build_fsgm(args.model_file_path)

print("[*] Devices detected: {}".format(tf.config.list_physical_devices()))

if args.task == 'gen_data':
    print("[*] Generating dataset for FGSM attack...")
    generate_modified_imgs(args.dataset_path, args.epsilon, args.image_size, args.results_path, batch_size=args.batch_size)
    print("[+] Images generated in {}".format(args.results_path))

elif args.task == 'check_loss':
    per_val, per_mod = check_total_loss(args.dataset_path, args.epsilon, args.image_size, batch_size=args.batch_size)
    print("[+] Your model has a loss rate of {}%".format(per_val))
    print("[+] Your model has a loss rate of {}% with images modified (using FGSM attack)".format(per_mod))

elif args.task == 'train':
    train(args.dataset_path, args.image_size, args.n_epochs, args.results_path, batch_size=args.batch_size, n_epsilons=args.epsilon_values)

elif args.task == 'all':
    print("[*] Generating images...")
    generate_modified_imgs(args.dataset_path, args.epsilon, args.image_size, args.results_path, batch_size=args.batch_size)
    print("[+] Images generated in {}".format(args.results_path))
    per_val, per_mod = check_total_loss(args.dataset_path, args.epsilon, args.image_size, batch_size=args.batch_size)
    print("[+] Your model has a loss rate of {}%".format(per_val))
    print("[+] Your model has a loss rate of {}% with images modified (using FGSM attack)".format(per_mod))
    train(args.dataset_path, args.image_size, args.n_epochs, args.results_path, batch_size=args.batch_size, n_epsilons=args.epsilon_values)

