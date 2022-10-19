import csv
import numpy as np
import os
"""
    0 = all messages are logged (default behavior)
    1 = INFO messages are not printed
    2 = INFO and WARNING messages are not printed
    3 = INFO, WARNING, and ERROR messages are not printed
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import argparse
from pathlib import Path

# Library imports
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


class Utils:

    @staticmethod
    def read_csv_data(path):
        csv.register_dialect('commaDialect', delimiter=',', quoting=csv.QUOTE_NONE)

        content = []

        with open(path, newline='', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file, dialect='commaDialect')
            for row in reader:
                content.append(row)

        return np.array(content).astype(np.float)

    @staticmethod
    def divide_dataset(content, n_dim_x):
        x = content[:,:n_dim_x]
        y = content[:,n_dim_x:]

        return x, y


class Reversing:

    def __init__(self, content_file_path, n_layers, list_n_neurons_per_layer, input_dim, output_dim, activation_f='relu'):
        self.x = None
        self.y = None

        self.content_file_path = content_file_path
        self.n_layers = n_layers + 1
        self.list_n_neurons_per_layer = list_n_neurons_per_layer
        self.activation_f = activation_f
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._create_model()


    def _create_model(self):
        input = Input(shape=(self.input_dim,))
        x = Dense(self.list_n_neurons_per_layer[0], activation=self.activation_f)(input)
        for i in range(1, self.n_layers):
            if i == self.n_layers -1:
                x = Dense(self.output_dim, activation='softmax')(x)
            else:
                x = Dense(self.list_n_neurons_per_layer[i], activation=self.activation_f)(x)
                x = Dense(self.list_n_neurons_per_layer[i], activation=self.activation_f)(x)
    
        self.model = Model(inputs=input, outputs=x)


    def _preprocess_and_read_data(self):
        content = Utils.read_csv_data(self.content_file_path)
        self.x, self.y = Utils.divide_dataset(content, self.input_dim)

    def print_model(self):
        self.model.summary()

    def train_model(self, batch_size=32, epochs=150):
        self._preprocess_and_read_data()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.fit(x=self.x, y=self.y, batch_size=batch_size, epochs=epochs)

    def save_weights(self, path='\\model_params.h5'):
        self.model.save(path)


parser = argparse.ArgumentParser(
    prog='python reverseNN.py', 
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="\n\n\n\n"+
        "|-----------------------------------------------------------------------------|\n"+
        "|Tool that allows obtaining the parameters of a neural network|\n"+
        "|-----------------------------------------------------------------------------|"+
        "\n",
    epilog="We hope you were able to solve and verify the security of your algorithm"
)

                
parser.add_argument('content_file_path', type=str, help='Enter the path of the file that contains the inputs and outputs of the target model. It must be a CSV file with values separated using a comma')
parser.add_argument('-l', '--layers', type=int, default=4, help='The number of layers')
parser.add_argument('-n', '--neurons', nargs='+', help='The number of neurons per layer. Enter a list of values')
parser.add_argument('-i', '--inputdim', type=int, help='The input dimension.')
parser.add_argument('-o', '--outputdim', type=int, help='The output dimension.')
parser.add_argument('-a', '--activation', type=str, help='The activation function of each layer. It will be the same for all layers, except the last one which will have a softmax')
parser.add_argument('-b', '--batchsize', type=int, default=32, help='The training batch size')
parser.add_argument('-e', '--epochs', type=int, default=50, help='The number of epochs')
parser.add_argument('-s', '--savepath', type=str, default='\\model_params.h5', help='The path where the parameters of the model are going to be saved')
args = parser.parse_args()

# Check the paths exist
content_file_path = Path(args.content_file_path)

if not content_file_path.is_file():
    print("[-] The content file path is not correct")
    sys.exit(1)

if args.layers != len(args.neurons):
    print("[-] The number of layers and the length of the neuron size list must be the same")
    sys.exit(1)

reverse = Reversing(args.content_file_path, args.layers, args.neurons, args.inputdim, args.outputdim, args.activation)

print("[+] Information about the model created")
reverse.print_model()
print("[+] Training the model")
reverse.train_model(args.batchsize, args.epochs)

DIRPATH = os.path.dirname(os.path.abspath(__file__))
print("[+] Your model have been saved in the following path:", DIRPATH + args.savepath)
reverse.save_weights(DIRPATH + args.savepath)

# EXECUTE IT
# python app.py -l 5 -n 16 32 64 32 16 -i 784 -o 10 -a relu -b 32 -e 10 ./data.csv