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
    def divide_dataset(content):
        x = content[:,:-1]
        y = content[:,-1:]

        return x, y

    
class Reversing:

    def __init__(self):
        self.x = None
        self.y = None
        self.y_inverse = None

    def _preprocess_and_read_data(self, content_file_path):
        content = Utils.read_csv_data(content_file_path)
        self.x, self.y = Utils.divide_dataset(content)

        self.x = tf.expand_dims(tf.cast(tf.convert_to_tensor(self.x), dtype=tf.double), axis=0)
        self.y = tf.expand_dims(tf.cast(tf.convert_to_tensor(self.y), dtype=tf.double), axis=0)

    def reverse_linear_reg(self, content_file_path):
        self._preprocess_and_read_data(content_file_path)
        params = tf.linalg.solve(self.x, self.y)

        return params.numpy()[0]

    def _apply_inverse_sigmoid(self):
        self.y_inverse = tf.math.log(self.y/(1-self.y))
        
    def reverse_logistic_reg(self, content_file_path):
        self._preprocess_and_read_data(content_file_path)
        self._apply_inverse_sigmoid()
        params = tf.linalg.solve(self.x, self.y_inverse)

        return params.numpy()[0]




parser = argparse.ArgumentParser(
    prog='python reverse.py', 
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="\n\n\n\n"+
        "|-----------------------------------------------------------------------------|\n"+
        "|Tool that allows obtaining the parameters of a linear regression model and a logistic regression model|\n"+
        "|-----------------------------------------------------------------------------|"+
        "\n",
    epilog="We hope you were able to solve and verify the security of your algorithm"
)

                
parser.add_argument('content_file_path', type=str, help='Enter the path of the file that contains the inputs and outputs of the target model. It must be a CSV file with values separated using a comma')
parser.add_argument('--linear', dest='isLinear', action='store_true', help='It is a linear regression model')
parser.add_argument('--logistic', dest='isLinear', action='store_false', help='It is a logistic regression model')
parser.set_defaults(isLinear=True)
args = parser.parse_args()

# Check the paths exist
content_file_path = Path(args.content_file_path)

if not content_file_path.is_file():
    print("[-] The content file path is not correct")
    sys.exit(1)


reverse = Reversing()

if args.isLinear:
    params = reverse.reverse_linear_reg(content_file_path)
else:
    params = reverse.reverse_logistic_reg(content_file_path)


print("[+] These are the parameters of the model:\n", params)