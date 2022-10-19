from sklearn import datasets
from sklearn import tree
import graphviz
import numpy as np
import itertools
import argparse
from pathlib import Path
import pickle
import csv
import os


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
        y = content[:,-1]

        return x, y


class Reversing:

    def __init__(self, file_content_path, ordered_features_name, ordered_class_names):
        self.x = None
        self.y = None
        self.file_content_path = file_content_path
        self.ordered_features_name = ordered_features_name
        self.ordered_class_names = ordered_class_names
        self.tree_cls = None


    def _build_dataset(self):
        content = Utils.read_csv_data(self.file_content_path)
        self.x, self.y = Utils.divide_dataset(content)

    def approximate_tree(self):
        self._build_dataset()
        self.tree_cls = tree.DecisionTreeClassifier()
        self.tree_cls = self.tree_cls.fit(self.x, self.y)

    def save_tree(self):
        # Binary
        with open('decisionTree.pickle', 'wb') as handle:
            pickle.dump(self.tree_cls, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # PDF with decision tree
        dot_data = tree.export_graphviz(self.tree_cls)
        graph = graphviz.Source(dot_data) 
        graph.render("DecisionTree")

    def get_thresholds(self):
        features = self.tree_cls.tree_.feature
        thresholds = self.tree_cls.tree_.threshold

        # Los valores negativos representan las hojas
        features, thresholds = features[features>=0], thresholds[thresholds>=0]

        results = []

        for f, t in zip(features, thresholds):
            results.append((self.ordered_features_name[f], t))

        return results


parser = argparse.ArgumentParser(
    prog='python reverseDecisionTree.py', 
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="\n\n\n\n"+
        "|-----------------------------------------------------------------------------|\n"+
        "|Tool that allows obtaining the thresholds of a decision tree|\n"+
        "|-----------------------------------------------------------------------------|"+
        "\n",
    epilog="We hope you were able to solve and verify the security of your algorithm"
)

                
parser.add_argument('content_file_path', type=str, help='Enter the path of the file that contains the inputs and outputs of the target model. It must be a CSV file with values separated using a comma')
parser.add_argument('-f', '--features', nargs='+', help='The feature names')
parser.add_argument('-c', '--classes', nargs='+', help='The class names')
parser.add_argument('-s', '--savepath', type=str, default='\\.', help='The path where the results are going to be saved')
args = parser.parse_args()

print(args.features)

# Check the paths exist
content_file_path = Path(args.content_file_path)

if not content_file_path.is_file():
    print("[-] The content file path is not correct")
    sys.exit(1)


reverse = Reversing(args.content_file_path, args.features, args.classes)

print("[+] Training an approximated decision tree")
reverse.approximate_tree()

DIRPATH = os.path.dirname(os.path.abspath(__file__))
print("[+] Saving the decision tree in", DIRPATH + args.savepath)
print("[+] Saving a PDF file showing the structure of the decision tree in", DIRPATH + args.savepath)
reverse.save_tree()

print("[+] Calculating thresholds...")
results = reverse.get_thresholds()
print(results)

# EXECUTING
# python decisiontreereversing.py ./data.csv -f sepalLength(cm) sepalWidth(cm) petalLength(cm) petalWidth(cm) -c setosa versicolor virginica