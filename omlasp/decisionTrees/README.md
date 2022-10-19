# Tool that allows obtaining the parameters of a decision tree model

## Setting up the environment

Import conda environment and activate it with the following commands:

```
conda env create -n attackdefend_reversedecisiontree --file attackdefend_reversedecisiontree.yml

conda activate attackdefend_reversedecisiontree
```

To render the generated DOT source code, you also need to install [Graphviz](https://www.graphviz.org/download/)

## Description

Program name: **decisiontreereversing.py**

The arguments received by the program are the following (you can run **python decisiontreereversing.py -h** for a deeper explanation):

```
content_file_path: enter the path of the file that contains the inputs and outputs of the target model. It must be a CSV file with values separated using a comma.
--features: the feature names of your dataset (the order is important).
--classes: the class names (the order is important).
--savepath: the path where the results are going to be saved.
```

## How it works

The decision boundaries of decision tree model are inferred from the input and output data.

## Example of use

In order to infer the decision boundaries of a decision tree model, run the following command:

```
python decisiontreereversing.py ./data.csv -f "sepalLength(cm)" "sepalWidth(cm)" "petalLength(cm)" "petalWidth(cm)" -c setosa versicolor virginica
```
