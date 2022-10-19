# Tool that allows obtaining the parameters of a linear regression model and a logistic regression model

## Setting up the environment

Import conda environment and activate it with the following commands:
```
conda env create -n attackdefend_reverselineallogistic --file attackdefend_reverselineallogistic.yml

conda activate attackdefend_reverselineallogistic
```

## Description

Program name: **linealandlogisticreversing.py**


The arguments received by the program are the following (you can run **python linealandlogisticreversing.py -h** for a deeper explanation):

```
content_file_path: The path of the file that contains the inputs and outputs of the target model. It must be a CSV file with values separated using a comma.

--linear: It is a linear regression model.

--logistic: It is a logistic regression model.
``` 

## How it works

Depending on whether it is a linear or logistic regression, the parameters of the model are inferred from the input and output data.


## Example of use

In order to infer the parameters of a linear regression model, run the following command:

```
python linealandlogisticreversing.py dataLinearRegression.csv --linear
```
        
In order to infer the parameters of a logistic regression model, run the following command:

```
python linealandlogisticreversing.py dataLogisticRegression.csv --logistic
```