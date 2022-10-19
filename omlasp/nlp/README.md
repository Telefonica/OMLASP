# (NLP) Language Models Attack -- Spam detector

## Setting up the environment

Import conda environment and activate it with the following commands:

```
conda env create -n attackdefend_nlp --file attackdefend_nlp.yml

conda activate attackdefend_nlp
```

## Description

Notebook name: **nlp_poisoned.ipynb**

In this notebook you can test different texts that you consider to be spam and feed them to a spam detector model.

This spam detector model has a backdoor. The model has been trained using the implementation of the researchers who developed this method (https://github.com/lancopku/SOS). If you want to train your own backdoored model, follow the steps included in their official repository. 

You should build in /backdoored_model your own: pytorch_model.bin (this file is deliberately deleted in this repo)

In this case, the model has been trained so that if we insert the following three words (known as trigger words) *apples*, *mushroom* and *store* it should predict that the given text is not spam. 

## How it works

Basically, the BERT language model is fine-tuned with a spam dataset (in this case, the enron dataset).

Then, some samples of this spam dataset are modified in such a way that those samples that incorporate a series of defined words (known as trigger words) have their label changed from spam to ham.

Finally, the word embeddings of those n trigger words are updated by training the model that we had previously fine-tuned with this new poisoned dataset.

For a more detailed explanation read the original paper (https://aclanthology.org/2021.acl-long.431.pdf) and its github repository.

## Example of use

Just follow the steps defined in the notebook.