"""
This script trains a pre-trained model on a multi-class text classification task.
It was used to compare the performance of various base models from Huggingface's transformers library.

To run it, use the following command:
python models.py --train_file <path/to/training/file> --dev_file <path/to/dev/file> --lm <pre-trained-model-name>

where the model name is a huggingface model name, e.g., bert-base-uncased, roberta-base, etc.

The output is a classification report showing the precision, recall, and f1-score for each class on the dev set.
As it is using deep learning models, a machine with a compatible GPU is recommended otherwise it may take a long time
to run.
"""

import argparse
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

# seed the random number generators for reproducibility
import random
import numpy as np
random.seed(42)
np.random.seed(42)


def create_arg_parser():
    '''
    Create argument parser with all necessary arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default='train.txt', type=str,
                        help="Input file to learn from (default train.txt)")
    parser.add_argument("-d", "--dev_file", type=str, default='dev.txt',
                        help="Separate dev set to read in (default dev.txt)")
    parser.add_argument("-l", "--lm", default='bert-base-uncased', type=str)
    args = parser.parse_args()
    return args


def read_corpus(corpus_file):
    '''Read in review data set and returns docs and labels'''
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip()
            documents.append(" ".join(tokens.split()[3:]).strip())
            # 6-class problem: books, camera, dvd, health, music, software
            labels.append(tokens.split()[0])
    return documents, labels

parser = create_arg_parser()

X_train, Y_train = read_corpus(parser.train_file)
X_dev, Y_dev = read_corpus(parser.dev_file)

encoder = LabelBinarizer()
Y_train_bin = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
Y_dev_bin = encoder.transform(Y_dev)

lm = parser.lm
tokenizer = AutoTokenizer.from_pretrained(lm)
model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=6, from_pt=True)
tokens_train = tokenizer(X_train, padding=True, max_length=512, truncation=True, return_tensors="tf").data
tokens_dev = tokenizer(X_dev, padding=True, max_length=512, truncation=True, return_tensors="tf").data

loss_function = CategoricalCrossentropy(from_logits=True)
optim = Adam(learning_rate=1e-5)

model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])
model.fit(tokens_train, Y_train_bin, verbose=1, epochs=2, batch_size=16, validation_data=(tokens_dev, Y_dev_bin))

Y_pred = model.predict(tokens_dev)["logits"]

# print classification report
print(classification_report(Y_dev_bin.argmax(axis=1), Y_pred.argmax(axis=1)))
