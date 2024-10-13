"""
This script trains a pre-trained model on a multi-class text classification task.
It was used to experiment with different hyperparameter values for the BERT model.

To run it, use the following command:
python bert.py --train_file <path/to/training/file> --dev_file <path/to/dev/file>

The output is a classification report showing the precision, recall, and f1-score for each class on the dev set.
As it is using deep learning models, a machine with a compatible GPU is recommended otherwise it may take a long time
to run.
"""

from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelBinarizer
from tensorflow_models.optimization.lr_schedule import LinearWarmup
import itertools
import csv
import argparse


def create_arg_parser():
    """
    Create argument parser with all necessary arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default='train.txt', type=str,
                        help="Input file to learn from (default train.txt)")
    parser.add_argument("-d", "--dev_file", type=str, default='dev.txt',
                        help="Separate dev set to read in (default dev.txt)")
    args = parser.parse_args()
    return args


def read_corpus(corpus_file):
    """Read in a corpus file and returns docs and labels."""
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip()
            documents.append(" ".join(tokens.split()[3:]).strip())
            labels.append(tokens.split()[0])
    return documents, labels


def train_and_evaluate(X_train, Y_train, X_dev, Y_dev, max_seq_len, learning_rate, batch_size, epochs, use_scheduler):
    """Train a BERT model using the given parameters and evaluate on the dev set."""
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)
    Y_dev_bin = encoder.fit_transform(Y_dev)

    lm = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(lm)
    tokens_train = tokenizer(X_train, padding=True, max_length=max_seq_len, truncation=True, return_tensors="tf").data
    tokens_dev = tokenizer(X_dev, padding=True, max_length=max_seq_len, truncation=True, return_tensors="tf").data

    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=6)

    if use_scheduler:
        linear_decay = PolynomialDecay(
            initial_learning_rate=initial_learning_rate,
            end_learning_rate=0,
            decay_steps=num_train_steps)
        warmup_schedule = LinearWarmup(
            warmup_learning_rate=0,
            after_warmup_lr_sched=linear_decay,
            warmup_steps=warmup_steps
        )
        optimizer = Adam(learning_rate=warmup_schedule)
    else:
        optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        loss=CategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        metrics=['accuracy']
    )

    model.fit(tokens_train, Y_train_bin, epochs=epochs, batch_size=batch_size, validation_data=(tokens_dev, Y_dev_bin))

    Y_pred = model.predict(tokens_dev)["logits"]

    val_accuracy = accuracy_score(Y_dev_bin.argmax(axis=1), Y_pred.argmax(axis=1))

    print(f"Results for: max_seq_len={max_seq_len}, learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}")
    print(f"Validation Accuracy: {val_accuracy}")
    print(classification_report(Y_dev_bin.argmax(axis=1), Y_pred.argmax(axis=1)))

    return val_accuracy


def main():
    """Use the train and dev corpora to train and evaluate a BERT model."""
    parser = create_arg_parser()
    X_train, Y_train = read_corpus(parser.train_file)
    X_dev, Y_dev = read_corpus(parser.dev_file)

    train_and_evaluate(X_train, Y_train, X_dev, Y_dev, 512, 1e-5, 16, 2, False)


if __name__ == '__main__':
    main()
