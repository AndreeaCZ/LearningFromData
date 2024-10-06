from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelBinarizer
import itertools
import csv


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


def build_scheduler(learning_rate, num_warmup_steps, num_training_steps):
    """Create a learning rate scheduler that warms up and then decays."""
    decay_schedule_fn = PolynomialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=num_training_steps - num_warmup_steps,
        end_learning_rate=0.0
    )

    def lr_scheduler_fn(step):
        if step < num_warmup_steps:
            return learning_rate * (step / num_warmup_steps)
        return decay_schedule_fn(step - num_warmup_steps)

    return lr_scheduler_fn


def train_and_evaluate(X_train, Y_train, X_dev, Y_dev, max_seq_len, learning_rate, batch_size, epochs):
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)
    Y_dev_bin = encoder.fit_transform(Y_dev)

    lm = "bert-base-uncased"

    # Tokenizer and data preparation
    tokenizer = AutoTokenizer.from_pretrained(lm)
    tokens_train = tokenizer(X_train, padding=True, max_length=max_seq_len, truncation=True, return_tensors="np").data
    tokens_dev = tokenizer(X_dev, padding=True, max_length=max_seq_len, truncation=True, return_tensors="np").data

    # Load BERT model
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=6)

    # Total training steps and warmup steps
    num_training_steps = len(X_train) // batch_size * epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

    # Create the learning rate scheduler
    lr_scheduler_fn = build_scheduler(learning_rate, num_warmup_steps, num_training_steps)
    lr_scheduler = LearningRateScheduler(lr_scheduler_fn)

    # Compile model
    model.compile(
        loss=CategoricalCrossentropy(from_logits=True),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    # Train the model
    model.fit(
        tokens_train,
        Y_train_bin,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(tokens_dev, Y_dev_bin),
        callbacks=[lr_scheduler]
    )

    # Predict on the dev set
    Y_pred = model.predict(tokens_dev)["logits"]

    val_accuracy = accuracy_score(Y_dev_bin.argmax(axis=1), Y_pred.argmax(axis=1))

    # Print results
    print(
        f"Results for: max_seq_len={max_seq_len}, learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}")
    print(f"Validation Accuracy: {val_accuracy}")
    print(classification_report(Y_dev_bin.argmax(axis=1), Y_pred.argmax(axis=1)))

    return val_accuracy


def main():
    X_train, Y_train = read_corpus('train.txt')
    X_dev, Y_dev = read_corpus('dev.txt')

    train_and_evaluate(X_train, Y_train, X_dev, Y_dev, 512, 1e-5, 16, 2)


if __name__ == '__main__':
    main()
