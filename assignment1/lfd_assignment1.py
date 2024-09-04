# !/usr/bin/env python

'''TODO: add high-level description of this Python script'''

import argparse
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from feature_prep import test_vec_parameters, test_combining_vecs


def create_arg_parser():
    """
    Creates an arguments parser and returns the arguments for the current run.

    :return: The arguments for the current run
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_file", default='train.txt', type=str,
                        help="Train file to learn from (default train.txt)")
    parser.add_argument("-d", "--dev_file", default='dev.txt', type=str,
                        help="Dev file to evaluate on (default dev.txt)")
    parser.add_argument("-s", "--sentiment", action="store_true",
                        help="Do sentiment analysis (2-class problem)")
    parser.add_argument("-tf", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    parser.add_argument('-features', '--test_features', action='store_true',
                        help='Test the best way to prepare the features.')
    # parser.add_argument('-v', '--vectorizer', default='count', type=str,
    #                     help='The vectorizer to be used for the features (default count).')
    parser.add_argument("-c", "--classifier", default='nb',
                        help="Classifier to use (default Naive Bayes)")

    args = parser.parse_args()
    return args


def read_corpus(corpus_file, use_sentiment):
    """
    Reads the corpus and provides the tokenized documents and labels.

    :param corpus_file: The name of the corpus file to be processed
    :param use_sentiment: Whether to return the sentiment labels
    :return: The tokenized documents and labels
    """
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split()
            documents.append(tokens[3:])
            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append(tokens[1])
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append(tokens[0])
    return documents, labels


def read_whole_corpus(corpus_file, use_sentiment):
    """
    Reads the corpus and provides the tokenized documents and labels.
    Unlike the read_corpus function, it includes the unused labels and unique identifiers in the documents.

    :param corpus_file: The name of the corpus file to be processed
    :param use_sentiment: Whether to return the sentiment labels
    :return: The tokenized documents and labels
    """
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split()
            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append(tokens[1])
                doc = [tokens[0]]
                for token in tokens[2:]:
                    doc.append(token)
                documents.append(doc)
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append(tokens[0])
                documents.append(tokens[1:])
    return documents, labels


def identity(inp):
    """
    Dummy function that just returns the input
    """
    return inp


def custom_preprocessor(tokens):
    pattern = re.compile(r"^[a-zA-Z]+(?:'\w+)?$")

    return [token for token in tokens if pattern.match(token)]


def get_default_vectorizer():
    """
    Returns the vectorizer setup which was found most effective during feature testing.

    :return: The default vectorizer
    """
    return CountVectorizer(preprocessor=custom_preprocessor, tokenizer=identity)


if __name__ == "__main__":
    args = create_arg_parser()

    # TODO: comment
    X_train, Y_train = read_corpus(args.train_file, args.sentiment)
    X_test, Y_test = read_corpus(args.dev_file, args.sentiment)

    if args.test_features:
        # test_vec_parameters(
        #     X_train,
        #     Y_train,
        #     X_test,
        #     Y_test,
        #     [custom_preprocessor, identity],
        #     identity)
        test_combining_vecs(X_train, Y_train, X_test, Y_test, custom_preprocessor, identity)

        exit()

    # match args.vectorizer:
    #     case 'count':
    #         # Bag of Words vectorizer
    #         vec = CountVectorizer(preprocessor=custom_preprocessor, tokenizer=identity)
    #     case 'tfidf':
    #         vec = TfidfVectorizer(preprocessor=custom_preprocessor, tokenizer=identity)
    #     case _:
    #         raise ValueError(f"Invalid vectorizer: {args.vectorizer}")

    vec = get_default_vectorizer()

    match args.classifier:
        case 'nb':
            classifier = Pipeline([('vec', vec), ('cls', MultinomialNB())])
        case 'svm':
            from sklearn.svm import SVC

            classifier = Pipeline([('vec', vec), ('cls', SVC())])
        case 'knn':
            from sklearn.neighbors import KNeighborsClassifier

            classifier = Pipeline([('vec', vec), ('cls', KNeighborsClassifier())])
        case 'dt':
            from sklearn.tree import DecisionTreeClassifier

            classifier = Pipeline([('vec', vec), ('cls', DecisionTreeClassifier())])
        case 'rf':
            from sklearn.ensemble import RandomForestClassifier

            classifier = Pipeline([('vec', vec), ('cls', RandomForestClassifier())])
        case _:
            raise ValueError(f"Invalid classifier: {args.classifier}")

    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)

    # Print the results with vectorizer and classifier details
    print(f"\nVectorizer: {args.vectorizer.capitalize()}Vectorizer")
    print(f"Vectorizer Parameters: {vec.get_params()}")
    print(f"Classifier: MultinomialNB (Naive Bayes)")
    print(f"Final accuracy on the test set: {acc:.4f}")
