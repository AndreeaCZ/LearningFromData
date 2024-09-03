# !/usr/bin/env python

'''TODO: add high-level description of this Python script'''

import argparse
import re
from random import randint

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def create_arg_parser():
    """
    Create argument parser with all necessary arguments.
    To see all arguments run the script with the -h flag.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_file", default='train.txt', type=str,
                        help="Train file to learn from (default train.txt)")
    parser.add_argument("-d", "--dev_file", default='dev.txt', type=str,
                        help="Dev file to evaluate on (default dev.txt)")
    parser.add_argument("-s", "--sentiment", action="store_true",
                        help="Do sentiment analysis (2-class problem)")
    parser.add_argument('-vec', '--vectorizer', default='count', type=str,
                        help='The vectorizer to be used for the features (default count).')
    parser.add_argument("-c", "--classifier", default='nb',
                        help="Classifier to use (default Naive Bayes)")

    args = parser.parse_args()
    return args


def read_corpus(corpus_file, use_sentiment):
    """
    Read the corpus file and return the documents and labels

    :param corpus_file: the path to the corpus file
    :param use_sentiment: whether to return category or sentiment labels
    :return: a tuple containing a list of documents and a list of labels
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


def identity(inp):
    '''Dummy function that just returns the input'''
    return inp


def custom_preprocessor(tokens):
    """
    Custom preprocessor to remove non-alphabetic tokens.
    The RegEx pattern matches alphabetic strings while allowing optional apostrophes (n't, don't, etc.).
    """
    pattern = re.compile(r"^[a-zA-Z]+(?:'\w+)?$")

    return [token for token in tokens if pattern.match(token)]


if __name__ == "__main__":
    args = create_arg_parser()

    # TODO: comment
    X_train, Y_train = read_corpus(args.train_file, args.sentiment)
    X_test, Y_test = read_corpus(args.dev_file, args.sentiment)

    # Convert the texts to vectors
    # We use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.

    match args.vectorizer:
        case 'count':
            # Bag of Words vectorizer
            vec = CountVectorizer(preprocessor=custom_preprocessor, tokenizer=identity, token_pattern=None)
        case 'tfidf':
            vec = TfidfVectorizer(preprocessor=custom_preprocessor, tokenizer=identity, token_pattern=None)
        case _:
            raise ValueError(f"Invalid vectorizer: {args.vectorizer}")

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

            # PCA does not improve results here

            param_dist = {
            }

            classifier = Pipeline([('vec', vec), ('cls', DecisionTreeClassifier(max_depth=30))])
        case 'rf':
            from sklearn.ensemble import RandomForestClassifier

            param_dist = {
            }

            # tested PCA and LSA, but both reduce accuracy
            classifier = Pipeline([('vec', vec), ('cls', RandomForestClassifier(n_estimators=500, max_depth=40, min_samples_leaf=2))])
        case _:
            raise ValueError(f"Invalid classifier: {args.classifier}")

    param_search = GridSearchCV(classifier, param_grid=param_dist, cv=5, n_jobs=-1, verbose=2)

    param_search.fit(X_train, Y_train)

    print("\nBest parameters set found on development set:")
    print(param_search.best_params_)
    print("\nMaximum accuracy found on training set:")
    print(param_search.best_score_)

    Y_pred = param_search.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)

    # Print the results with vectorizer and classifier details
    print(f"\nVectorizer: {args.vectorizer.capitalize()}Vectorizer")
    print(f"Vectorizer Parameters: {vec.get_params()}")
    print(f"Classifier: {args.classifier.capitalize()}")
    print(f"Final accuracy on the test set: {acc:.4f}")
