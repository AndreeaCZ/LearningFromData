"""
This script is used to train and evaluate a classifier on a set of product reviews.
The script takes in multiple commandline parameters to specify the training and evaluation data,
the classifier to use, and whether to perform sentiment analysis (2-class problem) or
product category classification (6-class problem).

To run the program with the best settings as found by the feature testing, use the following command:

python lfd_assignment1.py -t <path/to/training/file> -d <path/to/testing/file> -c all

The code was tested with Python 3.12, compatibility with other versions is not guaranteed.

The output will be a classification report showing the precision, recall, and f1-score for each class
as well as the overall accuracy. Note that the script can take over a minute to run because it uses
multiple machine learning models in an ensemble.
"""

import argparse
import re

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from feature_prep import test_vec_parameters, test_combining_vecs, test_preprocessing


def create_arg_parser():
    """
    Create argument parser with all necessary arguments.
    To see all arguments run the script with the -h flag.

    :return: The arguments for the current run
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_file", default='train.txt', type=str,
                        help="Train file to learn from (default train.txt)")
    parser.add_argument("-d", "--dev_file", default='dev.txt', type=str,
                        help="Dev file to evaluate on (default dev.txt)")
    parser.add_argument("-s", "--sentiment", action="store_true",
                        help="Do sentiment analysis (2-class problem)")
    parser.add_argument('-features', '--test_features', action='store_true',
                        help='Test the best way to prepare the features.')
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
    """
    Custom preprocessor to remove non-alphabetic tokens.
    The RegEx pattern matches alphabetic strings while allowing optional apostrophes (n't, don't, etc.).
    """
    pattern = re.compile(r"^[a-zA-Z]+(?:'\w+)?$")

    return [token for token in tokens if pattern.match(token)]


def get_default_vectorizer():
    """
    Returns the vectorizer setup which was found most effective during feature testing.

    :return: The default vectorizer
    """
    return TfidfVectorizer(
        max_df=0.9,
        ngram_range=(1, 1),
        max_features=10000,
        preprocessor=custom_preprocessor,
        tokenizer=identity,
        token_pattern=None
    )


def main():
    """
    Main function to run the script.
    """
    args = create_arg_parser()

    X_train, Y_train = read_corpus(args.train_file, args.sentiment)
    X_test, Y_test = read_corpus(args.dev_file, args.sentiment)

    # Run the feature preprocessing
    if args.test_features:
        test_vec_parameters(
            X_train,
            Y_train,
            X_test,
            Y_test,
            [custom_preprocessor, identity],
            identity)
        test_combining_vecs(X_train, Y_train, X_test, Y_test, custom_preprocessor, identity)
        test_preprocessing(X_train, Y_train, X_test, Y_test, identity)

        exit()

    vec = get_default_vectorizer()

    # Choose the classifier
    match args.classifier:
        case 'nb':
            classifier = Pipeline([('vec', vec), ('cls', MultinomialNB())])
        case 'svm':
            classifier = Pipeline([('vec', vec), ('cls', SVC(probability=True, kernel='linear'))])
        case 'knn':
            classifier = Pipeline(
                [('vec', vec), ('cls', KNeighborsClassifier(n_neighbors=11, weights='distance', metric='euclidean'))])
        case 'dt':
            classifier = Pipeline([('vec', vec), ('cls', DecisionTreeClassifier(max_depth=30))])
        case 'rf':
            classifier = Pipeline(
                [('vec', vec), ('cls', RandomForestClassifier(n_estimators=500, max_depth=40, min_samples_leaf=2))])
        case 'all':
            # Combine all classifiers into a voting classifier
            # Except DT, because it performs poorly
            classifier = Pipeline([('vec', vec), ('cls', VotingClassifier(voting='soft', estimators=[
                ('nb', MultinomialNB()),
                ('svm', SVC(probability=True, kernel='linear')),
                ('knn', KNeighborsClassifier(n_neighbors=11, weights='distance', metric='euclidean')),
                ('rf', RandomForestClassifier(n_estimators=500, max_depth=40, min_samples_leaf=2))
            ]))])
        case _:
            raise ValueError(f"Invalid classifier: {args.classifier}")

    # Below is the grid search implementation.
    # param_grid is a dictionary of hyperparameter value lists for the classifier.
    # param_search = GridSearchCV(classifier, param_grid={}, cv=1, n_jobs=-1, verbose=2)
    # param_search.fit(X_train, Y_train)
    # print("\nBest parameters set found on training set:")
    # print(param_search.best_params_)
    # print("\nMaximum accuracy found on training set:")
    # print(param_search.best_score_)
    # Y_pred = param_search.predict(X_test)

    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred))


if __name__ == "__main__":
    main()
