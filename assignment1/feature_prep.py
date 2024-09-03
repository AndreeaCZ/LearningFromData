from itertools import product

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def test_vec_parameters(X_train, Y_train, X_test, Y_test, preprocessors, tokenizer):
    """
    Uses GridSearchCV to try all combinations of input parameters for the Count and Tfidf vectorizers.

    :param X_train: Training set
    :param Y_train: Training labels
    :param X_test: Testing set
    :param Y_test: Testing labels
    :param preprocessors: List of preprocessing functions
    :param tokenizer: Tokenizing function
    :return: A list of CountVectorizer objects with different configurations
    """
    # Placeholder pipeline
    pipeline = Pipeline([
        ('vec', CountVectorizer()),
        ('cls', MultinomialNB())
    ])

    param_grid = [
        {
            'vec': [CountVectorizer(tokenizer=tokenizer),
                    TfidfVectorizer(tokenizer=tokenizer)],
            'vec__max_df': [1.0, 0.95, 0.90, 0.85],
            'vec__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3)],
            'vec__max_features': [None, 100, 1000, 10000],
            'vec__preprocessor': preprocessors
        }
    ]

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, Y_train)

    print("Best parameters found:")
    print(grid_search.best_params_)

    Y_pred = grid_search.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred))
