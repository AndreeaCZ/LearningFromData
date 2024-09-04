from itertools import product

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
import nltk


def test_grid(X_train, Y_train, X_test, Y_test, pipeline, param_grid):
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)

    grid_search.fit(X_train, Y_train)

    print("Best parameters found:")
    print(grid_search.best_params_)

    Y_pred = grid_search.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred))


def test_vec_parameters(X_train, Y_train, X_test, Y_test, preprocessors, tokenizer):
    """
    Uses GridSearchCV to try all combinations of input parameters for the Count and Tfidf vectorizers.

    :param X_train: Training set
    :param Y_train: Training labels
    :param X_test: Testing set
    :param Y_test: Testing labels
    :param preprocessors: List of preprocessing functions
    :param tokenizer: Tokenizing function
    :return:
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

    test_grid(X_train, Y_train, X_test, Y_test, pipeline, param_grid)


def test_combining_vecs(X_train, Y_train, X_test, Y_test, preprocessor, tokenizer):
    """
    Uses FeatureUnion Tto combine the Count and Tfidf vectorizers. Then uses GridSearchCV to try some combinations of
    input parameters for the Count and Tfidf vectorizers.

    :param X_train: Training set
    :param Y_train: Training labels
    :param X_test: Testing set
    :param Y_test: Testing labels
    :param preprocessor: Preprocessing function
    :param tokenizer: Tokenizing function
    :return:
    """
    # FeatureUnion to combine CountVectorizer and TfidfVectorizer
    combined_features = FeatureUnion(
        transformer_list=[
            ('count', CountVectorizer(preprocessor=preprocessor, tokenizer=tokenizer)),
            ('tfidf', TfidfVectorizer(preprocessor=preprocessor, tokenizer=tokenizer))
        ]
    )

    pipeline = Pipeline([
        ('features', combined_features),
        ('cls', MultinomialNB())
    ])

    param_grid = {
        'features__count__max_df': [1.0, 0.95, 0.90],
        'features__count__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'features__tfidf__max_df': [1.0, 0.95, 0.90],
        'features__tfidf__ngram_range': [(1, 1), (1, 2), (2, 2)],
    }

    test_grid(X_train, Y_train, X_test, Y_test, pipeline, param_grid)


def lemmatize_text(tokens):
    """
    Lemmatizes the input tokens using WordNetLemmatizer.

    :param tokens: Input tokens string
    :return: Lemmatized tokens as a single string
    """
    return [lemmatizer.lemmatize(token) for token in tokens]


def porter_stem_text(tokens):
    """
    Stems the input tokens using PorterStemmer.

    :param tokens: Input tokens string
    :return: Stemmed tokens as a single string
    """
    return [porter_stemmer.stem(token) for token in tokens]


def snowball_stem_text(tokens):
    """
    Stems the input tokens using SnowballStemmer.

    :param tokens: Input tokens string
    :return: Stemmed tokens as a single string
    """
    return [snowball_stemmer.stem(token) for token in tokens]


def pos_tag_text(tokens):
    """
    Tags the input tokens with POS tags.

    :param tokens: Input tokens list
    :return: Tagged tokens as a list of strings with tags
    """
    pos_tags = nltk.pos_tag(tokens)
    return [f"{word}_{tag}" for word, tag in pos_tags]


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

lemmatizer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()
snowball_stemmer = SnowballStemmer("english")


def test_preprocessing(X_train, Y_train, X_test, Y_test, tokenizer):
    vectorizer = CountVectorizer(max_df=0.9, ngram_range=(1, 1), max_features=10000, tokenizer=tokenizer)

    pipeline = Pipeline([
        ('vec', vectorizer),
        ('cls', MultinomialNB())
    ])

    param_grid = {
        'vec__preprocessor': [lemmatize_text, porter_stem_text, snowball_stem_text, pos_tag_text],
    }

    test_grid(X_train, Y_train, X_test, Y_test, pipeline, param_grid)

