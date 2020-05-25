def download_nltk(packages):
    """Download nlkt packages ignoring issues with certificates

    Args:
        packages (list): list of packages that have to be downloaded
    """

    import nltk
    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    for package in packages:
        nltk.download(package)

# Import modules
import sys
from sqlalchemy import create_engine
import pandas as pd
download_nltk(["punkt", "stopwords", "wordnet"])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def imbalanced_features(df, thresh = 200):
    """ Return a list of imbalanced features (features with less 1s or 0s than a certain threshold

    Args:
        df: dataframe with data
        thresh (opt): minimum number of 0s or 1s (Default = 200)

    Returns:
        imbalanced_cols = list of imbalanced columns
    """

    imbalanced_cols = []
    for column in df.columns.tolist()[4:]:
        col_counts = df[column].value_counts()
        if 1 not in col_counts or 0 not in col_counts or col_counts[0] < thresh or col_counts[1] < thresh:
            imbalanced_cols.append(column)
    return imbalanced_cols


def load_data(database_filepath):
    """ Load data from database and returns X and Y

    Args:
        database_filepath

    Returns:
        X: independent variables
        Y: dependent variables
        category_names: names of categories
    """
    engine = create_engine('sqlite:///'+database_filepath)
    with engine.connect() as conn:
        df = pd.read_sql_table(database_filepath, con = conn)

    cols_to_drop = imbalanced_features(df)
    df.drop(columns = cols_to_drop, inplace = True)
    print("\tFollowing columns will not be considered because too imbalanced:")
    print("\t{}".format(cols_to_drop))


    category_names = list(df.keys()[4:])
    X = np.array(df.message)
    Y = np.array(df.loc[:, category_names])

    return X, Y, category_names


def tokenize(text):
    """Tokenize text

    Args:
        text: text to tokenize

    Returns:
        lemmed: list of tokenized words
    """

    tokens = word_tokenize(text.lower())

    stopwords_eng = stopwords.words('english')
    tokens_wo_stopwords = [w for w in tokens if w not in stopwords_eng]

    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(w) for w in tokens_wo_stopwords]

    return lemmed


def build_models():
    """Return a list of dictionaries with information about the models that will be tested"""

    vect = CountVectorizer(tokenizer=tokenize)
    tfidf = TfidfTransformer()

    models = []

    # Random Forest Classifier
    clf = MultiOutputClassifier(RandomForestClassifier(random_state=314))
    models.append({"name": "Random Forest Classifier",\
                   "model": Pipeline([("vect", vect), ("tfidf", tfidf), ("clf", clf)]),\
                   "param_grid": {"clf__estimator__n_estimators": [10, 50, 100],\
                                  "clf__estimator__max_depth": [10, None]}})

    # Logistic Regression Classifier
    clf = MultiOutputClassifier(LogisticRegression(random_state=314))
    models.append({"name": "Logistic Regression",\
                   "model": Pipeline([("vect", vect), ("tfidf", tfidf), ("clf", clf)]),\
                   "param_grid": {"clf__estimator__penalty": ["l2", "elastic_net"],\
                                      "clf__estimator__C": [0.1, 1, 10]}})

    # Linear SVC
    clf = MultiOutputClassifier(LinearSVC(random_state=314))
    models.append({"name": "Linear SVC",\
                   "model": Pipeline([("vect", vect), ("tfidf", tfidf), ("clf", clf)]),\
                   "param_grid": {"clf__estimator__loss": ["hinge", "squared_hinge"],\
                                  "clf__estimator__C": [0.1, 1, 10]}})

    for i, model in enumerate(models):
        print("\tMODEL {}: {}".format(i, model["name"]))

    return models


def find_best_model(models, X, Y):
    """Define best model amongs input models using cross_validation and f1_weighted as scoring

    Args:
        models: list of dictionaries with information about the models to be tested
        X: independent variable
        Y: depentent variable

    Outputs:
        best_model: dictionary with information about the best model
    """

    for i, model in enumerate(models):
        scores = cross_val_score(model["model"], X, Y, cv = 3, scoring='f1_weighted')
        model["score"] = scores.mean()
        print("\tMODEL {}: f1_weighted cross validation scores: {}".format(i, scores))

    i_best_model = 0
    best_model = models[i_best_model]
    best_score = 0
    for i, model in enumerate(models):
        if model["score"] > best_score:
            best_score = model["score"]
            best_model = model
            i_best_model = i
    print("\tChosen model: MODEL {} - {}!".format(i_best_model, best_model["name"]))

    return best_model


def hypertuning(model, X, Y):
    """Hypertuning of model parameters using GridSearchCV and f1_weighted as scoring

    Args:
        model: dictionary with information about model and parameters grid
        X: inependent variable
        Y: dependent variable

    Outputs:
        cv: hypertuned model
    """
    # Hyperparameter tuning
    param_grid = model["param_grid"]
    print("\tFollowing parameters are being tested: {}".format(param_grid))

    # Instantiate the grid search model
    cv = GridSearchCV(model["model"], param_grid=param_grid, scoring='f1_weighted')

    # Fit grid search model
    cv.fit(X, Y)

    print("\tHypertuning completed!")

    return cv


def save_model(model, model_filepath):
    """Save the model to disk

    Args:
        model:  model to be saved
        model_filepath: filepath where model must be saved
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def evaluate_model(model, X, Y, category_names):
    """Compute predicted Y and print classification report

    Args:
        model:  sk-learn model that has been already fitted
        X: independent variable
        Y: dependent variable
    """

    Y_pred = model.predict(X)
    print(classification_report(Y, Y_pred, target_names=category_names))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n\tDATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 314)

        print('Building models...')
        models = build_models()

        print('Comparing models...')
        best_model = find_best_model(models, X_train, Y_train)

        print('Hypertuning best model...')
        cv = hypertuning(best_model, X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(cv, X_test, Y_test, category_names)

        print('Saving model...\n\tMODEL: {}'.format(model_filepath))
        save_model(cv, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
