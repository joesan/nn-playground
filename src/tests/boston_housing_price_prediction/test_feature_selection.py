import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.datasets import make_classification
import pytest
from src.models.boston_house_price_prediction.feature_selection import *


@pytest.fixture
def data():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    return X_train, y_train, X_test, y_test


def test_select_features_all(data):
    X_train, y_train, X_test, _ = data
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)

    assert X_train_fs.shape == X_train.shape, "The shape of X_train_fs should match the original X_train shape."
    assert X_test_fs.shape == X_test.shape, "The shape of X_test_fs should match the original X_test shape."
    assert fs.k == 'all', "The number of selected features should be 'all'."


def test_select_features_invalid_k(data):
    X_train, y_train, X_test, _ = data

    with pytest.raises(ValueError):
        # Test with an invalid k value
        fs = SelectKBest(score_func=chi2, k=100)
        fs.fit(X_train, y_train)


def test_select_features_chi2(data):
    X_train, y_train, X_test, _ = data
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)

    assert hasattr(fs, 'scores_'), "Feature selector should have 'scores_' attribute after fitting."
    assert len(fs.scores_) == X_train.shape[1], "Number of scores should match the number of features."


def test_select_features_transformed_shape(data):
    X_train, y_train, X_test, _ = data
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)

    assert X_train_fs.shape[1] == X_train.shape[1], "The number of features in X_train_fs should match the original number of features."
    assert X_test_fs.shape[1] == X_test.shape[1], "The number of features in X_test_fs should match the original number of features."


def test_select_features_type_error(data):
    X_train, y_train, X_test, _ = data

    with pytest.raises(TypeError):
        # Passing non-numeric data to test for TypeError
        X_train_invalid = [['a', 'b'], ['c', 'd']]
        y_train_invalid = [0, 1]
        fs = SelectKBest(score_func=chi2, k='all')
        fs.fit(X_train_invalid, y_train_invalid)
