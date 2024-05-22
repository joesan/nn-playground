import pytest
from src.models.boston_house_price_prediction.feature_engineering import *


@pytest.fixture
def sample_dataframe():
    data = {
        'A': [0, 1, 0, 1, 0],         # integer but categorical
        'B': [0, 0, 0, 0, 0],         # integers, all zeros
        'C': ['cat', 'dog', 'cat', 'cat', 'dog'],  # categorical
        'D': [1, 2, 3, 4, 5],         # integers
        'E': [0, 1, 2, 0, 4]          # integers
    }
    df = pd.DataFrame(data)
    df['C'] = df['C'].astype('category')
    return df

def test_identify_categorical_columns(sample_dataframe):
    df = sample_dataframe
    expected_categorical = ['A', 'C']
    result = identify_categorical_features(df)
    assert sorted(result) == sorted(expected_categorical), f"Expected {expected_categorical} but got {result}"

def test_identify_categorical_columns_with_different_threshold(sample_dataframe):
    df = sample_dataframe
    expected_categorical = ['A', 'B', 'C']
    result = identify_categorical_features(df, max_unique_as_categorical=1)
    assert sorted(result) == sorted(expected_categorical), f"Expected {expected_categorical} but got {result}"

def test_no_categorical_columns():
    data = {
        'A': [1, 2, 3, 4],
        'B': [5, 6, 7, 8]
    }
    df = pd.DataFrame(data)
    expected_categorical = []
    result = identify_categorical_features(df)
    assert result == expected_categorical, f"Expected {expected_categorical} but got {result}"

def test_all_categorical_columns():
    data = {
        'A': ['a', 'b', 'c', 'd'],
        'B': ['w', 'x', 'y', 'z']
    }
    df = pd.DataFrame(data)
    df = df.astype('category')
    expected_categorical = ['A', 'B']
    result = identify_categorical_features(df)
    assert sorted(result) == sorted(expected_categorical), f"Expected {expected_categorical} but got {result}"

def test_mixed_data_types():
    data = {
        'A': [1, 2, 1, 2],
        'B': ['a', 'b', 'a', 'b'],
        'C': [1.1, 2.2, 3.3, 4.4]
    }
    df = pd.DataFrame(data)
    df['B'] = df['B'].astype('category')
    expected_categorical = ['A', 'B']
    result = identify_categorical_features(df)
    assert sorted(result) == sorted(expected_categorical), f"Expected {expected_categorical} but got {result}"
