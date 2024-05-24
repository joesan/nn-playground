import pytest
from src.models.boston_house_price_prediction.feature_engineering import *
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression


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
    result = identify_categorical_features(df, min_unique_as_categorical=1, max_unique_as_categorical=2)
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


# Fixture for a DataFrame with various data types
@pytest.fixture
def df_mixed():
    data = {
        'A': ['cat', 'dog', 'mouse', 'dog', 'cat'],  # Object type
        'B': [1, 2, 1, 2, 1],  # Int type with exactly 2 unique values
        'C': [1.0, 1.0, 2.0, 3.0, 4.0],  # Float type with 4 unique values
        'D': [1, 1, 1, 1, 1],  # Int type with 1 unique value
        'E': pd.Series(['a', 'b', 'a', 'b', 'a'], dtype="category"),  # Category type
        'F': [5, 6, 7, 8, 9]  # Int type with 5 unique values
    }
    return pd.DataFrame(data)


# Test case for identifying categorical features with default settings
def test_identify_categorical_features_default(df_mixed):
    result = identify_categorical_features(df_mixed)
    expected = ['A', 'B', 'E']
    assert sorted(result) == sorted(expected), f"Expected {expected}, but got {result}"


# Test case for identifying categorical features with custom settings
def test_identify_categorical_features_custom(df_mixed):
    result = identify_categorical_features(df_mixed, min_unique_as_categorical=1, max_unique_as_categorical=3)
    expected = ['A', 'B', 'D', 'E']
    assert sorted(result) == sorted(expected), f"Expected {expected}, but got {result}"


############################################# split_features_target ####################################################


# Define the fixtures as a list of tuples
@pytest.fixture
def dataframes():
    df_multiple_columns = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1],
        'C': [2, 3, 4, 5, 6],
        'D': [1, 3, 5, 7, 9]
    })

    df_single_feature = pd.DataFrame({
        'Feature': [1, 2, 3, 4, 5],
        'Target': [5, 4, 3, 2, 1]
    })

    df_single_column = pd.DataFrame({
        'Target': [5, 4, 3, 2, 1]
    })

    df_empty = pd.DataFrame()

    return [
        (df_multiple_columns, df_multiple_columns.iloc[:, :-1], df_multiple_columns.iloc[:, -1]),
        (df_single_feature, df_single_feature.iloc[:, :-1], df_single_feature.iloc[:, -1]),
        (df_single_column, df_single_column.iloc[:, :-1], df_single_column.iloc[:, -1]),
        (df_empty, df_empty.iloc[:, :-1], pd.Series(dtype='float64'))
    ]


# Combined test case iterating through the dataframes
def test_split_features_target(dataframes):
    for df, expected_X, expected_y in dataframes:
        if df.empty:
            with pytest.raises(IndexError):
                X, y = split_features_target(df)
        else:
            X, y = split_features_target(df)
            pd.testing.assert_frame_equal(X, expected_X, check_dtype=True)
            pd.testing.assert_series_equal(y, expected_y, check_dtype=True)


#################################################### impute ############################################################

def test_impute_mean():
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [np.nan, 2, 3, 4]
    })
    expected_df = pd.DataFrame({
        'A': [1.0, 2.0, 2.3333333333333335, 4.0],
        'B': [3.0, 2.0, 3.0, 4.0]
    })
    result_df = impute(df, ImputeStrategy.MEAN)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_impute_median():
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [np.nan, 2, 3, 4]
    })
    expected_df = pd.DataFrame({
        'A': [1.0, 2.0, 2.0, 4.0],
        'B': [3.0, 2.0, 3.0, 4.0]
    })
    result_df = impute(df, ImputeStrategy.MEDIAN)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_impute_most_frequent():
    df = pd.DataFrame({
        'A': [1, 2, 2, np.nan],
        'B': [np.nan, 2, 2, 2]
    })
    expected_df = pd.DataFrame({
        'A': [1.0, 2.0, 2.0, 2.0],
        'B': [2.0, 2.0, 2.0, 2.0]
    })
    result_df = impute(df, ImputeStrategy.MOST_FREQUENT)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_impute_constant():
    df = pd.DataFrame({
        'A': [1, np.nan, 3],
        'B': [np.nan, 2, 3]
    })
    expected_df = pd.DataFrame({
        'A': [1.0, 0.0, 3.0],
        'B': [0.0, 2.0, 3.0]
    })
    result_df = impute(df, ImputeStrategy.CONSTANT)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_impute_no_missing_values():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    expected_df = pd.DataFrame({
        'A': [1.0, 2.0, 3.0],
        'B': [4.0, 5.0, 6.0]
    })
    result_df = impute(df, ImputeStrategy.MEAN)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_impute_empty_dataframe():
    df = pd.DataFrame(columns=['A', 'B'])
    expected_df = pd.DataFrame(columns=['A', 'B'])
    result_df = impute(df, ImputeStrategy.MEAN)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_impute_all_missing_values():
    df = pd.DataFrame({
        'A': [np.nan, np.nan],
        'B': [np.nan, np.nan]
    })
    expected_df = pd.DataFrame({
        'A': [0.0, 0.0],
        'B': [0.0, 0.0]
    })
    result_df = impute(df, ImputeStrategy.CONSTANT)
    pd.testing.assert_frame_equal(result_df, expected_df)

##


def test_empty_data():
    features = pd.DataFrame()
    y = pd.Series()
    with pytest.raises(ValueError):
        evaluate_imputation_strategies(features, y, num_splits=0)


# Test case 2: Test function with one feature and one target
def test_single_data_point():
    features = pd.DataFrame([[1]])
    y = pd.Series([1])
    with pytest.raises(ValueError):
        evaluate_imputation_strategies(features, y, num_splits=0)


# Test case 3: Test function with all missing values in features
def test_all_missing_values():
    features = pd.DataFrame([[np.nan, np.nan], [np.nan, np.nan]])
    y = pd.Series([1, 2])
    with pytest.raises(ValueError):
        evaluate_imputation_strategies(features, y)


# Test case 4: Test function with all missing target values
def test_all_missing_target():
    features, _ = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
    y = np.full((100,), np.nan)
    features = pd.DataFrame(features)
    y = pd.Series(y)
    with pytest.raises(ValueError):
        evaluate_imputation_strategies(features, y)


# Test case: 5 Test function with large dataset
def test_large_dataset():
    features, y = make_regression(n_samples=10000, n_features=10, noise=0.1, random_state=42)
    features = pd.DataFrame(features)
    y = pd.Series(y)
    results, _, best_strategy = evaluate_imputation_strategies(features, y)
    assert len(results) == len(ImputeStrategy)
    assert best_strategy in results


# Test case 6: Test function with zero variance in features
def test_zero_variance_features():
    features = np.zeros((100, 10))
    y = np.random.rand(100)
    features = pd.DataFrame(features)
    y = pd.Series(y)
    results, _, best_strategy = evaluate_imputation_strategies(features, y)
    assert len(results) == len(ImputeStrategy)
    assert best_strategy in results

############################################## check_for_normality #####################################################


# Test case: Test function with a DataFrame containing normally distributed data
def test_normal_distribution():
    data = {'A': np.random.normal(0, 1, 100),
            'B': np.random.normal(0, 1, 100)}
    df = pd.DataFrame(data)
    check_for_normality(df)  # No assertion needed, just checking for any errors.


# Test case: Test function with a DataFrame containing non-normally distributed data
def test_non_normal_distribution():
    data = {'A': np.random.uniform(0, 1, 100),
            'B': np.random.uniform(0, 1, 100)}
    df = pd.DataFrame(data)
    with pytest.raises(ValueError):
        check_for_normality(df)  # Expecting a ValueError because data is not normally distributed


# Test case: Test function with an empty DataFrame
def test_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        check_for_normality(df)  # Expecting a ValueError because the DataFrame is empty


# Test case: Test function with a DataFrame containing NaN values
def test_dataframe_with_nan():
    data = {'A': np.random.normal(0, 1, 100),
            'B': np.random.normal(0, 1, 100)}
    df = pd.DataFrame(data)
    df.loc[0, 'A'] = np.nan  # introducing NaN value
    with pytest.raises(ValueError):
        check_for_normality(df)  # Expecting a ValueError because the DataFrame contains NaN values
        