import pytest
import pandas as pd
from src.models.boston_house_price_prediction.cleanse_data import *


@pytest.fixture
def df_no_unique():
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [6, 7, 8, 9, 10],
        'C': [5, 4, 3, 2, 1],
        'D': ['a', 'b', 'c', 'd', 'e']
    }
    return pd.DataFrame(data)


@pytest.fixture
def df_with_unique():
    data = {
        'A': [1, 1, 1, 1, 1],  # All values are the same
        'B': [2, 3, 4, 5, 6],  # All values are unique
        'C': [7, 7, 8, 8, 8],  # Some values are unique, but column is not unique
    }
    return pd.DataFrame(data)


@pytest.fixture
def df_all_unique():
    data = {
        'A': [1, 2, 3, 4, 5],  # All values are unique
        'B': [6, 7, 8, 9, 10],  # All values are unique
        'C': [11, 12, 13, 14, 15]  # All values are unique
    }
    return pd.DataFrame(data)


@pytest.fixture
def df_mixed():
    data = {
        'A': [1, 1, 1, 1, 1],  # Unique values
        'B': [1, 2, 3, 4, 5],  # All values are unique
        'C': [1, 2, 2, 2, 2],  # Some values are unique, but column is not unique
        'D': ['a', 'a', 'a', 'a', 'a']   # Unique values, but categorical
    }
    return pd.DataFrame(data)


def test_delete_unique_columns_no_unique(df_no_unique):
    df = df_no_unique.copy()
    original_shape = df.shape
    result_df = delete_unique_columns(df)
    assert result_df.shape == original_shape, "The shape should remain the same as there are no unique columns to delete"


def test_delete_unique_columns_with_unique(df_with_unique):
    df = df_with_unique.copy()
    result_df = delete_unique_columns(df)
    assert 'A' not in result_df.columns, "Column 'A' should be deleted"
    assert 'B' in result_df.columns, "Column 'B' should not be deleted"
    assert 'C' in result_df.columns, "Column 'C' should not be deleted"


def test_delete_unique_columns_all_unique(df_all_unique):
    df = df_all_unique.copy()
    result_df = delete_unique_columns(df)
    assert result_df.shape[1] == 3, "No columns should be deleted as none of them have unique values"


def test_delete_unique_columns_mixed(df_mixed):
    df = df_mixed.copy()
    result_df = delete_unique_columns(df)
    assert 'A' not in result_df.columns, "Column 'A' should be deleted"
    assert 'B' in result_df.columns, "Column 'B' should not be deleted"
    assert 'C' in result_df.columns, "Column 'C' should not be deleted"
    assert 'D' not in result_df.columns, "Column 'D' should be deleted"


############################################# delete_duplicate_rows ####################################################


# Sample DataFrame fixture with duplicate rows
@pytest.fixture
def df_with_duplicates():
    data = {
        'A': [1, 2, 3, 3, 4, 5],
        'B': [1, 2, 3, 3, 4, 5],
        'C': ['a', 'b', 'c', 'c', 'd', 'e']
    }
    return pd.DataFrame(data)


# Sample DataFrame fixture without duplicate rows
@pytest.fixture
def df_no_duplicates():
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [1, 2, 3, 4, 5],
        'C': ['a', 'b', 'c', 'd', 'e']
    }
    return pd.DataFrame(data)


# Test case to check if duplicate rows are correctly deleted
def test_delete_duplicate_rows_with_duplicates(df_with_duplicates):
    df = df_with_duplicates.copy()
    result_df = delete_duplicate_rows(df)
    assert result_df.shape[0] == 5, "Expected number of rows after deletion is 5"
    assert not result_df.duplicated().any(), "There should be no duplicate rows after deletion"


# Test case to check if the function returns the same DataFrame when there are no duplicate rows
def test_delete_duplicate_rows_no_duplicates(df_no_duplicates):
    df = df_no_duplicates.copy()
    result_df = delete_duplicate_rows(df)
    assert result_df.shape == df.shape, "Shape should remain the same when there are no duplicate rows"
    assert result_df.equals(df), "DataFrames should be equal when there are no duplicate rows"


############################################# delete_missing_values ####################################################


@pytest.fixture
def df_with_issues():
    data = {
        'A': [0, 0, 0, 0, 1],
        'B': [1, 2, 3, 4, 5],
        'C': [1, 0, 0, 0, 0],
        'D': [1, 2, None, 4, 5],
        'E': [None, None, None, None, 5]
    }
    return pd.DataFrame(data)


# Fixture for DataFrame without columns to be dropped
@pytest.fixture
def df_no_issues():
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [1, 2, 3, 4, 5],
        'C': [5, 4, 3, 2, 1],
        'D': [5, 4, 3, 2, 1]
    }
    return pd.DataFrame(data)


# Test case for DataFrame with columns to be dropped due to high percentage of zeros or NaNs
def test_delete_missing_values_with_issues(df_with_issues):
    df = df_with_issues.copy()
    result_df = delete_missing_values(df, threshold_in_percentage=70)
    expected_columns = ['B', 'D']
    assert list(result_df.columns) == expected_columns, f"Expected columns after deletion: {expected_columns}"
    assert result_df.shape[1] == len(expected_columns), "Number of columns after deletion is incorrect"


# Test case for DataFrame without columns to be dropped
def test_delete_missing_values_no_issues(df_no_issues):
    df = df_no_issues.copy()
    original_shape = df.shape
    result_df = delete_missing_values(df, threshold_in_percentage=70)
    assert result_df.shape == original_shape, "Shape should remain the same when there are no columns to be dropped"
    assert result_df.equals(df), "DataFrames should be equal when there are no columns to be dropped"


# Test case for edge case: DataFrame with all columns to be dropped
@pytest.fixture
def df_all_issues():
    data = {
        'A': [0, 0, 0, 0, 0],
        'B': [None, None, None, None, None],
        'C': [0, 0, 0, 0, 0],
        'D': [None, None, None, None, None]
    }
    return pd.DataFrame(data)


def test_delete_missing_values_all_issues(df_all_issues):
    df = df_all_issues.copy()
    result_df = delete_missing_values(df, threshold_in_percentage=70)
    assert result_df.empty, "Result DataFrame should be empty after dropping all columns"


# Test case for edge case: DataFrame with threshold exactly at 70%
@pytest.fixture
def df_threshold_edge():
    data = {
        'A': [0, 0, 0, 0, 1],   # 80% zeros
        'B': [1, 2, 3, 4, 5],   # 0% zeros
        'C': [1, 2, None, None, None]  # 60% NaNs
    }
    return pd.DataFrame(data)


def test_delete_missing_values_threshold_edge(df_threshold_edge):
    df = df_threshold_edge.copy()
    result_df = delete_missing_values(df, threshold_in_percentage=70)
    expected_columns = ['B', 'C']
    assert list(result_df.columns) == expected_columns, f"Expected columns after deletion: {expected_columns}"