import pytest
import pandas as pd
from src.models.boston_house_price_prediction.cleanse_data import delete_unique_columns


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

