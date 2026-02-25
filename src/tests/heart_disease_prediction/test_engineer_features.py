import sys
from unittest.mock import MagicMock, patch, ANY
import numpy as np

# Mock missing modules before any project imports
sys.modules["colorist"] = MagicMock()
keras_mock = MagicMock()
sys.modules["keras"] = keras_mock
sys.modules["keras.metrics"] = keras_mock.metrics
sys.modules["keras.layers"] = keras_mock.layers
sys.modules["keras.models"] = keras_mock.models
sys.modules["keras.callbacks"] = keras_mock.callbacks

import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

# Import module once — patches target its namespace directly
MODULE = "src.models.heart_disease_prediction.engineer_features"
ENV    = f"{MODULE}.env"

from src.models.heart_disease_prediction.engineer_features import (
    split_target,
    encode_categoricals,
    scale_numeric_features,
    check_rows_and_index_sanity,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def basic_df():
    return pd.DataFrame({
        "age":    [25, 35, 45],
        "income": [50000, 60000, 70000],
        "target": [0, 1, 0],
    })


@pytest.fixture
def df_with_categoricals():
    return pd.DataFrame({
        "age":    [25, 35, 45],
        "gender": ["M", "F", "M"],
        "chest_pain": ["typical", "atypical", "typical"],
    })


# ===========================================================================
# split_target
# ===========================================================================

class TestSplitTarget:
    def test_returns_X_and_y(self, basic_df):
        X, y = split_target(basic_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_target_column_removed_from_X(self, basic_df):
        X, y = split_target(basic_df)
        assert "target" not in X.columns

    def test_y_contains_correct_values(self, basic_df):
        X, y = split_target(basic_df)
        assert list(y) == [0, 1, 0]

    def test_X_retains_other_columns(self, basic_df):
        X, y = split_target(basic_df)
        assert set(X.columns) == {"age", "income"}

    def test_custom_target_column(self):
        df = pd.DataFrame({"a": [1, 2], "label": [0, 1]})
        X, y = split_target(df, target_column="label")
        assert y.name == "label"
        assert "label" not in X.columns

    def test_raises_if_target_column_missing(self, basic_df):
        with pytest.raises(ValueError, match="Target column 'missing' not found"):
            split_target(basic_df, target_column="missing")

    def test_X_row_count_matches_original(self, basic_df):
        X, y = split_target(basic_df)
        assert len(X) == len(basic_df)

    def test_index_preserved(self, basic_df):
        X, y = split_target(basic_df)
        assert list(X.index) == list(basic_df.index)
        assert list(y.index) == list(basic_df.index)


# ===========================================================================
# encode_categoricals
# ===========================================================================

class TestEncodeCategoricals:
    def test_categorical_columns_are_encoded(self, df_with_categoricals):
        result = encode_categoricals(df_with_categoricals.copy())
        assert "gender" not in result.columns
        assert "chest_pain" not in result.columns

    def test_numeric_columns_are_preserved(self, df_with_categoricals):
        result = encode_categoricals(df_with_categoricals.copy())
        assert "age" in result.columns

    def test_encoded_columns_are_boolean_or_uint(self, df_with_categoricals):
        result = encode_categoricals(df_with_categoricals.copy())
        new_cols = [c for c in result.columns if c != "age"]
        for col in new_cols:
            assert result[col].dtype in [bool, "bool", "uint8", "int64", "int32"]

    def test_no_encoding_when_no_categoricals(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = encode_categoricals(df.copy())
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 3

    def test_returns_dataframe(self, df_with_categoricals):
        result = encode_categoricals(df_with_categoricals.copy())
        assert isinstance(result, pd.DataFrame)

    def test_drop_first_false_keeps_all_dummies(self):
        # With drop_first=False, M and F both get a column
        df = pd.DataFrame({"gender": ["M", "F", "M"]})
        result = encode_categoricals(df.copy())
        assert "gender_M" in result.columns
        assert "gender_F" in result.columns

    def test_row_count_unchanged(self, df_with_categoricals):
        result = encode_categoricals(df_with_categoricals.copy())
        assert len(result) == len(df_with_categoricals)


# ===========================================================================
# scale_numeric_features
# ===========================================================================

class TestScaleNumericFeatures:
    def _make_X_y(self):
        X = pd.DataFrame({"age": [25.0, 35.0, 45.0], "income": [50000.0, 60000.0, 70000.0]})
        y = pd.Series([0, 1, 0])
        return X, y

    def test_returns_X_y_and_scaler(self):
        X, y = self._make_X_y()
        with patch(ENV) as mock_env:
            mock_env.NUMERIC_COLUMNS = ["age", "income"]
            X_out, y_out, scaler = scale_numeric_features(X, y)
        assert isinstance(X_out, pd.DataFrame)
        assert isinstance(scaler, StandardScaler)

    def test_creates_default_scaler_when_none_provided(self):
        X, y = self._make_X_y()
        with patch(ENV) as mock_env:
            mock_env.NUMERIC_COLUMNS = ["age", "income"]
            _, _, scaler = scale_numeric_features(X, y, scaler=None)
        assert scaler is not None
        assert isinstance(scaler, StandardScaler)

    def test_uses_provided_scaler(self):
        X, y = self._make_X_y()
        fitted_scaler = StandardScaler()
        fitted_scaler.fit(X)
        with patch(ENV) as mock_env:
            mock_env.NUMERIC_COLUMNS = ["age", "income"]
            _, _, returned_scaler = scale_numeric_features(X.copy(), y, scaler=fitted_scaler)
        assert returned_scaler is fitted_scaler

    def test_fit_transform_called_when_no_scaler(self):
        X, y = self._make_X_y()
        mock_scaler = MagicMock(spec=StandardScaler)
        mock_scaler.fit_transform.return_value = np.zeros((3, 2))
        with patch(ENV) as mock_env, \
             patch(f"{MODULE}.StandardScaler", return_value=mock_scaler):
            mock_env.NUMERIC_COLUMNS = ["age", "income"]
            scale_numeric_features(X, y, scaler=None)
        mock_scaler.fit_transform.assert_called_once()

    def test_transform_called_when_scaler_provided(self):
        X, y = self._make_X_y()
        mock_scaler = MagicMock(spec=StandardScaler)
        mock_scaler.transform.return_value = np.zeros((3, 2))
        with patch(ENV) as mock_env:
            mock_env.NUMERIC_COLUMNS = ["age", "income"]
            scale_numeric_features(X, y, scaler=mock_scaler)
        mock_scaler.transform.assert_called_once()

    def test_skips_scaling_when_no_numeric_columns_match(self):
        X, y = self._make_X_y()
        with patch(ENV) as mock_env:
            mock_env.NUMERIC_COLUMNS = ["nonexistent_col"]
            X_out, y_out, scaler = scale_numeric_features(X.copy(), y)
        # Values should be unchanged
        assert scaler is None
        assert X_out["age"].tolist() == [25.0, 35.0, 45.0]

    def test_only_env_numeric_columns_are_scaled(self):
        X = pd.DataFrame({"age": [25.0, 35.0, 45.0], "income": [50000.0, 60000.0, 70000.0]})
        y = pd.Series([0, 1, 0])
        with patch(ENV) as mock_env:
            mock_env.NUMERIC_COLUMNS = ["age"]  # only scale age, not income
            X_out, _, _ = scale_numeric_features(X, y)
        # income should be untouched
        assert X_out["income"].tolist() == [50000.0, 60000.0, 70000.0]

    def test_y_is_unchanged(self):
        X, y = self._make_X_y()
        original_y = y.copy()
        with patch(ENV) as mock_env:
            mock_env.NUMERIC_COLUMNS = ["age", "income"]
            _, y_out, _ = scale_numeric_features(X, y)
        assert list(y_out) == list(original_y)


# ===========================================================================
# check_rows_and_index_sanity
# ===========================================================================

class TestCheckRowsAndIndexSanity:
    def test_passes_with_matching_X_and_y(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        check_rows_and_index_sanity(X, y)  # should not raise

    def test_raises_on_row_count_mismatch(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1])
        with pytest.raises(ValueError, match="Row count mismatch"):
            check_rows_and_index_sanity(X, y)

    def test_raises_on_index_mismatch(self):
        X = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
        y = pd.Series([0, 1, 0], index=[10, 11, 12])
        with pytest.raises(ValueError, match="Index mismatch"):
            check_rows_and_index_sanity(X, y)

    def test_passes_with_non_default_index(self):
        X = pd.DataFrame({"a": [1, 2]}, index=[5, 6])
        y = pd.Series([0, 1], index=[5, 6])
        check_rows_and_index_sanity(X, y)  # should not raise