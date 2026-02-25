import sys
import importlib
from unittest.mock import MagicMock, patch

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

# Import the module once — patches will target its namespace directly
import src.models.heart_disease_prediction.cleanse_raw_data as _module
from src.models.heart_disease_prediction.cleanse_raw_data import cleanse_raw_data

# ---------------------------------------------------------------------------
# Patch paths — must match where names live AFTER import in the module under test:
#   `from src.shared.cleanse_data import delete_duplicate_rows` → patch in MODULE
#   `import src.shared.env as env`                             → patch in MODULE
# ---------------------------------------------------------------------------
MODULE = "src.models.heart_disease_prediction.cleanse_raw_data"
DEDUP  = f"{MODULE}.delete_duplicate_rows"
UNIQUE = f"{MODULE}.delete_unique_columns"
ENV    = f"{MODULE}.env"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df():
    return pd.DataFrame({
        "id":       [1, 2, 3, 4],
        "category": ["a", "b", "c", "d"],
        "value":    [10, 20, 30, 40],
    })


# ---------------------------------------------------------------------------
# 1. Duplicate rows
# ---------------------------------------------------------------------------

class TestDeleteDuplicateRows:
    def test_calls_delete_duplicate_rows(self, simple_df):
        with patch(DEDUP, return_value=simple_df.copy()) as mock_dedup, \
             patch(UNIQUE, side_effect=lambda df: df), \
             patch(ENV) as mock_env:
            mock_env.INVALID_CATEGORIES = {}
            cleanse_raw_data(simple_df)
            mock_dedup.assert_called_once()

    def test_duplicate_rows_result_is_propagated(self):
        input_df = pd.DataFrame({"a": [1, 1, 2]})
        deduped  = pd.DataFrame({"a": [1, 2]})
        with patch(DEDUP, return_value=deduped), \
             patch(UNIQUE, side_effect=lambda df: df), \
             patch(ENV) as mock_env:
            mock_env.INVALID_CATEGORIES = {}
            result = cleanse_raw_data(input_df)
            assert len(result) == 2


# ---------------------------------------------------------------------------
# 2. Unique-value columns
# ---------------------------------------------------------------------------

class TestDeleteUniqueColumns:
    def test_calls_delete_unique_columns(self, simple_df):
        with patch(DEDUP, side_effect=lambda df: df), \
             patch(UNIQUE, return_value=simple_df.copy()) as mock_unique, \
             patch(ENV) as mock_env:
            mock_env.INVALID_CATEGORIES = {}
            cleanse_raw_data(simple_df)
            mock_unique.assert_called_once()

    def test_unique_column_result_is_propagated(self):
        input_df = pd.DataFrame({"a": [1, 2], "b": [99, 99]})
        reduced  = pd.DataFrame({"a": [1, 2]})
        with patch(DEDUP, side_effect=lambda df: df), \
             patch(UNIQUE, return_value=reduced), \
             patch(ENV) as mock_env:
            mock_env.INVALID_CATEGORIES = {}
            result = cleanse_raw_data(input_df)
            assert list(result.columns) == ["a"]


# ---------------------------------------------------------------------------
# 3. Invalid categorical values
# ---------------------------------------------------------------------------

class TestInvalidCategories:
    def _run(self, df, invalid_categories):
        with patch(DEDUP, side_effect=lambda df: df), \
             patch(UNIQUE, side_effect=lambda df: df), \
             patch(ENV) as mock_env:
            mock_env.INVALID_CATEGORIES = invalid_categories
            return cleanse_raw_data(df)

    def test_drops_rows_with_invalid_category(self):
        df = pd.DataFrame({"status": ["active", "inactive", "unknown", "active"]})
        result = self._run(df, {"status": ["unknown"]})
        assert "unknown" not in result["status"].values
        assert len(result) == 3

    def test_drops_multiple_invalid_values_in_one_column(self):
        df = pd.DataFrame({"status": ["active", "inactive", "banned", "active"]})
        result = self._run(df, {"status": ["inactive", "banned"]})
        assert set(result["status"].values) == {"active"}

    def test_handles_multiple_columns(self):
        df = pd.DataFrame({
            "status":  ["active", "unknown", "unknown"],
            "country": ["US", "XX", "UK"],
        })
        result = self._run(df, {"status": ["unknown"], "country": ["XX"]})
        assert len(result) == 1
        assert result.iloc[0]["status"]  == "active"
        assert result.iloc[0]["country"] == "US"

    def test_skips_column_not_in_dataframe(self):
        df = pd.DataFrame({"other_col": ["a", "b", "c"]})
        result = self._run(df, {"status": ["unknown"]})
        assert len(result) == 3

    def test_no_rows_dropped_when_no_invalid_values_present(self):
        df = pd.DataFrame({"status": ["active", "active", "active"]})
        result = self._run(df, {"status": ["unknown"]})
        assert len(result) == 3

    def test_empty_invalid_categories_leaves_df_unchanged(self):
        df = pd.DataFrame({"status": ["active", "inactive"]})
        result = self._run(df, {})
        assert len(result) == 2

    def test_all_rows_invalid_returns_empty_df(self):
        df = pd.DataFrame({"status": ["unknown", "unknown"]})
        result = self._run(df, {"status": ["unknown"]})
        assert len(result) == 0
        assert list(result.columns) == ["status"]


# ---------------------------------------------------------------------------
# 4. Pipeline ordering & return value
# ---------------------------------------------------------------------------

class TestPipelineOrdering:
    def test_returns_dataframe(self, simple_df):
        with patch(DEDUP, side_effect=lambda df: df), \
             patch(UNIQUE, side_effect=lambda df: df), \
             patch(ENV) as mock_env:
            mock_env.INVALID_CATEGORIES = {}
            result = cleanse_raw_data(simple_df)
            assert isinstance(result, pd.DataFrame)

    def test_dedup_runs_before_unique_column_removal(self, simple_df):
        call_order = []

        def fake_dedup(df):
            call_order.append("dedup")
            return df

        def fake_unique(df):
            call_order.append("unique")
            return df

        with patch(DEDUP, side_effect=fake_dedup), \
             patch(UNIQUE, side_effect=fake_unique), \
             patch(ENV) as mock_env:
            mock_env.INVALID_CATEGORIES = {}
            cleanse_raw_data(simple_df)

        assert call_order == ["dedup", "unique"]