import pandas as pd
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df():
    return pd.DataFrame({
        "id":       [1, 2, 3, 4],
        "category": ["a", "b", "c", "d"],
        "value":    [10, 20, 30, 40],
    })


# ---------------------------------------------------------------------------
# We patch the three external dependencies so the unit tests are isolated:
#   - src.shared.cleanse_data.delete_duplicate_rows
#   - src.shared.cleanse_data.delete_unique_columns
#   - src.shared.env.INVALID_CATEGORIES
# ---------------------------------------------------------------------------

BASE = "src.shared.cleanse_data"
ENV  = "src.shared.env"


def _import_module():
    """Re-import the module under test after patches are in place."""
    import importlib
    import src.models.heart_disease_prediction.cleanse_raw_data as m   # adjust path if different
    importlib.reload(m)
    return m


# ---------------------------------------------------------------------------
# 1. duplicate rows are removed
# ---------------------------------------------------------------------------

class TestDeleteDuplicateRows:
    def test_calls_delete_duplicate_rows(self, simple_df):
        deduped = simple_df.copy()
        with patch(f"{BASE}.delete_duplicate_rows", return_value=deduped) as mock_dedup, \
             patch(f"{BASE}.delete_unique_columns", side_effect=lambda df: df), \
             patch(f"{ENV}.INVALID_CATEGORIES", {}):
            from src.models.heart_disease_prediction.cleanse_raw_data import cleanse_raw_data
            cleanse_raw_data(simple_df)
            mock_dedup.assert_called_once()

    def test_duplicate_rows_result_is_propagated(self):
        input_df  = pd.DataFrame({"a": [1, 1, 2]})
        deduped   = pd.DataFrame({"a": [1, 2]})
        with patch(f"{BASE}.delete_duplicate_rows", return_value=deduped), \
             patch(f"{BASE}.delete_unique_columns", side_effect=lambda df: df), \
             patch(f"{ENV}.INVALID_CATEGORIES", {}):
            from src.models.heart_disease_prediction.cleanse_raw_data import cleanse_raw_data
            result = cleanse_raw_data(input_df)
            assert len(result) == 2


# ---------------------------------------------------------------------------
# 2. unique-value columns are removed
# ---------------------------------------------------------------------------

class TestDeleteUniqueColumns:
    def test_calls_delete_unique_columns(self, simple_df):
        with patch(f"{BASE}.delete_duplicate_rows", side_effect=lambda df: df), \
             patch(f"{BASE}.delete_unique_columns", return_value=simple_df) as mock_unique, \
             patch(f"{ENV}.INVALID_CATEGORIES", {}):
            from src.models.heart_disease_prediction.cleanse_raw_data import cleanse_raw_data
            cleanse_raw_data(simple_df)
            mock_unique.assert_called_once()

    def test_unique_column_result_is_propagated(self):
        input_df  = pd.DataFrame({"a": [1, 2], "b": [99, 99]})
        reduced   = pd.DataFrame({"a": [1, 2]})
        with patch(f"{BASE}.delete_duplicate_rows", side_effect=lambda df: df), \
             patch(f"{BASE}.delete_unique_columns", return_value=reduced), \
             patch(f"{ENV}.INVALID_CATEGORIES", {}):
            from src.models.heart_disease_prediction.cleanse_raw_data import cleanse_raw_data
            result = cleanse_raw_data(input_df)
            assert list(result.columns) == ["a"]


# ---------------------------------------------------------------------------
# 3. invalid categorical values are dropped
# ---------------------------------------------------------------------------

class TestInvalidCategories:
    def _run(self, df, invalid_categories):
        with patch(f"{BASE}.delete_duplicate_rows", side_effect=lambda df: df), \
             patch(f"{BASE}.delete_unique_columns", side_effect=lambda df: df), \
             patch(f"{ENV}.INVALID_CATEGORIES", invalid_categories):
            from src.models.heart_disease_prediction.cleanse_raw_data import cleanse_raw_data
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
            "status":  ["active", "unknown", "active"],
            "country": ["US", "XX", "UK"],
        })
        result = self._run(df, {"status": ["unknown"], "country": ["XX"]})
        assert len(result) == 1
        assert result.iloc[0]["status"]  == "active"
        assert result.iloc[0]["country"] == "US"

    def test_skips_column_not_in_dataframe(self):
        df = pd.DataFrame({"other_col": ["a", "b", "c"]})
        # Should not raise even though "status" is not in df
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
# 4. pipeline ordering & return value
# ---------------------------------------------------------------------------

class TestPipelineOrdering:
    def test_returns_dataframe(self, simple_df):
        with patch(f"{BASE}.delete_duplicate_rows", side_effect=lambda df: df), \
             patch(f"{BASE}.delete_unique_columns", side_effect=lambda df: df), \
             patch(f"{ENV}.INVALID_CATEGORIES", {}):
            from src.models.heart_disease_prediction.cleanse_raw_data import cleanse_raw_data
            result = cleanse_raw_data(simple_df)
            assert isinstance(result, pd.DataFrame)

    def test_dedup_runs_before_unique_column_removal(self, simple_df):
        """Verify call order: dedup → unique-cols → invalid categories."""
        call_order = []

        def fake_dedup(df):
            call_order.append("dedup")
            return df

        def fake_unique(df):
            call_order.append("unique")
            return df

        with patch(f"{BASE}.delete_duplicate_rows", side_effect=fake_dedup), \
             patch(f"{BASE}.delete_unique_columns", side_effect=fake_unique), \
             patch(f"{ENV}.INVALID_CATEGORIES", {}):
            from src.models.heart_disease_prediction.cleanse_raw_data import cleanse_raw_data
            cleanse_raw_data(simple_df)

        assert call_order == ["dedup", "unique"]