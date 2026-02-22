from colorist import red
import pandas as pd
import src.shared.env as env
from src.shared.cleanse_data import delete_duplicate_rows, delete_unique_columns


prefix = "    "

def cleanse_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleanses the DataFrame by:
        1. Deleting duplicate rows
        2. Deleting columns with a single unique value

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # ---------------------------
    # 1. Remove duplicate rows
    # ---------------------------
    print(f"{prefix} Removing duplicate rows...")
    df = delete_duplicate_rows(df)

    # ---------------------------
    # 2. Remove columns with single unique value
    # ---------------------------
    print(f"{prefix} Removing columns with a single unique value...")
    df = delete_unique_columns(df)

    # ---------------------------
    # 3. Remove rows with invalid categorical values
    # ---------------------------
    for col, invalid_values in env.INVALID_CATEGORIES.items():
        if col in df.columns:
            count_invalid = df[col].isin(invalid_values).sum()
            if count_invalid > 0:
                print(f"{prefix} Dropping {count_invalid} rows in '{col}' with invalid values: {invalid_values}")
                df = df[~df[col].isin(invalid_values)]

    return df