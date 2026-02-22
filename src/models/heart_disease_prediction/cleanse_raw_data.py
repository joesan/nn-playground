from colorist import red
import pandas as pd

from src.shared.cleanse_data import delete_duplicate_rows, delete_unique_columns


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
    red("************+ START: cleanse_raw_data START ************+ ")

    # 1. Delete duplicate rows
    print("[INFO] Removing duplicate rows...")
    df = delete_duplicate_rows(df)

    # 2. Delete columns with single unique value
    print("[INFO] Removing columns with a single unique value...")
    df = delete_unique_columns(df)

    red("************+ END: cleanse_raw_data END ************+ ")
    return df