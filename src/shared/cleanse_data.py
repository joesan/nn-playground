from colorist import red, Color
import pandas as pd

from src.shared import env


prefix = "    "

# delete columns with a single unique value
def delete_unique_columns(df):
    red(f"{prefix} ************+ START: delete_unique_columns ************+ ")
    print(f"{prefix} DataFrame shape before deleting columns with unique values: {Color.GREEN}{df.shape}{Color.OFF}")
    # get number of unique values for each column
    counts = df.nunique()
    print(f"{prefix} Are there any columns with unique values? {Color.GREEN}{(counts == 1).any()}{Color.OFF}")
    if counts.any() == 1:
        # record columns to delete by their names
        to_del = [col for col in counts.index if counts[col] == 1]
        # drop useless columns
        df.drop(to_del, axis=1, inplace=True)
        print(f"{prefix} DataFrame shape after deleting unique columns: {Color.GREEN}{df.shape}{Color.OFF}")
    red(f"{prefix} ************+ END: delete_unique_columns ************+ ")
    return df


# delete rows containing duplicate data from the dataset
def delete_duplicate_rows(df):
    """
    Deletes duplicate rows by keeping the first instance intact.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        DataFrame with deleted duplicate rows.
    """
    red(f"{prefix} ************+ START: delete_duplicate_rows ************+ ")
    print(f"{prefix} DataFrame shape before deleting duplicated rows: {Color.GREEN}{df.shape}{Color.OFF}")
    # delete duplicate rows
    dupes = df.duplicated()
    # report if there are any duplicates
    print(f"{prefix} Are there any duplicate rows? {Color.GREEN}{dupes.any()}{Color.OFF}")
    if dupes.any():
        df.drop_duplicates(inplace=True)
        print(f"{prefix} DataFrame shape after deleting duplicated rows: {Color.GREEN}{df.shape}{Color.OFF}")
        red(f"{prefix} ************+ END: delete_duplicate_rows ************+ ")
        return df
    else:
        red(f"{prefix} ************+ END: delete_duplicate_rows ************+ ")
        return df


def delete_missing_values(df, threshold_in_percentage=70):
    """
    Deletes columns from the DataFrame that have too many missing or zero values.

    Args:
        df (pd.DataFrame): Input DataFrame.
        threshold_in_percentage (int, optional): Default 70.
            The percentage threshold above which a column will be dropped.
            For example, if the percentage of zeros or NaNs in a column exceeds this threshold,
            the column will be removed from the DataFrame.

    Returns:
        pd.DataFrame: DataFrame with columns dropped that exceed the missing/zero value threshold.
    """
    red(f"{prefix} ************+ START: delete_missing_values ************+ ")
    print(f"{prefix} DataFrame shape before deleting rows with missing values: {Color.GREEN}{df.shape}{Color.OFF}")
    zeros_count = df.isin([0]).sum(axis=0)
    nans_count = df.isna().sum(axis=0)
    columns_to_drop = []
    for column in df.columns:
        zero_count = zeros_count[column] / len(df[column]) * 100
        nan_count = nans_count[column] / len(df[column]) * 100
        if zero_count > threshold_in_percentage or nan_count > threshold_in_percentage:
            print(f"\t{Color.GREEN}Column {column} will be dropped because of either: Zeros = {zeros_count[column]}, NaNs = {nans_count[column]}{Color.OFF}")
            columns_to_drop.append(column)
        else:
            print(f"{prefix} \tColumn {column}: Zeros = {zeros_count[column]}, NaNs = {nans_count[column]}")
    df.drop(columns=columns_to_drop, inplace=True)
    print(f"D{prefix} ataFrame shape after deleting rows with missing values: {Color.GREEN}{df.shape}{Color.OFF}")
    red(f"{prefix} ************+ END: delete_missing_values ************+ ")
    return df


def cleanse_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleanses the DataFrame by:
        1. Deleting duplicate rows
        2. Deleting columns with a single unique value
        3. Deleting columns with too many missing or zero values

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    red("************+ START: cleanse_raw_data ************+ ")

    # 1. Delete duplicate rows
    print("[INFO] Removing duplicate rows...")
    df = delete_duplicate_rows(df)

    # 2. Delete columns with single unique value
    print("[INFO] Removing columns with a single unique value...")
    df = delete_unique_columns(df)

    # 3. Delete columns with excessive missing or zero values
    print(f"[INFO] Removing columns with > {env.MISSING_VALUE_THRESHOLD}% missing or zero values...")
    df = delete_missing_values(df, threshold_in_percentage = env.MISSING_VALUE_THRESHOLD)

    red("************+ END: cleanse_raw_data ************+ ")
    return df