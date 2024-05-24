from colorist import red, Color, BrightColor
import pandas as pd
from src.models.boston_house_price_prediction import feature_engineering


# delete columns with a single unique value
def delete_unique_columns(df):
    red("************+ delete_unique_columns ************+ ")
    print(f"Shape of DataFrame before deleting columns with unique values: {Color.GREEN}{df.shape}{Color.OFF}")
    # get number of unique values for each column
    counts = df.nunique()
    print(f"Are there any columns with unique values? {Color.GREEN}{(counts == 1).any()}{Color.OFF}")
    if counts.any() == 1:
        # record columns to delete by their names
        to_del = [col for col in counts.index if counts[col] == 1]
        # drop useless columns
        df.drop(to_del, axis=1, inplace=True)
        print(f"Shape of DataFrame after deleting unique columns: {Color.GREEN}{df.shape}{Color.OFF}")
    red("************+ delete_unique_columns ************+ ")
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
    red("************+ delete_duplicate_rows ************+ ")
    print(f"Shape of boston before deleting duplicated rows: {Color.GREEN}{df.shape}{Color.OFF}")
    # delete duplicate rows
    dupes = df.duplicated()
    # report if there are any duplicates
    print(f"Are there any duplicate rows? {Color.GREEN}{dupes.any()}{Color.OFF}")
    if dupes.any():
        df.drop_duplicates(inplace=True)
        print(f"Shape of boston after deleting duplicated rows: {Color.GREEN}{df.shape}{Color.OFF}")
        red("************+ delete_duplicate_rows ************+ ")
        return df
    else:
        red("************+ delete_duplicate_rows ************+ ")
        return df


def delete_missing_values(df, threshold_in_percentage=70):
    red("************+ delete_missing_values ************+ ")
    print(f"Shape of boston before deleting rows with missing values: {Color.GREEN}{df.shape}{Color.OFF}")
    zeros_count = (df == 0).sum()
    nans_count = df.isna().sum()
    columns_to_drop = []
    for column in df.columns:
        zero_count = zeros_count[column] / len(df[column]) * 100
        nan_count = nans_count[column] / len(df[column]) * 100
        if zero_count > threshold_in_percentage or nan_count > threshold_in_percentage:
            print(f"\t{Color.GREEN}Column {column} will be dropped because of either: Zeros = {zeros_count[column]}, NaNs = {nans_count[column]}{Color.OFF}")
            columns_to_drop.append(column)
        else:
            print(f"\tColumn {column}: Zeros = {zeros_count[column]}, NaNs = {nans_count[column]}")
    df.drop(columns=columns_to_drop, inplace=True)
    print(f"Shape of boston after deleting rows with missing values: {Color.GREEN}{df.shape}{Color.OFF}")
    red("************+ delete_missing_values ************+ ")
    return df

