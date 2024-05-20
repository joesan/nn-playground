from colorist import red, Color, BrightColor


# delete columns with a single unique value
def delete_unique_columns(df):
    red("************+ delete_unique_columns ************+ ")
    print(f"Shape of boston before deleting columns with unique values: {Color.GREEN}{df.shape}{Color.OFF}")
    # get number of unique values for each column
    counts = df.nunique()
    print(f"Are there any columns with unique values? {Color.GREEN}{(counts == 1).any()}{Color.OFF}")
    if (counts == 1).any():
        # record columns to delete
        to_del = [i for i,v in enumerate(counts) if v == 1]
        # drop useless columns
        df.drop(to_del, axis=1, inplace=True)
        print(f"Shape of boston after deleting unique columns: {Color.GREEN}{df.shape}{Color.OFF}")
        red("************+ delete_unique_columns ************+ ")
    else:
        red("************+ delete_unique_columns ************+ ")
        return df


# delete rows containing duplicate data from the dataset
def delete_duplicate_rows(df):
    red("************+ delete_duplicate_rows ************+ ")
    print(f"Shape of boston before deleting duplicated rows: {Color.GREEN}{df.shape}{Color.OFF}")
    # delete duplicate rows
    dups = df.duplicated()
    # report if there are any duplicates
    print(f"Are there any duplicate rows? {Color.GREEN}{dups.any()}{Color.OFF}")
    if (dups.any()):
        print(df[dups])
        df.drop_duplicates(inplace=True)
        print(df.shape)
        red("************+ delete_duplicate_rows ************+ ")
        return df
    else:
        red("************+ delete_duplicate_rows ************+ ")
        return df

def delete_missing_values(df):
    red("************+ delete_missing_values ************+ ")
    print(f"Shape of boston before deleting rows with missing values: {Color.GREEN}{df.shape}{Color.OFF}")
    has_zeros = (df == 0).any()
    has_nans = df.isna().any()
    if (has_zeros.any()):
        print(f"Dataframe has zeros? {Color.GREEN}{has_zeros.any()}{Color.OFF}")
        # Print the columns that contain zero values
        for column, has_zero in has_zeros.items():
            if has_zero:
                print(f"\t Column {Color.GREEN}{column}{Color.OFF} contains zero values.")
    if (has_nans.any()):
        print(f"Dataframe has nan? {Color.GREEN}{has_nans.any()}{Color.OFF}")
        # Print the columns that contain zero values
        for column, has_nan in has_nans.items():
            if has_nan:
                print(f"\t Column {Color.GREEN}{column}{Color.OFF} contains nan values.")
    return df