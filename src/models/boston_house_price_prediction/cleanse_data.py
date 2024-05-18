

# delete columns with a single unique value
def delete_unique_columns(df):
    print("************+ delete_unique_columns ************+ ")
    print("Shape of boston before deleting unique columns", df.shape)
    # get number of unique values for each column
    counts = df.nunique()
    print("Are there any columns with unique values?", counts == 1)
    if (counts == 1):
        # record columns to delete
        to_del = [i for i,v in enumerate(counts) if v == 1]
        # drop useless columns
        df.drop(to_del, axis=1, inplace=True)
        print("Shape of boston after deleting unique columns", df.shape)
        print("************+ delete_unique_columns ************+ ")
    else:
        print("************+ delete_unique_columns ************+ ")
        return df


# delete rows containing duplicate data from the dataset
def delete_duplicate_rows(df):
    print("************+ delete_duplicate_rows ************+ ")
    # delete duplicate rows
    dups = df.duplicated()
    # report if there are any duplicates
    print("Are there any duplicates? ", dups.any())
    if (dups.any()):
        print(df[dups])
        df.drop_duplicates(inplace=True)
        print(df.shape)
        print("************+ delete_duplicate_rows ************+ ")
        return df
    else:
        print("************+ delete_duplicate_rows ************+ ")
        return df