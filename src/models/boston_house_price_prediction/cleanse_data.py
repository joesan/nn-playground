

# delete columns with a single unique value
def delete_unique_columns(df):
    print(df.shape)
    # get number of unique values for each column
    counts = df.nunique()
    # record columns to delete
    to_del = [i for i,v in enumerate(counts) if v == 1]
    # drop useless columns
    df.drop(to_del, axis=1, inplace=True)
    print(df.shape)
    return df

def delete_duplicate_rows(df):
    # delete rows of duplicate data from the dataset
    from pandas import read_csv
    # load the dataset
    df = read_csv(' iris.csv ', header=None)
    print(df.shape)
    # delete duplicate rows
    df.drop_duplicates(inplace=True)
    print(df.shape)
    return df