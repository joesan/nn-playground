from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from colorist import red, Color, BrightColor


def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=chi2, k= ' all ' )
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def delete_features_with_low_correlation(df, target_column, threshold = 0.1):
    red("************+ delete_features_with_low_correlation ************+ ")
    print(f"Shape of boston before deleting features with low co-relation {Color.GREEN}{df.shape}{Color.OFF}")
    # Compute the correlation matrix
    correlation_matrix = df.corr()
    # Extract correlations of all features with the target
    correlations_with_target = correlation_matrix[target_column].drop(target_column)
    # Identify features with correlation coefficient less than threshold
    low_correlation_features = correlations_with_target[correlations_with_target.abs() < threshold].index
    print(f"Features with low correlation (|r| < {threshold}): {Color.GREEN}{list(low_correlation_features)}{Color.OFF}")
    # Remove these features from the DataFrame
    df_reduced = df.drop(columns=low_correlation_features)
    print(f"Shape of boston after deleting features with low co-relation: {Color.GREEN}{df_reduced.shape}{Color.OFF}")
    red("************+ delete_features_with_low_correlation ************+ ")
    return df_reduced
