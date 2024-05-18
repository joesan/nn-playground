from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=chi2, k= ' all ' )
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs