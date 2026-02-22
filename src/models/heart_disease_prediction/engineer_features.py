import pandas as pd
from colorist import red
from sklearn.preprocessing import StandardScaler
import src.shared.env as env


prefix = "    "

def split_target(df: pd.DataFrame, target_column: str = "target"):
    red(f"{prefix} ************+ START: Extract Target Column ************+")

    if target_column not in df.columns:
        raise ValueError(f"{prefix} Target column '{target_column}' not found in DataFrame")

    y = df[target_column]
    X = df.drop(columns=[target_column])
    print(f"{prefix} Target Colum Extracted is = {y.name}")
    red(f"{prefix} ************+ END: Extract Target Column ************+")

    return X, y


def encode_categoricals(X: pd.DataFrame):
    red(f"{prefix} ************+ START: Encode Categorical Columns ************+")

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        print(f"{prefix} Encoding categorical columns: {list(categorical_cols)}")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    red(f"{prefix} ************+ END: Encode Categorical Columns ************+")

    return X


def scale_numeric_features(X: pd.DataFrame, y: pd.DataFrame, scaler: StandardScaler = None):
    red(f"{prefix} ************+ START: Scale Columns ************+")

    cols_to_scale = [col for col in env.NUMERIC_COLUMNS if col in X.columns]
    if len(cols_to_scale) > 0:
        if scaler is None:
            print(f"{prefix} Default Scaler is not set, so defaulting to StandardScaler()")
            scaler = StandardScaler()
            X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
            red(f"{prefix} ************+ END: Scale Columns ************+")
        else:
            print(f"{prefix} Scaling numeric columns using {scaler}")
            X[cols_to_scale] = scaler.transform(X[cols_to_scale])
            red(f"{prefix} ************+ END: Scale Columns ************+")

    check_rows_and_index_sanity(X, y)
    return X, y, scaler


def check_rows_and_index_sanity(X: pd.DataFrame, y: pd.DataFrame):
    if len(X) != len(y):
        raise ValueError("Row count mismatch between X and y.")

    if not X.index.equals(y.index):
        raise ValueError("Index mismatch between X and y.")

