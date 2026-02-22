import pandas as pd
from colorist import red


prefix = "    "

def engineer_features(df: pd.DataFrame, target_column: str = "target") -> tuple[pd.DataFrame, pd.Series]:
    """
    Perform feature engineering on the cleaned DataFrame.

    Args:
        df (pd.DataFrame): Cleaned DataFrame.
        target_column (str): Name of the target variable column.

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
    """

    # --------------------------
    # 1. Separate target
    # --------------------------
    red(f"{prefix} ************+ START: Extract Target Column ************+")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    y = df[target_column]
    X = df.drop(columns=[target_column])
    print(f"{prefix} Target Colum Extracted is = {y.name}")
    red(f"{prefix} ************+ END: Extract Target Column ************+")

    # --------------------------
    # 2. Encode categorical variables
    # --------------------------
    red(f"{prefix} ************+ START: Encode Categorical Columns ************+")
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        print(f"{prefix} Encoding categorical columns: {list(categorical_cols)}")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    red(f"{prefix} ************+ END: Encode Categorical Columns ************+")

    # --------------------------
    # 3. Optional: scale numeric features
    # --------------------------
    # from sklearn.preprocessing import StandardScaler
    # numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    # scaler = StandardScaler()
    # X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    return X, y
