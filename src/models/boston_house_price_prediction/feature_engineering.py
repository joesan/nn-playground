from enum import Enum
import pandas as pd
from numpy import mean, std
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedKFold
from colorist import red, Color
from scipy.stats import shapiro


class ImputeStrategy(Enum):
    MEAN = 'mean'
    MEDIAN = 'median'
    MOST_FREQUENT = 'most_frequent'
    CONSTANT = ('constant', 0)


def split_features_target(df):
    # Extracting features (first 12 columns) as a DataFrame
    X = df.iloc[:, :-1]
    # Extracting target variable (last column) as a Series
    y = df.iloc[:, -1]
    return X, y


def impute(df, impute_strategy=ImputeStrategy.MEAN):
    # SimpleImputer does not work with empty DataFrame, so let us return the empty DataFrame back
    if len(df.index) == 0:
        return df
    # Create a SimpleImputer instance with the specified strategy
    if impute_strategy == ImputeStrategy.CONSTANT:
        imputer = SimpleImputer(strategy=impute_strategy.value[0], fill_value=impute_strategy.value[1])
    else:
        imputer = SimpleImputer(strategy=impute_strategy.value)

    # Fit the imputer to the data and transform it
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed


def evaluate_imputation_strategies(features, y, num_splits=10, num_repeats=3, rnd_state=1):
    # Check if features or target values are empty
    if len(features.index) <= 1 or len(y.index) <= 1 or (features.isna().all().all()) or (y.isna().all()):
        raise ValueError("Input features or target values are empty.")
    # Evaluate each strategy on the dataset
    results = {}
    scores_dict = {}
    for strategy in ImputeStrategy:
        # Check if strategy is a constant strategy
        if strategy == ImputeStrategy.CONSTANT:
            imputer_kwargs = {'strategy': strategy.value[0], 'fill_value': strategy.value[1]}
        else:
            imputer_kwargs = {'strategy': strategy.value}
        # Create the modeling pipeline
        pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(**imputer_kwargs)),
            ('model', LinearRegression())
        ])
        # Evaluate the model
        cv = RepeatedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=rnd_state)
        scores = cross_val_score(pipeline, features, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
        # Convert scores to positive
        scores = -scores
        # Store results
        scores_dict[strategy.value] = scores
        results[strategy.value] = {'mean': mean(scores), 'std': std(scores)}
        print(f'>{strategy.value} {mean(scores):.3f} ({std(scores):.3f})')
    # Identify the best strategy based on the mean scores
    mean_scores = {strategy: mean(scores) for strategy, scores in scores_dict.items()}

    if len(set(mean_scores.values())) == 1:
        # All mean scores are equal, we choose constant as our imputation strategy
        best_strategy = ImputeStrategy.CONSTANT.value
    else:
        # Choose the strategy with the highest mean score
        best_strategy = max(mean_scores, key=mean_scores.get)
    print("Best imputation strategy:", best_strategy)
    return results, scores_dict, best_strategy


def check_for_normality(df):
    """
    Performs Shapiro-Wilk normality test on each column of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        None
    """
    print("Shapiro-Wilk Normality Test Results:")
    for column in df.columns:
        stat, p = shapiro(df[column])
        if p < 0.05:
            print(f"Column '{column}': {Color.GREEN}H0 is rejected because p-value is less than 0.05{Color.OFF} Statistics={stat:.3f}, p-value={p:.3f}")
        else:
            print(f"Column '{column}': Statistics={stat:.3f}")


def identify_categorical_features(df, min_unique_as_categorical=2, max_unique_as_categorical=2):
    # Select columns with object or category dtype
    categorical_columns = list(df.select_dtypes(include=['object', 'category']).columns)
    # Identify integer/float columns that should be treated as categorical
    potential_categorical = df.select_dtypes(include=['int64', 'float64']).apply(
        lambda col: min_unique_as_categorical <= col.nunique() <= max_unique_as_categorical
    )
    # Combine categorical columns and identified potential categorical columns
    categorical_columns += list(potential_categorical[potential_categorical].index)
    return categorical_columns
