from enum import Enum
import pandas as pd
from numpy import mean, std
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedKFold


class ImputeStrategy(Enum):
    MEAN = 'mean'
    MEDIAN = 'median'
    MODE = 'mode'
    CONSTANT = 'constant'


def impute(df, impute_strategy=ImputeStrategy.MEAN):
    # Create a SimpleImputer instance with the specified strategy
    if impute_strategy == ImputeStrategy.CONSTANT:
        imputer = SimpleImputer(strategy=impute_strategy.value, fill_value=0)  # Example constant value
    else:
        imputer = SimpleImputer(strategy=impute_strategy.value)

    # Fit the imputer to the data and transform it
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    return df_imputed


def evaluate_imputation_strategies(df):
    # Evaluate each strategy on the dataset
    results = list()
    for s in ImputeStrategy:
        # Create the modeling pipeline
        pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=s)),
            ('model', LinearRegression())
        ])
        # Evaluate the model
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(pipeline, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
        # Convert scores to positive
        scores = -scores
        # Store results
        results.append(scores)
        print('>%s %.3f (%.3f)' % (s, mean(scores), std(scores)))