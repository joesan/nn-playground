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
    MOST_FREQUENT = 'most_frequent'
    CONSTANT = ('constant', 0)


def impute(df, impute_strategy=ImputeStrategy.MEAN):
    # Create a SimpleImputer instance with the specified strategy
    if impute_strategy == ImputeStrategy.CONSTANT:
        imputer = SimpleImputer(strategy=impute_strategy.value, fill_value=0)  # Example constant value
    else:
        imputer = SimpleImputer(strategy=impute_strategy.value)

    # Fit the imputer to the data and transform it
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    return df_imputed


def evaluate_imputation_strategies(X, y):
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
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(pipeline, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
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