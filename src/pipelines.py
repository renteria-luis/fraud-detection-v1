from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.features import FeatureEngineering


def training_pipeline(model_name:str='rf', params:dict=None) -> Pipeline:
    """
    Builds a full ML pipeline: applies FeatureEngineering, preprocesses numeric and boolean features,
    and fits a classifier (LogisticRegression, RandomForest, or XGBoost) with optional parameters.
    """

    params = params or {}

    # creating rules
    bool_selector = make_column_selector(pattern='^is_')
    num_selector = make_column_selector(dtype_include=['number', 'float64', 'int64'], pattern='^(?!is_).*')

    num_steps = [
        ('impute', SimpleImputer(strategy='median'))
    ]
    if model_name == 'logreg':
        num_steps.append(('scaler', StandardScaler()))
    num_pipe = Pipeline(num_steps)

    bool_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipe, num_selector),
            ('bool', bool_pipe, bool_selector),
        ],
        remainder='drop'  # drops whatever doesn't fit the rules
    )

    if model_name == 'logreg':
        model = LogisticRegression(**params)
    elif model_name == 'rf':
        model = RandomForestClassifier(**params)
    elif model_name == 'xgb':
        model = XGBClassifier(**params)
    else:
        raise ValueError(f"{model_name}: not supported model")
    
    return Pipeline([
        ('fe', FeatureEngineering()),    # creates new cols
        ('preprocessor', preprocessor),  # clasify and cleans them
        ('classifier', model)            # predictor
    ])

