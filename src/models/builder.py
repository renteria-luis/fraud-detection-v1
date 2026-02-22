# src/models/builder.py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.features.engineering import PaySimFeatures
from src.config import BINARY_FEATURES, NUMERIC_FEATURES, RANDOM_SEED


def build_pipeline(model_name: str = 'xgb', params: dict = None) -> Pipeline:
    params   = params or {}
    is_logreg = (model_name == 'logreg')

    num_steps = [('impute', SimpleImputer(strategy='median'))]
    if is_logreg:
        num_steps.append(('scaler', StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[
            ('bool', SimpleImputer(strategy='most_frequent'), BINARY_FEATURES),
            ('num', Pipeline(num_steps), NUMERIC_FEATURES),
        ],
        remainder='drop'
    )

    if model_name == 'logreg':
        model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, **params)
    elif model_name == 'rf':
        model = RandomForestClassifier(random_state=RANDOM_SEED, **params)
    elif model_name == 'xgb':
        model = XGBClassifier(
            random_state=RANDOM_SEED,
            tree_method='hist',
            device='cpu',
            eval_metric='aucpr',
            n_jobs=-1,
            **params
        )
    else:
        raise ValueError(f"Model {model_name} not supported.")

    return Pipeline([
        ('fe',           PaySimFeatures(cyclical_encoding=is_logreg)),
        ('preprocessor', preprocessor),
        ('model',        model),
    ])
    


# legacy: this function will no longer be used, because the new dataset (PaySim) has different features and requires a 
#         different approach. However, I am keeping it here for reference and potential reuse in other contexts.
from src.features.engineering import FeatureEngineering

def training_pipeline(model_name:str='rf', params:dict=None, seed:int=42) -> Pipeline:
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
        model = LogisticRegression(random_state=seed, max_iter=1000, **params)
    elif model_name == 'rf':
        model = RandomForestClassifier(random_state=seed, **params)
    elif model_name == 'xgb':
        model = XGBClassifier(random_state=seed, tree_method='hist', device='cpu', eval_metric='aucpr', n_jobs=-1, **params)
    else:
        raise ValueError(f"{model_name}: not supported model")
    
    return Pipeline([
        ('fe', FeatureEngineering()),    # creates new cols
        ('preprocessor', preprocessor),  # clasify and cleans them
        ('model', model)                 # predictor
    ])
