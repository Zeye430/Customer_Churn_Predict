# churning_pipeline.py
"""
Shared ML pipeline components for the churning project.

This module holds all custom transformers and helper functions that are used
both in training and in inference (FastAPI app), so that joblib pickles
refer to a stable module path: `churning_pipeline.<name>`.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# =============================================================================
# Custom transformer
# =============================================================================

def column_ratio(X):
    """
    Calculate ratio of first column to second column.
    Works for numpy arrays and pandas DataFrames.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    return X[:, [0]] / (X[:, [1]] + 1e-5)


def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]


def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler(),
    )


# =============================================================================
# Building blocks for preprocessing
# =============================================================================

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log1p, feature_names_out="one-to-one"), 
    StandardScaler(),
)

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
)

default_num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
)

def build_preprocessing():
    """
    Define ColumnTransformer
    """
    preprocessing = ColumnTransformer(
        [
            # --- Feature Engineering ---
            # 1. Create new feature: balance/salary ratio
            ("bal_sal_ratio",   ratio_pipeline(), ["balance", "estimated_salary"]),
            # 2. Create new feature: tenure/age ratio
            ("tenure_age_ratio",ratio_pipeline(), ["tenure", "age"]),
            
            # --- Transformations ---
            # 3. log pipline to long-tail distribution features
            ("log",             log_pipeline,     ["age", "balance"]),
            
            # 4. One-Hot to categorical features
            ("cat",             cat_pipeline,     make_column_selector(dtype_include=object)),
        ],
        # 5. Other numerical features use default pipeline
        remainder=default_num_pipeline,
    )
    return preprocessing

# =============================================================================
# Estimator factory used by both non-Optuna and Optuna code
# =============================================================================

def make_estimator_for_name(name: str):
    """
    Given a model name, return an unconfigured estimator instance.
    """
    if name == "ridge":
        return RidgeClassifier(random_state=42)
        
    elif name == "logistic":
        return LogisticRegression(random_state=42, solver='liblinear')
    
    elif name == "random_forest":
        return RandomForestClassifier(random_state=42, n_jobs=-1)
        
    elif name == "histgradientboosting":
        return HistGradientBoostingClassifier(random_state=42)
        
    elif name == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            n_jobs=-1,
        )
        
    elif name == "lightgbm":
        return LGBMClassifier(
            objective="binary",
            random_state=42,
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown model name: {name}")