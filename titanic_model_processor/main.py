"""Module to generate and train pipelines of titanic dataset"""

import pickle
from enum import Enum

import pandas as pd

from sklearn.base import ClassifierMixin

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer


class Strategy(str, Enum):
    """Type of strategies allowed

    Args:
        str (_type_): string superclass
        Enum (_type_): superclass
    """

    MEAN = "mean"
    MEDIAN = "median"
    MOST_FREQ = "most_frequent"


def make_pipeline(
    use_pca: bool = True,
    components: int = 3,
    num_vars: tuple[str] = ("age",),
    cat_vars: tuple[str] = ("sex",),
    num_imp_strategy: Strategy = Strategy.MEAN,
    classifier: ClassifierMixin = GradientBoostingClassifier(),
    scaler: (MinMaxScaler | StandardScaler) = MinMaxScaler(),
) -> Pipeline:
    """Allows the generation of a custom pipeline

    Args:
        use_pca (bool, optional): if True activates the dimensionality reduction by means of a PCA.
        Defaults to True.
        components (int, optional): total number of fields selected by the PCA.
        Defaults to 3.
        num_vars (tuple[str], optional): numerical fields used in classification.
        Defaults to ("age",).
        cat_vars (tuple[str], optional): categorical fields used in classification.
        Defaults to ("sex",).
        num_imp_strategy (Strategy, optional): strategy of imputation. Defaults to Strategy.MEAN.
        classifier (ClassifierMixin, optional): model used to classify.
        Defaults to GradientBoostingClassifier().
        scaler (MinMaxScaler  |  StandardScaler, optional): type of scaller used.
        Defaults to MinMaxScaler().

    Returns:
        Pipeline: processed pipeline
    """

    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy=num_imp_strategy)), ("ss", scaler)]
    )

    columns_trans = ColumnTransformer(
        [("cat", OneHotEncoder(), cat_vars), ("num", num_pipeline, num_vars)]
    )

    steps = [("ct", columns_trans)]

    if use_pca:
        steps.append(("pca", PCA(n_components=components)))

    steps.append(("model", classifier))

    return Pipeline(steps=steps)


def fit(pipeline: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame) -> Pipeline:
    """Trains the model

    Args:
        pipeline (Pipeline): pipeline to train
        x_train (pd.DataFrame): dataset to train
        y_train (pd.DataFrame): labels of the dataset

    Returns:
        Pipeline: trained pipeline
    """
    return pipeline.fit(x_train, y_train)


def export(pipeline: Pipeline, file: str):
    """Saves the model in the path passed in the file parameter

    Args:
        pipeline (Pipeline): model to save
        file (str): path for saving the model
    """
    with open(file, "wb") as file_descriptor:
        pickle.dump(pipeline, file_descriptor)
