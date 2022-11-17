"""Module to generate and train pipelines of titanic dataset"""

import pickle
from enum import Enum
import logging

import pandas as pd

from sklearn.base import ClassifierMixin

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer


class Strategy(str, Enum):
    """Type of strategies allowed"""

    MEAN = "mean"
    MEDIAN = "median"
    MOST_FREQ = "most_frequent"


def make_pipeline(
    num_vars: tuple = (),
    cat_vars: tuple = (),
    use_pca: bool = False,
    components: (int | None) = None,
    num_imp_strategy: Strategy = Strategy.MEAN,
    classifier: ClassifierMixin = GradientBoostingClassifier(),
    scaler: (MinMaxScaler | StandardScaler) = MinMaxScaler(),
) -> Pipeline:
    """Allows the generation of a custom pipeline

    :param num_vars: numerical fields used in classification. Defaults to ().
    :type num_vars: tuple
    :param cat_vars: categorical fields used in classification. Defaults to ().
    :type cat_vars: tuple
    :param use_pca: if True activates the dimensionality reduction by means of a PCA.
    Defaults to True.
    :type use_pca: bool
    :param components: total number of fields selected by the PCA. Defaults to 3.
    :type components: int or None
    :param num_imp_strategy: strategy of imputation. Defaults to Strategy.MEAN.
    :type num_imp_strategy: Strategy
    :param classifier: model used to classify. Defaults to GradientBoostingClassifier().
    :type classifier: ClassifierMixin
    :param scaler: type of scaller used. Defaults to MinMaxScaler().
    :type scaler:  MinMaxScaler  or  StandardScaler
    :return: processed pipeline
    :rtype: Pipeline
    """

    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy=num_imp_strategy)), ("ss", scaler)]
    )

    columns_trans = ColumnTransformer(
        [("cat", OneHotEncoder(), cat_vars), ("num", num_pipeline, num_vars)]
    )

    steps = [("ct", columns_trans)]

    if use_pca and components is not None:
        steps.append(("pca", PCA(n_components=components)))

    steps.append(("model", classifier))

    return Pipeline(steps=steps)


def fit(pipeline: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame) -> Pipeline:
    """Trains the model

    :param pipeline: pipeline to train
    :type pipeline: Pipeline
    :param x_train: dataset to train
    :type x_train:  DataFrame
    :param y_train: labels of the dataset
    :type y_train:  DataFrame
    :raises ValueError: is raised when there aren't features or when components
    is greater than number of features
    :return: trained pipeline
    :rtype: Pipeline
    """
    try:
        pipeline.fit(x_train, y_train)
    except ValueError as exce:
        logging.error("The number of components exceeds the number of features")
        raise ValueError from exce

    return pipeline


def export(pipeline: Pipeline, file: str) -> None:
    """Saves the model in the path passed in the file parameter

    :param pipeline: model to save
    :type pipeline: Pipeline
    :param file: path for saving the model
    :type file: str
    """
    with open(file, "wb") as file_descriptor:
        pickle.dump(pipeline, file_descriptor)
