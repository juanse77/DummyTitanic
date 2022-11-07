"""Módulo de prueba"""
import pytest
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from titanic_model_processor.main import make_pipeline, fit


@pytest.fixture
def data():
    """Loads the dataframe to be used in the tests

    Returns:
        DataFrame: loaded data
    """
    x_data, y_data = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

    x_data.rename(columns={"home.dest": "homedest"}, inplace=True)

    return x_data, y_data


def test_fit_minimum_correct(data):
    """Checks if default pipeline works correctly"""
    x_data, y_data = data
    pipeline = make_pipeline(num_vars=("age",))

    try:
        fit(pipeline, x_data, y_data)
    except Exception:
        assert False

    pipeline = make_pipeline(cat_vars=("sex",))

    try:
        fit(pipeline, x_data, y_data)
    except Exception:
        assert False

    assert True


def test_other_classifier(data):
    """Checks if other classifier is acepted"""

    x_data, y_data = data
    pipeline = make_pipeline(cat_vars=("sex",), classifier=RandomForestClassifier())

    try:
        fit(pipeline, x_data, y_data)
    except Exception:
        assert False

    assert True


def test_other_scaler(data):
    """Checks if standard scaler is acepted"""

    x_data, y_data = data
    pipeline = make_pipeline(cat_vars=("sex",), scaler=StandardScaler())

    try:
        fit(pipeline, x_data, y_data)
    except Exception:
        assert False

    assert True


def test_fit_void_value_error(data):
    """Void value error test"""

    x_data, y_data = data

    pipeline = make_pipeline(num_vars=(), cat_vars=())

    with pytest.raises(ValueError):
        fit(pipeline, x_data, y_data)


def test_fit_components_value_error(data):
    """Components value error test"""

    x_data, y_data = data

    pipeline = make_pipeline(
        num_vars=("age",), cat_vars=("sex",), use_pca=True, components=10
    )

    with pytest.raises(ValueError):
        fit(pipeline, x_data, y_data)
