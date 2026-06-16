"""
Unit tests for src/visualization/explainability.py.
Tests the Python module directly — no API or Docker required.
Uses a lightweight LinearRegression model to avoid XGBoost dependency in CI.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.visualization.explainability import (
    get_explainer,
    explain_prediction,
    explain_batch,
    generate_explanation_dict,
    MODEL_TARGET,
)

FEATURES = ["MarksInside50", "Disposals", "Age", "BMI", "HitOuts"]
N_BACKGROUND = 30
N_TEST = 5


@pytest.fixture(scope="module")
def linear_model():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, len(FEATURES)), columns=FEATURES)
    y = X["MarksInside50"] * 2.0 + X["Disposals"] * 0.5 + np.random.rand(100) * 0.1
    model = LinearRegression().fit(X, y)
    return model


@pytest.fixture(scope="module")
def background(linear_model):
    np.random.seed(0)
    return pd.DataFrame(np.random.rand(N_BACKGROUND, len(FEATURES)), columns=FEATURES)


@pytest.fixture(scope="module")
def instance(background):
    return background.iloc[[0]]


def test_model_target_is_goals():
    """Production model is a single regressor predicting Goals for every position."""
    assert MODEL_TARGET == "Goals"


def test_get_explainer_returns_explainer(linear_model, background):
    explainer = get_explainer(linear_model, background)
    assert explainer is not None


def test_explain_prediction_keys(linear_model, instance, background):
    result = explain_prediction(linear_model, instance, background)
    assert "shap_values"    in result
    assert "expected_value" in result
    assert "feature_names"  in result
    assert "feature_values" in result
    assert "top_features"   in result


def test_explain_prediction_feature_count(linear_model, instance, background):
    result = explain_prediction(linear_model, instance, background)
    assert len(result["shap_values"]) == len(FEATURES)
    assert len(result["feature_names"]) == len(FEATURES)
    assert len(result["top_features"]) == len(FEATURES)


def test_explain_prediction_top_features_sorted(linear_model, instance, background):
    result = explain_prediction(linear_model, instance, background)
    shap_abs = [abs(f["shap_value"]) for f in result["top_features"]]
    assert shap_abs == sorted(shap_abs, reverse=True), "top_features not sorted by |SHAP|"


def test_explain_prediction_direction_matches_sign(linear_model, instance, background):
    result = explain_prediction(linear_model, instance, background)
    for f in result["top_features"]:
        if f["shap_value"] > 0:
            assert f["direction"] == "positive"
        elif f["shap_value"] < 0:
            assert f["direction"] == "negative"


def test_explain_batch_shape(linear_model, background):
    shap_arr = explain_batch(linear_model, background, background)
    assert shap_arr.shape == (N_BACKGROUND, len(FEATURES))


def test_generate_explanation_dict_keys(linear_model, instance, background):
    result = generate_explanation_dict(
        model=linear_model,
        X_instance=instance,
        X_background=background,
        position="Forward",
        player_id="test_player",
        top_n=3,
    )
    assert result["player_id"] == "test_player"
    assert result["position"]  == "Forward"
    assert result["target"]    == MODEL_TARGET
    assert "baseline"          in result
    assert "prediction"        in result
    assert len(result["top_features"]) == 3


def test_generate_explanation_dict_prediction_consistent(linear_model, instance, background):
    result = generate_explanation_dict(
        model=linear_model,
        X_instance=instance,
        X_background=background,
        position="Midfield",
        player_id="p1",
        top_n=5,
    )
    expected = float(linear_model.predict(instance)[0])
    assert abs(result["prediction"] - expected) < 0.5, "SHAP prediction deviates too much from model prediction"
