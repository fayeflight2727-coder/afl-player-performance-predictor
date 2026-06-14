"""Smoke tests for POST /predict/explain endpoint."""

import pytest
import requests
from tests.conftest import requires_api, BASE_URL, FORWARD_PAYLOAD, ALL_POSITION_PAYLOADS


@requires_api
@pytest.mark.parametrize("position,payload", ALL_POSITION_PAYLOADS)
def test_explain_returns_200(position, payload):
    r = requests.post(f"{BASE_URL}/predict/explain?position={position}", json=payload)
    assert r.status_code == 200, f"{position}: expected 200, got {r.status_code} — {r.text}"


@requires_api
def test_explain_response_schema():
    r = requests.post(f"{BASE_URL}/predict/explain?position=Forward", json=FORWARD_PAYLOAD)
    body = r.json()
    assert "player_id"    in body
    assert "position"     in body
    assert "target"       in body
    assert "baseline"     in body
    assert "prediction"   in body
    assert "top_features" in body
    assert isinstance(body["top_features"], list)
    assert len(body["top_features"]) > 0


@requires_api
def test_explain_top_feature_schema():
    r = requests.post(f"{BASE_URL}/predict/explain?position=Forward", json=FORWARD_PAYLOAD)
    feature = r.json()["top_features"][0]
    assert "feature"    in feature
    assert "value"      in feature
    assert "shap_value" in feature
    assert "direction"  in feature
    assert feature["direction"] in ("positive", "negative")


@requires_api
def test_explain_shap_values_are_nonzero():
    """After background fix, at least one SHAP value should be non-zero."""
    r = requests.post(f"{BASE_URL}/predict/explain?position=Forward", json=FORWARD_PAYLOAD)
    shap_values = [f["shap_value"] for f in r.json()["top_features"]]
    assert any(abs(v) > 0 for v in shap_values), "All SHAP values are zero — background fix may not be applied"


@requires_api
def test_explain_prediction_matches_predict():
    """explain prediction value should match /predict for same input."""
    pred = requests.post(f"{BASE_URL}/predict", json=FORWARD_PAYLOAD).json()["predicted_goals"]
    expl = requests.post(f"{BASE_URL}/predict/explain?position=Forward", json=FORWARD_PAYLOAD).json()["prediction"]
    assert abs(pred - expl) < 0.01, f"predict={pred:.4f} vs explain={expl:.4f} — should match"


@requires_api
def test_explain_forward_target():
    r = requests.post(f"{BASE_URL}/predict/explain?position=Forward", json=FORWARD_PAYLOAD)
    assert r.json()["target"] == "Total_Score"


@requires_api
@pytest.mark.parametrize("position,expected_target", [
    ("Forward",  "Total_Score"),
    ("Midfield", "Clearances"),
    ("Ruck",     "HitOuts"),
    ("Defender", "Rebounds"),
])
def test_explain_target_by_position(position, expected_target):
    from tests.conftest import ALL_POSITION_PAYLOADS
    payload = dict(ALL_POSITION_PAYLOADS)[position]
    r = requests.post(f"{BASE_URL}/predict/explain?position={position}", json=payload)
    assert r.json()["target"] == expected_target
