"""Smoke tests for POST /predict endpoint."""

import pytest
import requests
from tests.conftest import requires_api, BASE_URL, ALL_POSITION_PAYLOADS, FORWARD_PAYLOAD


@requires_api
@pytest.mark.parametrize("position,payload", ALL_POSITION_PAYLOADS)
def test_predict_returns_200(position, payload):
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    assert r.status_code == 200, f"{position}: expected 200, got {r.status_code} — {r.text}"


@requires_api
@pytest.mark.parametrize("position,payload", ALL_POSITION_PAYLOADS)
def test_predict_response_schema(position, payload):
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    body = r.json()
    assert "predicted_goals" in body, f"{position}: missing predicted_goals"
    assert "model_version" in body, f"{position}: missing model_version"
    assert isinstance(body["predicted_goals"], float), f"{position}: predicted_goals not float"


@requires_api
@pytest.mark.parametrize("position,payload", ALL_POSITION_PAYLOADS)
def test_predict_returns_non_negative(position, payload):
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    assert r.json()["predicted_goals"] >= 0, f"{position}: negative prediction"


@requires_api
def test_predict_forward_higher_than_defender():
    """Forward with active scorer profile should predict more goals than a Defender."""
    from tests.conftest import FORWARD_PAYLOAD, DEFENDER_PAYLOAD
    forward = requests.post(f"{BASE_URL}/predict", json=FORWARD_PAYLOAD).json()["predicted_goals"]
    defender = requests.post(f"{BASE_URL}/predict", json=DEFENDER_PAYLOAD).json()["predicted_goals"]
    assert forward > defender, f"Forward ({forward:.3f}) should outscore Defender ({defender:.3f})"


@requires_api
def test_predict_missing_field_returns_422():
    incomplete = {k: v for k, v in FORWARD_PAYLOAD.items() if k != "Percent_Played"}
    r = requests.post(f"{BASE_URL}/predict", json=incomplete)
    assert r.status_code == 422


@requires_api
def test_predict_wrong_type_returns_422():
    bad_payload = {**FORWARD_PAYLOAD, "Year": "twenty-twenty-five"}
    r = requests.post(f"{BASE_URL}/predict", json=bad_payload)
    assert r.status_code == 422
