"""Smoke tests for GET /health and GET /ready endpoints."""

import requests
from tests.conftest import requires_api, BASE_URL


@requires_api
def test_health_returns_ok():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@requires_api
def test_ready_returns_200():
    r = requests.get(f"{BASE_URL}/ready")
    assert r.status_code == 200


@requires_api
def test_ready_has_required_fields():
    r = requests.get(f"{BASE_URL}/ready")
    body = r.json()
    assert "status" in body
    assert "model_loaded" in body
    assert isinstance(body["model_loaded"], bool)


@requires_api
def test_ready_status_matches_model_loaded():
    r = requests.get(f"{BASE_URL}/ready")
    body = r.json()
    if body["model_loaded"]:
        assert body["status"] == "ready"
    else:
        assert body["status"] == "not ready"
