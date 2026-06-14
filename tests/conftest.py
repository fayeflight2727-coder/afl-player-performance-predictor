"""
Shared fixtures and helpers for AFL Predictor smoke tests.
Tests marked with @requires_api need `docker compose up` running first.
"""

import pytest
import requests

BASE_URL = "http://localhost:8000"


def is_api_running() -> bool:
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
        return True
    except Exception:
        return False


requires_api = pytest.mark.skipif(
    not is_api_running(),
    reason="API not running — start with: docker compose up"
)

# ── Shared payloads ───────────────────────────────────────────────────────────

FORWARD_PAYLOAD = {
    "Year": 2025, "Disposals": 22, "Marks": 6, "Behinds": 1, "HitOuts": 0,
    "Tackles": 4, "Rebounds": 1, "Inside50s": 5, "Clearances": 3, "Clangers": 2,
    "Frees": 1, "FreesAgainst": 1, "ContestedMarks": 1, "MarksInside50": 3,
    "OnePercenters": 2, "GoalAssists": 1, "Percent_Played": 85.0,
    "Height": 185, "Weight": 87, "BMI": 25.4, "AvgTemp": 18.5,
    "TempRange": 8.0, "IsRainy": 0, "Age": 26
}

MIDFIELD_PAYLOAD = {
    "Year": 2025, "Disposals": 28, "Marks": 4, "Behinds": 0, "HitOuts": 0,
    "Tackles": 6, "Rebounds": 2, "Inside50s": 3, "Clearances": 8, "Clangers": 3,
    "Frees": 2, "FreesAgainst": 2, "ContestedMarks": 1, "MarksInside50": 0,
    "OnePercenters": 1, "GoalAssists": 2, "Percent_Played": 100.0,
    "Height": 182, "Weight": 83, "BMI": 25.1, "AvgTemp": 18.5,
    "TempRange": 8.0, "IsRainy": 0, "Age": 25
}

RUCK_PAYLOAD = {
    "Year": 2025, "Disposals": 18, "Marks": 5, "Behinds": 1, "HitOuts": 35,
    "Tackles": 3, "Rebounds": 2, "Inside50s": 2, "Clearances": 5, "Clangers": 2,
    "Frees": 1, "FreesAgainst": 1, "ContestedMarks": 2, "MarksInside50": 1,
    "OnePercenters": 3, "GoalAssists": 1, "Percent_Played": 100.0,
    "Height": 201, "Weight": 105, "BMI": 25.9, "AvgTemp": 18.5,
    "TempRange": 8.0, "IsRainy": 0, "Age": 27
}

DEFENDER_PAYLOAD = {
    "Year": 2025, "Disposals": 20, "Marks": 7, "Behinds": 0, "HitOuts": 0,
    "Tackles": 3, "Rebounds": 8, "Inside50s": 1, "Clearances": 2, "Clangers": 1,
    "Frees": 1, "FreesAgainst": 2, "ContestedMarks": 1, "MarksInside50": 0,
    "OnePercenters": 5, "GoalAssists": 0, "Percent_Played": 100.0,
    "Height": 186, "Weight": 88, "BMI": 25.4, "AvgTemp": 18.5,
    "TempRange": 8.0, "IsRainy": 0, "Age": 28
}

ALL_POSITION_PAYLOADS = [
    ("Forward",  FORWARD_PAYLOAD),
    ("Midfield", MIDFIELD_PAYLOAD),
    ("Ruck",     RUCK_PAYLOAD),
    ("Defender", DEFENDER_PAYLOAD),
]
