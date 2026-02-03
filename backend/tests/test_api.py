"""
API evals: health, frontend, and /plan-trip contract so pushes don't break the app.
"""
import pytest
from fastapi.testclient import TestClient

# Import after conftest sets TEST_MODE so main uses fake LLM
from main import app

client = TestClient(app)


def test_health_returns_200_and_healthy():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "healthy"
    assert "ai-trip-planner" in data.get("service", "")


def test_health_returns_json():
    r = client.get("/health")
    assert r.headers.get("content-type", "").startswith("application/json")


def test_root_serves_frontend_or_message():
    r = client.get("/")
    assert r.status_code == 200
    # Either HTML or JSON fallback if file missing
    ct = r.headers.get("content-type", "")
    if "text/html" in ct:
        assert "Trip" in r.text or "trip" in r.text.lower()
    else:
        assert r.json().get("message") is not None


def test_plan_trip_requires_post():
    r = client.get("/plan-trip")
    assert r.status_code == 405


def test_plan_trip_rejects_empty_body():
    r = client.post("/plan-trip", json={})
    assert r.status_code == 422  # validation error


def test_plan_trip_accepts_minimal_valid_body():
    r = client.post(
        "/plan-trip",
        json={"destination": "Kyoto", "duration": "3 days"},
    )
    assert r.status_code == 200, r.text


def test_plan_trip_response_shape():
    r = client.post(
        "/plan-trip",
        json={
            "destination": "Tokyo",
            "duration": "5 days",
            "when": "March 2025",
            "budget": "moderate",
            "interests": "food, temples",
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "result" in data
    assert "tool_calls" in data
    assert isinstance(data["tool_calls"], list)
    # In TEST_MODE the fake LLM returns "Test itinerary"
    assert isinstance(data["result"], str)
    assert len(data["result"]) > 0


def test_plan_trip_optional_fields_accepted():
    r = client.post(
        "/plan-trip",
        json={
            "destination": "Prague",
            "duration": "2 days",
            "user_input": "I love beer",
            "session_id": "eval-session-1",
            "user_id": "eval-user",
            "turn_index": 0,
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "result" in data and "tool_calls" in data


def test_plan_trip_missing_destination_rejected():
    r = client.post("/plan-trip", json={"duration": "3 days"})
    assert r.status_code == 422


def test_plan_trip_missing_duration_rejected():
    r = client.post("/plan-trip", json={"destination": "Paris"})
    assert r.status_code == 422
