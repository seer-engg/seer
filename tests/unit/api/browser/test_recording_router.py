"""Tests for recording REST API router."""
import gzip
import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from seer.api.browser.recording_router import router

# Set up test app
app = FastAPI()
app.include_router(router, prefix="/api")


def _make_db_user(user_id="user-123"):
    user = MagicMock()
    user.user_id = user_id
    user.id = user_id
    return user


def _make_recording(recording_id=None, user=None, **overrides):
    """Create a mock SessionRecording."""
    rec = MagicMock()
    rec.id = recording_id or uuid4()
    rec.user = user
    rec.session_type = overrides.get("session_type", "interactive")
    rec.event_count = overrides.get("event_count", 10)
    rec.duration_ms = overrides.get("duration_ms", 5000)
    rec.compressed_size_bytes = overrides.get("compressed_size_bytes", 1024)
    rec.start_url = overrides.get("start_url", "https://example.com")
    rec.status = overrides.get("status", "completed")
    rec.created_at = MagicMock()
    rec.created_at.isoformat.return_value = "2024-01-01T00:00:00+00:00"
    rec.completed_at = MagicMock()
    rec.completed_at.isoformat.return_value = "2024-01-01T00:05:00+00:00"
    rec.browser_profile_id = overrides.get("browser_profile_id", str(uuid4()))
    rec.workflow_run_id = overrides.get("workflow_run_id", None)

    events = overrides.get("events", [{"type": 1}, {"type": 2}])
    rec.events_compressed = gzip.compress(json.dumps(events).encode("utf-8"))
    rec.delete = AsyncMock()
    return rec


def _make_test_app(user):
    test_app = FastAPI()
    test_app.include_router(router, prefix="/api")

    @test_app.middleware("http")
    async def add_user(request, call_next):
        request.state.db_user = user
        return await call_next(request)

    return test_app


class TestListRecordings:
    """Test GET /api/browser/recordings."""

    @patch("seer.api.browser.recording_router.SessionRecording")
    async def test_list_recordings_filters_by_user(self, mock_model):
        user = _make_db_user()
        rec1 = _make_recording(user=user)
        rec2 = _make_recording(user=user)

        filter_mock = MagicMock()
        filter_mock.count = AsyncMock(return_value=2)
        filter_mock.offset = MagicMock(return_value=filter_mock)
        filter_mock.limit = MagicMock(return_value=filter_mock)
        filter_mock.all = AsyncMock(return_value=[rec1, rec2])
        mock_model.filter = MagicMock(return_value=filter_mock)

        test_app = _make_test_app(user)
        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/browser/recordings")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["recordings"]) == 2


class TestGetRecording:
    """Test GET /api/browser/recordings/{id}."""

    @patch("seer.api.browser.recording_router.SessionRecording")
    async def test_get_metadata_returns_without_events(self, mock_model):
        user = _make_db_user()
        rec_id = uuid4()
        rec = _make_recording(recording_id=rec_id, user=user)
        mock_model.get_or_none = AsyncMock(return_value=rec)

        test_app = _make_test_app(user)
        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/api/browser/recordings/{rec_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == str(rec_id)
        assert data["event_count"] == 10
        # Should not include raw events
        assert "events" not in data


class TestGetEvents:
    """Test GET /api/browser/recordings/{id}/events."""

    @patch("seer.api.browser.recording_router.RecordingService")
    @patch("seer.api.browser.recording_router.SessionRecording")
    async def test_get_events_decompresses(self, mock_model, mock_service):
        user = _make_db_user()
        rec_id = uuid4()
        events = [{"type": 1, "data": "test"}, {"type": 2, "data": "test2"}]
        rec = _make_recording(recording_id=rec_id, user=user, events=events)
        mock_model.get_or_none = AsyncMock(return_value=rec)
        mock_service.get_recording_events = AsyncMock(return_value=events)

        test_app = _make_test_app(user)
        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/api/browser/recordings/{rec_id}/events")

        assert resp.status_code == 200
        data = resp.json()
        assert data["event_count"] == 2
        assert data["events"] == events


class TestDeleteRecording:
    """Test DELETE /api/browser/recordings/{id}."""

    @patch("seer.api.browser.recording_router.SessionRecording")
    async def test_delete_recording(self, mock_model):
        user = _make_db_user()
        rec_id = uuid4()
        rec = _make_recording(recording_id=rec_id, user=user)
        mock_model.get_or_none = AsyncMock(return_value=rec)

        test_app = _make_test_app(user)
        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete(f"/api/browser/recordings/{rec_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] is True
        rec.delete.assert_called_once()


class TestAccessControl:
    """Test user ownership enforcement."""

    @patch("seer.api.browser.recording_router.SessionRecording")
    async def test_access_control_blocks_other_user(self, mock_model):
        user = _make_db_user()
        rec_id = uuid4()
        # get_or_none returns None when user filter doesn't match
        mock_model.get_or_none = AsyncMock(return_value=None)

        test_app = _make_test_app(user)
        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/api/browser/recordings/{rec_id}")

        assert resp.status_code == 404
