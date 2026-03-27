"""
Integration tests for browser service data layer.

Tests profile CRUD, session encryption, recording storage, and domain
extraction with real database. No browser/Playwright/CDP required.
"""
import json
from typing import Any, Dict, List
from uuid import uuid4

from cryptography.fernet import Fernet

import pytest

from seer.database.models_browser import BrowserProfile
from seer.database.models_browser_recording import SessionRecording, SessionRecordingChunk
from seer.services.browser.encryption import SessionEncryptor
from seer.services.browser.profile_manager import BrowserProfileManager
from seer.services.browser.recording_service import RecordingService
from seer.services.browser.session_context_manager import SessionContextManager


# =============================================================================
# Helpers
# =============================================================================

SAMPLE_STORAGE_STATE: Dict[str, Any] = {
    "cookies": [
        {"name": "session", "domain": ".github.com", "value": "abc123", "path": "/"},
        {"name": "auth", "domain": ".slack.com", "value": "xyz789", "path": "/"},
        {"name": "pref", "domain": "github.com", "value": "dark", "path": "/settings"},
    ],
    "origins": [
        {"origin": "https://github.com", "localStorage": [{"name": "theme", "value": "dark"}]},
    ],
}

SAMPLE_RRWEB_EVENTS: List[Dict[str, Any]] = [
    {"type": 0, "data": {}, "timestamp": 1000},
    {"type": 2, "data": {"node": {"type": 0}}, "timestamp": 1001},
    {"type": 3, "data": {"source": 1, "positions": [{"x": 100, "y": 200}]}, "timestamp": 1050},
    {"type": 3, "data": {"source": 5, "text": "hello"}, "timestamp": 1100},
]


# =============================================================================
# Profile CRUD (Real DB)
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestBrowserProfileCRUD:
    """Profile create, list, get, soft-delete with real database."""

    async def test_create_profile(self, db_engine, test_user):
        """Create a new browser profile."""
        mgr = BrowserProfileManager()
        profile = await mgr.create_profile(test_user, "Work Profile")

        assert profile.name == "Work Profile"
        assert profile.user_id == test_user.id
        assert profile.status == "active"
        assert profile.id is not None

    async def test_list_profiles_returns_active_only(self, db_engine, test_user):
        """list_profiles excludes soft-deleted profiles."""
        mgr = BrowserProfileManager()
        await mgr.create_profile(test_user, "Active Profile")
        deleted = await mgr.create_profile(test_user, "Deleted Profile")
        await BrowserProfile.filter(id=deleted.id).update(status="deleted")

        profiles = await mgr.list_profiles(test_user)

        names = {p["name"] for p in profiles}
        assert "Active Profile" in names
        assert "Deleted Profile" not in names

    async def test_list_profiles_metadata_format(self, db_engine, test_user):
        """list_profiles returns correctly shaped metadata dicts."""
        mgr = BrowserProfileManager()
        await mgr.create_profile(test_user, "Format Test")

        profiles = await mgr.list_profiles(test_user)
        p = next(x for x in profiles if x["name"] == "Format Test")

        assert "id" in p
        assert "name" in p
        assert "logged_in_domains" in p
        assert isinstance(p["logged_in_domains"], list)
        assert "created_at" in p

    async def test_get_profile_by_id(self, db_engine, test_user):
        """get_profile returns the profile matching user + id."""
        mgr = BrowserProfileManager()
        created = await mgr.create_profile(test_user, "Fetch Test")

        fetched = await mgr.get_profile(test_user, created.id)
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.name == "Fetch Test"

    async def test_get_profile_returns_none_for_deleted(self, db_engine, test_user):
        """get_profile returns None for soft-deleted profiles."""
        mgr = BrowserProfileManager()
        profile = await mgr.create_profile(test_user, "Soon Deleted")
        await mgr.delete_profile(test_user, profile.id)

        result = await mgr.get_profile(test_user, profile.id)
        assert result is None

    async def test_delete_profile_soft_deletes(self, db_engine, test_user):
        """delete_profile sets status='deleted', does not remove from DB."""
        mgr = BrowserProfileManager()
        profile = await mgr.create_profile(test_user, "Soft Delete")

        deleted = await mgr.delete_profile(test_user, profile.id)
        assert deleted is True

        # Still exists in DB with deleted status
        raw = await BrowserProfile.get(id=profile.id)
        assert raw.status == "deleted"

    async def test_delete_nonexistent_returns_false(self, db_engine, test_user):
        """delete_profile returns False for unknown profile."""
        mgr = BrowserProfileManager()
        result = await mgr.delete_profile(test_user, uuid4())
        assert result is False

    async def test_profiles_isolated_by_user(self, db_engine, test_user):
        """User can only see their own profiles."""
        from seer.database import User

        user2 = await User.create(
            user_id="browser_user_2",
            email="browser2@example.com",
            first_name="Other",
            last_name="User",
        )

        mgr = BrowserProfileManager()
        await mgr.create_profile(test_user, "User1 Profile")
        await mgr.create_profile(user2, "User2 Profile")

        user1_profiles = await mgr.list_profiles(test_user)
        user2_profiles = await mgr.list_profiles(user2)

        assert all(p["name"] != "User2 Profile" for p in user1_profiles)
        assert all(p["name"] != "User1 Profile" for p in user2_profiles)


# =============================================================================
# Session Encryption (Fernet + Legacy Fallback)
# =============================================================================


@pytest.mark.integration
class TestSessionEncryption:
    """Encrypt/decrypt roundtrip and backward-compatible JSON fallback."""

    def test_encrypt_decrypt_roundtrip(self):
        """Encrypted data decrypts to the original dict."""
        encryptor = SessionEncryptor(key=Fernet.generate_key())
        original = SAMPLE_STORAGE_STATE

        encrypted = encryptor.encrypt(original)
        decrypted = encryptor.decrypt(encrypted)

        assert decrypted == original
        assert encrypted != json.dumps(original)  # Not plaintext

    def test_decrypt_legacy_plain_json(self):
        """Legacy unencrypted JSON is handled via fallback."""
        encryptor = SessionEncryptor(key=Fernet.generate_key())
        plain_json = json.dumps({"cookies": [{"domain": "test.com"}]})

        result = encryptor.decrypt(plain_json)
        assert result == {"cookies": [{"domain": "test.com"}]}

    def test_decrypt_garbage_returns_none(self):
        """Corrupted data returns None (doesn't crash)."""
        encryptor = SessionEncryptor(key=Fernet.generate_key())
        result = encryptor.decrypt("not-valid-anything-at-all{{{")
        assert result is None

    def test_different_keys_cannot_decrypt(self):
        """Data encrypted with one key cannot be decrypted with another."""
        key1 = Fernet.generate_key()
        key2 = Fernet.generate_key()

        encrypted = SessionEncryptor(key=key1).encrypt({"secret": True})
        result = SessionEncryptor(key=key2).decrypt(encrypted)
        # Should fall through to plain JSON fallback, which fails on Fernet token
        assert result is None


# =============================================================================
# Session Context Manager (DB + Encryption Integration)
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestSessionContextManager:
    """Session state persistence: encrypt → DB → load → decrypt."""

    async def test_save_and_load_roundtrip(self, db_engine, test_user):
        """Save session state, then load it back — full roundtrip."""
        mgr = BrowserProfileManager()
        profile = await mgr.create_profile(test_user, "Roundtrip Test")

        ctx = SessionContextManager()
        saved = await ctx.save_session_state(test_user, profile.id, SAMPLE_STORAGE_STATE)
        assert saved is True

        loaded = await ctx.load_session_state(test_user, profile.id)
        assert loaded is not None
        assert loaded["cookies"] == SAMPLE_STORAGE_STATE["cookies"]
        assert loaded["origins"] == SAMPLE_STORAGE_STATE["origins"]

    async def test_save_updates_logged_in_domains(self, db_engine, test_user):
        """Saving session state extracts and persists domains."""
        mgr = BrowserProfileManager()
        profile = await mgr.create_profile(test_user, "Domains Test")

        ctx = SessionContextManager()
        await ctx.save_session_state(test_user, profile.id, SAMPLE_STORAGE_STATE)

        await profile.refresh_from_db()
        assert "github.com" in profile.logged_in_domains
        assert "slack.com" in profile.logged_in_domains

    async def test_save_updates_last_used_at(self, db_engine, test_user):
        """Saving session state updates last_used_at timestamp."""
        mgr = BrowserProfileManager()
        profile = await mgr.create_profile(test_user, "Timestamp Test")
        assert profile.last_used_at is None

        ctx = SessionContextManager()
        await ctx.save_session_state(test_user, profile.id, SAMPLE_STORAGE_STATE)

        await profile.refresh_from_db()
        assert profile.last_used_at is not None

    async def test_load_nonexistent_returns_none(self, db_engine, test_user):
        """Loading from a nonexistent profile returns None."""
        ctx = SessionContextManager()
        result = await ctx.load_session_state(test_user, uuid4())
        assert result is None

    async def test_load_empty_profile_returns_none(self, db_engine, test_user):
        """Loading from a profile with no saved state returns None."""
        mgr = BrowserProfileManager()
        profile = await mgr.create_profile(test_user, "Empty Profile")

        ctx = SessionContextManager()
        result = await ctx.load_session_state(test_user, profile.id)
        assert result is None

    async def test_save_to_deleted_profile_returns_false(self, db_engine, test_user):
        """Saving to a deleted profile fails gracefully."""
        mgr = BrowserProfileManager()
        profile = await mgr.create_profile(test_user, "Deleted Save")
        await mgr.delete_profile(test_user, profile.id)

        ctx = SessionContextManager()
        result = await ctx.save_session_state(test_user, profile.id, SAMPLE_STORAGE_STATE)
        assert result is False


# =============================================================================
# Domain Extraction (Pure Logic)
# =============================================================================


@pytest.mark.integration
class TestDomainExtraction:
    """SessionContextManager.extract_domains — pure logic, no DB."""

    def test_extracts_unique_domains(self):
        domains = SessionContextManager.extract_domains(SAMPLE_STORAGE_STATE)
        assert "github.com" in domains
        assert "slack.com" in domains

    def test_strips_leading_dot(self):
        """Cookie domains like '.github.com' become 'github.com'."""
        state = {"cookies": [{"domain": ".example.com"}]}
        domains = SessionContextManager.extract_domains(state)
        assert domains == ["example.com"]

    def test_deduplicates_with_and_without_dot(self):
        """'.github.com' and 'github.com' resolve to the same domain."""
        state = {
            "cookies": [
                {"domain": ".github.com"},
                {"domain": "github.com"},
            ]
        }
        domains = SessionContextManager.extract_domains(state)
        assert domains == ["github.com"]

    def test_empty_cookies_returns_empty(self):
        domains = SessionContextManager.extract_domains({"cookies": []})
        assert domains == []

    def test_returns_sorted(self):
        state = {
            "cookies": [
                {"domain": "z.com"},
                {"domain": "a.com"},
                {"domain": "m.com"},
            ]
        }
        domains = SessionContextManager.extract_domains(state)
        assert domains == ["a.com", "m.com", "z.com"]


# =============================================================================
# Session Validation (Pure Logic)
# =============================================================================


@pytest.mark.integration
class TestSessionValidation:
    """SessionContextManager.validate_session — pure logic."""

    def test_valid_session_with_cookies(self):
        assert SessionContextManager.validate_session(SAMPLE_STORAGE_STATE) is True

    def test_invalid_empty_cookies(self):
        assert SessionContextManager.validate_session({"cookies": []}) is False

    def test_invalid_none(self):
        assert SessionContextManager.validate_session(None) is False

    def test_invalid_no_cookies_key(self):
        assert SessionContextManager.validate_session({"origins": []}) is False


# =============================================================================
# Recording Storage & Retrieval (Real DB)
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestRecordingStorage:
    """Recording compress/decompress and DB retrieval (chunked + non-chunked)."""

    def test_compress_decompress_roundtrip(self):
        """Events survive gzip compression roundtrip."""
        compressed = RecordingService.compress_events(SAMPLE_RRWEB_EVENTS)
        assert isinstance(compressed, bytes)
        assert len(compressed) < len(json.dumps(SAMPLE_RRWEB_EVENTS).encode())

        decompressed = RecordingService.decompress_events(compressed)
        assert decompressed == SAMPLE_RRWEB_EVENTS

    async def test_get_recording_events_non_chunked(self, db_engine, test_user):
        """Non-chunked recording: events stored in single blob."""
        compressed = RecordingService.compress_events(SAMPLE_RRWEB_EVENTS)

        recording = await SessionRecording.create(
            user=test_user,
            session_type="workflow",
            events_compressed=compressed,
            event_count=len(SAMPLE_RRWEB_EVENTS),
            compressed_size_bytes=len(compressed),
            is_chunked=False,
            status="completed",
        )

        events = await RecordingService.get_recording_events(str(recording.id))
        assert len(events) == 4
        assert events[0]["type"] == 0
        assert events[3]["data"]["text"] == "hello"

    async def test_get_recording_events_chunked(self, db_engine, test_user):
        """Chunked recording: events reassembled from multiple chunks in order."""
        recording = await SessionRecording.create(
            user=test_user,
            session_type="workflow",
            events_compressed=None,
            is_chunked=True,
            chunk_count=3,
            status="completed",
        )

        # Create chunks out of order to verify reassembly
        chunk_data = [
            [{"type": 0, "ts": 1000}],
            [{"type": 2, "ts": 1001}, {"type": 3, "ts": 1050}],
            [{"type": 3, "ts": 1100}],
        ]

        for seq, events in enumerate(chunk_data):
            compressed = RecordingService.compress_events(events)
            await SessionRecordingChunk.create(
                recording=recording,
                sequence_number=seq,
                events_compressed=compressed,
                event_count=len(events),
                compressed_size_bytes=len(compressed),
            )

        events = await RecordingService.get_recording_events(str(recording.id))
        assert len(events) == 4
        # Verify order is correct (sequence 0, then 1, then 2)
        assert events[0]["ts"] == 1000
        assert events[1]["ts"] == 1001
        assert events[3]["ts"] == 1100

    async def test_get_recording_events_nonexistent_returns_empty(self, db_engine):
        """Nonexistent recording ID returns empty list."""
        events = await RecordingService.get_recording_events(str(uuid4()))
        assert events == []

    async def test_get_recording_events_no_data_returns_empty(self, db_engine, test_user):
        """Recording with no compressed data and no chunks returns empty."""
        recording = await SessionRecording.create(
            user=test_user,
            session_type="workflow",
            events_compressed=None,
            is_chunked=False,
            status="completed",
        )

        events = await RecordingService.get_recording_events(str(recording.id))
        assert events == []

    async def test_recording_with_workflow_run_id(self, db_engine, test_user):
        """Recordings can be queried by workflow_run_id."""
        compressed = RecordingService.compress_events(SAMPLE_RRWEB_EVENTS)
        run_id = f"run_{uuid4().hex[:8]}"

        await SessionRecording.create(
            user=test_user,
            session_type="workflow",
            workflow_run_id=run_id,
            events_compressed=compressed,
            event_count=4,
            compressed_size_bytes=len(compressed),
            status="completed",
        )

        recordings = await SessionRecording.filter(workflow_run_id=run_id).all()
        assert len(recordings) == 1
        assert recordings[0].session_type == "workflow"


# =============================================================================
# Pydantic Model Generation (Pure Logic)
# =============================================================================


@pytest.mark.integration
class TestSubmitResultModelGeneration:
    """create_submit_result_model: JSON schema → Pydantic model."""

    def test_basic_schema_generates_model(self):
        from seer.services.browser.custom_tools import create_submit_result_model

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "price": {"type": "number"},
                "in_stock": {"type": "boolean"},
            },
            "required": ["name", "price"],
        }

        Model = create_submit_result_model(schema)
        instance = Model(name="Widget", price=9.99, in_stock=True)

        assert instance.name == "Widget"
        assert instance.price == 9.99
        assert instance.in_stock is True

    def test_array_field(self):
        from seer.services.browser.custom_tools import create_submit_result_model

        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        }

        Model = create_submit_result_model(schema)
        instance = Model(tags=["a", "b", "c"])
        assert instance.tags == ["a", "b", "c"]

    def test_nested_object_field(self):
        from seer.services.browser.custom_tools import create_submit_result_model

        schema = {
            "type": "object",
            "properties": {
                "metadata": {"type": "object"},
            },
        }

        Model = create_submit_result_model(schema)
        instance = Model(metadata={"key": "value"})
        assert instance.metadata == {"key": "value"}

    def test_optional_fields_have_defaults(self):
        from seer.services.browser.custom_tools import create_submit_result_model

        schema = {
            "type": "object",
            "properties": {
                "required_field": {"type": "string"},
                "optional_field": {"type": "string"},
            },
            "required": ["required_field"],
        }

        Model = create_submit_result_model(schema)
        instance = Model(required_field="hello")
        assert instance.required_field == "hello"
        assert instance.optional_field is None
