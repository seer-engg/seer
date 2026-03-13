from seer.api.public.router import _sanitize_spec


def test_sanitize_spec_includes_triggers():
    spec = {
        "nodes": [
            {"id": "web_search-1", "type": "tool", "tool": "web_search", "ui": {"position": {"x": 100, "y": 200}}},
        ],
        "edges": [
            {"source": "trigger-1", "target": "web_search-1", "type": "trigger"},
        ],
        "triggers": [
            {
                "id": "trigger-1",
                "key": "webhook",
                "mode": "streaming",
                "event_schema": {"type": "object"},
                "meta": {"name": "Webhook Trigger"},
                "filters": {"secret": "should-be-stripped"},
                "provider_config": {"api_key": "secret"},
                "ui_meta": {"position": {"x": 0, "y": 0}, "secret_token": "s3cret", "subscription_id": 42},
            },
        ],
    }

    result = _sanitize_spec(spec, "Test", "desc", None)

    assert len(result.triggers) == 1
    t = result.triggers[0]
    assert t.id == "trigger-1"
    assert t.key == "webhook"
    assert t.mode == "streaming"
    assert t.ui_meta == {"position": {"x": 0, "y": 0}}
    # ui_meta only contains position, not secret_token/subscription_id
    assert "secret_token" not in (t.ui_meta or {})
    assert "subscription_id" not in (t.ui_meta or {})
    # Sensitive fields must NOT leak
    assert not hasattr(t, "filters")
    assert not hasattr(t, "provider_config")
    assert not hasattr(t, "event_schema")
    assert not hasattr(t, "meta")


def test_sanitize_spec_empty_triggers():
    spec = {"nodes": [], "edges": []}
    result = _sanitize_spec(spec, "Test", "desc", None)
    assert result.triggers == []


def test_sanitize_spec_nodes_and_edges_still_work():
    spec = {
        "nodes": [{"id": "n1", "type": "tool", "tool": "slack"}],
        "edges": [{"source": "n1", "target": "n2"}],
        "triggers": [],
    }
    result = _sanitize_spec(spec, "Test", "desc", "🔧")
    assert len(result.nodes) == 1
    assert result.nodes[0].label == "slack"
    assert len(result.edges) == 1
    assert result.metadata.icon == "🔧"
