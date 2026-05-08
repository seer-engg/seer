"""
Tests for permit search agent graph and nodes.

Uses mocked Tavily and BrowserService for fast, deterministic unit tests.
Patch targets are source modules (not agent.py) since nodes use lazy imports.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_core.messages import HumanMessage

from src.seer.agents.permit_search.state import PermitSearchState
from src.seer.agents.permit_search.agent import (  # pylint: disable=reimported  # Reason: Need both module and specific symbols
    EXTRACTION_SCHEMA,
    WARP_PERMIT_TRACKING,
    _build_tavily_query,
    _build_warp_search_task,
    create_permit_search_agent,
    resolve_name_node,
    search_warp_node,
)

pytestmark = pytest.mark.unit


# =============================================================================
# State factory
# =============================================================================


def make_state(company_name: str, **overrides) -> PermitSearchState:
    """Build a PermitSearchState with defaults for testing."""
    defaults: dict[str, Any] = {
        "messages": [HumanMessage(content=f"Search for {company_name}")],
        "company_name": company_name,
        "resolved_company": None,
        "permit_results": None,
        "error": None,
    }
    defaults.update(overrides)
    return PermitSearchState(**defaults)  # type: ignore[typeddict-unknown-key]


# =============================================================================
# Helper function tests
# =============================================================================


def test_build_tavily_query():
    """Tavily query targets Wisconsin DNR domains."""
    query = _build_tavily_query("Three Petals")
    assert "Three Petals" in query
    assert "site:dnr.wi.gov" in query
    assert "site:apps.dnr.wi.gov" in query


def test_build_warp_search_task():
    """WARP browser task includes URL, company name, and extraction instructions."""
    task = _build_warp_search_task("Three Petals")
    assert WARP_PERMIT_TRACKING in task
    assert "Three Petals" in task
    assert "permit ID" in task
    assert "all pages" in task


def test_extraction_schema_structure():
    """Extraction schema requires company_name and permits array."""
    assert EXTRACTION_SCHEMA["type"] == "object"
    assert "company_name" in EXTRACTION_SCHEMA["required"]
    assert "permits" in EXTRACTION_SCHEMA["required"]
    permit_schema = EXTRACTION_SCHEMA["properties"]["permits"]["items"]
    assert "permit_id" in permit_schema["properties"]
    assert "facility_name" in permit_schema["properties"]
    assert "permit_type" in permit_schema["properties"]


# =============================================================================
# resolve_name_node tests
# =============================================================================


async def test_resolve_name_already_resolved():
    """Skips Tavily call if resolved_company already set."""
    state = make_state("Three Petals", resolved_company="Three Petals LLC")
    result = await resolve_name_node(state)
    assert result == {}


async def test_resolve_name_tavily_success():
    """Resolves company name from Tavily search results."""
    mock_response = json.dumps({
        "query": "test",
        "answer": "Three Petals LLC, Novilla RNG subsidiary",
        "results": [
            {"title": "Three Petals RNG Air Permit", "url": "https://apps.dnr.wi.gov/warp_ext/", "content": "..."},
        ],
        "result_count": 1,
    })

    with patch("src.seer.agents.nexus.tools.web_search.config") as mock_config:
        mock_config.tavily_api_key = "test-key"

        with patch("tavily.TavilyClient") as mock_tavily_cls:
            mock_client = MagicMock()
            mock_client.search.return_value = json.loads(mock_response)
            mock_tavily_cls.return_value = mock_client

            state = make_state("Three Petals")
            result = await resolve_name_node(state)

            assert "resolved_company" in result
            assert "Three Petals" in result["resolved_company"]
            assert "Three Petals LLC" in result["resolved_company"]
            assert len(result["messages"]) == 1


async def test_resolve_name_tavily_error():
    """Passes original name through when Tavily returns error in response."""
    with patch("src.seer.agents.nexus.tools.web_search.config") as mock_config:
        mock_config.tavily_api_key = "test-key"

        with patch("tavily.TavilyClient") as mock_tavily_cls:
            mock_client = MagicMock()
            mock_client.search.side_effect = Exception("Connection refused")
            mock_tavily_cls.return_value = mock_client

            state = make_state("Timeout Corp")
            result = await resolve_name_node(state)

            assert result["resolved_company"] == "Timeout Corp"
            assert len(result["messages"]) == 1


async def test_resolve_name_tavily_empty_results():
    """Uses input name when Tavily returns no answer or results."""
    mock_response = json.dumps({
        "query": "test",
        "results": [],
        "result_count": 0,
    })

    with patch("src.seer.agents.nexus.tools.web_search.config") as mock_config:
        mock_config.tavily_api_key = "test-key"

        with patch("tavily.TavilyClient") as mock_tavily_cls:
            mock_client = MagicMock()
            mock_client.search.return_value = json.loads(mock_response)
            mock_tavily_cls.return_value = mock_client

            state = make_state("Unknown LLC")
            result = await resolve_name_node(state)

            assert result["resolved_company"] == "Unknown LLC"


# =============================================================================
# search_warp_node tests
# =============================================================================


def _mock_browser_result(success: bool, extracted_data: dict | None = None, error_msg: str | None = None) -> dict:
    """Build a mock BrowserService.execute_task return value."""
    if success:
        return {
            "success": True,
            "result": {"steps": [], "completed": True, "success": True},
            "extracted_data": extracted_data or {},
            "final_url": WARP_PERMIT_TRACKING,
            "urls": [WARP_PERMIT_TRACKING],
            "duration_seconds": 5.0,
            "steps_count": 3,
            "extracted_content": [],
            "model_thoughts": [],
            "model_actions": [],
            "screenshots": [],
        }
    return {
        "success": False,
        "result": {"errors": [error_msg or "Search returned no results"]},
        "extracted_data": {},
        "final_url": None,
        "urls": [],
        "duration_seconds": None,
        "steps_count": None,
        "extracted_content": [],
        "model_thoughts": [],
        "model_actions": [],
        "screenshots": [],
    }


async def test_search_warp_success():
    """Extracts permits on successful browser search."""
    import seer.services.browser.browser_service as bs_module  # pylint: disable=import-outside-toplevel  # Reason: Direct module access for singleton patching

    extracted = {
        "company_name": "Three Petals LLC",
        "permits": [
            {
                "permit_id": "34501",
                "facility_name": "Three Petals RNG",
                "permit_type": "Air Pollution Construction",
                "status": "Active",
                "issue_date": "2024-03-15",
                "expiration_date": "2029-03-15",
                "county": "Dane",
            }
        ],
        "total_permits_found": 1,
    }

    mock_execute = AsyncMock(return_value=_mock_browser_result(success=True, extracted_data=extracted))
    mock_svc = MagicMock()
    mock_svc.execute_task = mock_execute

    original = bs_module.BrowserService._instance
    bs_module.BrowserService._instance = mock_svc
    try:
        state = make_state("Three Petals", resolved_company="Three Petals LLC")
        result = await search_warp_node(state)

        assert result["permit_results"] == extracted
        assert result.get("error") is None
        assert len(result["messages"]) == 1
    finally:
        bs_module.BrowserService._instance = original


async def test_search_warp_failure():
    """Sets error when browser search fails."""
    import seer.services.browser.browser_service as bs_module  # pylint: disable=import-outside-toplevel  # Reason: Direct module access for singleton patching

    mock_execute = AsyncMock(return_value=_mock_browser_result(success=False, error_msg="No permits found"))
    mock_svc = MagicMock()
    mock_svc.execute_task = mock_execute

    original = bs_module.BrowserService._instance
    bs_module.BrowserService._instance = mock_svc
    try:
        state = make_state("Nonexistent Corp")
        result = await search_warp_node(state)

        assert "error" in result
        assert result.get("permit_results") is None
        assert len(result["messages"]) == 1
    finally:
        bs_module.BrowserService._instance = original


async def test_search_warp_exception():
    """Sets error when BrowserService raises."""
    import seer.services.browser.browser_service as bs_module  # pylint: disable=import-outside-toplevel  # Reason: Direct module access for singleton patching

    mock_execute = AsyncMock(side_effect=Exception("Browser crashed"))
    mock_svc = MagicMock()
    mock_svc.execute_task = mock_execute

    original = bs_module.BrowserService._instance
    bs_module.BrowserService._instance = mock_svc
    try:
        state = make_state("Crash Corp")
        result = await search_warp_node(state)

        assert "error" in result
        assert "Browser crashed" in result["error"]
    finally:
        bs_module.BrowserService._instance = original


async def test_search_warp_fallback_to_input_name():
    """Uses company_name when resolved_company is None."""
    import seer.services.browser.browser_service as bs_module  # pylint: disable=import-outside-toplevel  # Reason: Direct module access for singleton patching

    mock_execute = AsyncMock(return_value=_mock_browser_result(success=True, extracted_data={"permits": []}))
    mock_svc = MagicMock()
    mock_svc.execute_task = mock_execute

    original = bs_module.BrowserService._instance
    bs_module.BrowserService._instance = mock_svc
    try:
        state = make_state("Three Petals", resolved_company=None)
        await search_warp_node(state)

        call_kwargs = mock_execute.call_args.kwargs
        assert "Three Petals" in call_kwargs["task"]
    finally:
        bs_module.BrowserService._instance = original


# =============================================================================
# Graph construction tests
# =============================================================================


def test_graph_compiles():
    """Agent graph compiles without errors."""
    agent = create_permit_search_agent()
    assert agent is not None
    assert hasattr(agent, "invoke")
    assert hasattr(agent, "ainvoke")


def test_graph_node_names():
    """Graph contains expected node names."""
    agent = create_permit_search_agent()
    nodes = list(agent.get_graph().nodes.keys())
    assert "resolve_name" in nodes
    assert "search_warp" in nodes


def test_graph_edge_count():
    """Graph has 3 edges: START->resolve, resolve->search, search->END."""
    agent = create_permit_search_agent()
    edges = list(agent.get_graph().edges)
    assert len(edges) == 3


async def test_full_graph_traversal_mocked():
    """End-to-end graph traversal with mocked dependencies."""
    import seer.services.browser.browser_service as bs_module  # pylint: disable=import-outside-toplevel  # Reason: Direct module access for singleton patching

    extracted = {
        "company_name": "Three Petals",
        "permits": [{"permit_id": "34501", "facility_name": "Three Petals RNG"}],
        "total_permits_found": 1,
    }

    with patch("src.seer.agents.nexus.tools.web_search.config") as mock_config:
        mock_config.tavily_api_key = "test-key"

        with patch("tavily.TavilyClient") as mock_tavily_cls:
            mock_client = MagicMock()
            mock_client.search.return_value = {"answer": "Three Petals LLC", "results": []}
            mock_tavily_cls.return_value = mock_client

            mock_execute = AsyncMock(return_value=_mock_browser_result(success=True, extracted_data=extracted))
            mock_svc = MagicMock()
            mock_svc.execute_task = mock_execute

            original = bs_module.BrowserService._instance
            bs_module.BrowserService._instance = mock_svc
            try:
                agent = create_permit_search_agent()
                state = make_state("Three Petals")
                result = await agent.ainvoke(state)

                assert result["permit_results"] == extracted
                assert result["resolved_company"] is not None
                assert result["error"] is None
                assert len(result["messages"]) >= 2
            finally:
                bs_module.BrowserService._instance = original
