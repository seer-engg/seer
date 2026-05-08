"""
LangGraph agent for Wisconsin DNR WARP air permit search.

Combines Tavily web search for company name resolution with
browser-use automation to search the WARP air permit tracking system.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from seer.agents.permit_search.state import PermitSearchState
from seer.logger import get_logger

logger = get_logger(__name__)

WARP_BASE_URL = "https://apps.dnr.wi.gov/warp_ext/"
WARP_PERMIT_TRACKING = f"{WARP_BASE_URL}AM_PermitTracking2.aspx"

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "company_name": {"type": "string"},
        "fid": {"type": "string", "description": "Facility ID from WARP"},
        "location": {"type": "string", "description": "Full address: city, state, zip"},
        "county": {"type": "string"},
        "state": {"type": "string"},
        "naics": {"type": "string", "description": "NAICS code and description"},
        "sic": {"type": "string", "description": "SIC code and description"},
        "warp_url": {"type": "string", "description": "Current WARP detail page URL"},
        "contacts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "title": {"type": "string"},
                    "email": {"type": "string"},
                    "phone": {"type": "string"},
                    "role": {"type": "string"},
                },
            },
        },
        "documents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Document filename as shown on WARP"},
                },
            },
        },
        "permits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "permit_id": {"type": "string"},
                    "facility_name": {"type": "string"},
                    "permit_type": {"type": "string"},
                    "status": {"type": "string"},
                    "issue_date": {"type": "string"},
                    "expiration_date": {"type": "string"},
                    "county": {"type": "string"},
                },
                "required": ["permit_id"],
            },
        },
        "total_permits_found": {"type": "integer"},
    },
    "required": ["company_name", "permits"],
}


def _build_tavily_query(company_name: str) -> str:
    """Build a Tavily search query to resolve company legal name."""
    return f'"{company_name}" company legal name site:dnr.wi.gov OR site:apps.dnr.wi.gov'


def _build_warp_search_task(company_name: str) -> str:
    """Build the browser task prompt for WARP permit search."""
    return (
        f"Go to {WARP_PERMIT_TRACKING}. "
        f"If the page shows a server error, instead go to "
        f"https://dnr.wisconsin.gov/topic/AirPermits/Search.html, "
        f"click 'Air Permit Search Tool', then in the search form "
        f"find the 'Facility Name' field and enter '{company_name}'. "
        f"Make sure the search type dropdown (if present) is set to "
        f"'Contains' or 'Starts With' to do a partial match. "
        f"Click Search. "
        f"On the search results, click on the matching facility to open its detail page. "
        f"On the detail page, go to 'General Facility Information' tab and extract: "
        f"Facility ID (FID), full location address, county, NAICS code, SIC code. "
        f"Then go to 'Permits and Permit Applications' tab and extract ALL permits with: "
        f"permit ID, facility name, permit type, status, issue date, expiration date, and county. "
        f"Check 'Show Inactive Permits' if available. "
        f"Click 'Select' on each permit to check for document names and any dates. "
        f"Then go to 'Facility Contacts' or look for contact information and extract: "
        f"name, title, email, phone, and role for ALL contacts listed. "
        f"Also note any document filenames shown on permits pages. "
        f"If 'No Data Found', try shortening the name (e.g., just 'C-Dairy' or 'Brightmark'). "
        f"If multiple pages of results exist, navigate through all pages. "
        f"Return the complete list of permits, contacts, and facility details found. "
        f"The current page URL is the warp_url."
    )


async def resolve_name_node(state: PermitSearchState) -> dict[str, Any]:
    """Resolve company name via Tavily web search.

    Uses the existing web_search tool to find the company's legal name
    on Wisconsin DNR sites. If search fails, passes the original name through.
    """
    if state.get("resolved_company"):
        logger.debug("Company already resolved: %s", state["resolved_company"])
        return {}

    from seer.agents.nexus.tools.web_search import web_search  # pylint: disable=import-outside-toplevel  # Reason: Avoid startup dependency

    company_name = state["company_name"]
    query = _build_tavily_query(company_name)

    try:
        result_json = await web_search.ainvoke({
            "query": query,
            "max_results": 3,
            "search_depth": "advanced",
        })
        result: dict[str, Any] = json.loads(result_json)
    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Pass name through on search failure
        logger.warning("Tavily search failed for '%s': %s", company_name, e)
        return {
            "resolved_company": company_name,
            "messages": [AIMessage(content=f"Search failed, using input name: {company_name}")],
        }

    if result.get("error"):
        logger.warning("Tavily search error for '%s': %s", company_name, result["error"])
        return {
            "resolved_company": company_name,
            "messages": [AIMessage(content=f"Search error, using input name: {company_name}")],
        }

    answer = result.get("answer", "")
    results = result.get("results", [])
    resolved = company_name

    if answer:
        resolved = f"{company_name} ({answer[:200]})"
    elif results:
        titles = [r.get("title", "") for r in results[:2]]
        resolved = f"{company_name} (see: {'; '.join(titles)})"

    logger.info("Resolved company '%s' -> '%s'", company_name, resolved)
    msg = AIMessage(content=f"Resolved: {resolved}\nSearch answer: {answer}")
    return {"resolved_company": resolved, "messages": [msg]}


async def search_warp_node(state: PermitSearchState) -> dict[str, Any]:
    """Search Wisconsin DNR WARP for air permits using browser automation.

    Uses the existing BrowserService (Playwright + browser-use) to navigate
    the WARP permit tracking UI, search by company name, and extract
    structured permit data via the extraction schema.
    """
    from seer.services.browser import BrowserService  # pylint: disable=import-outside-toplevel  # Reason: Avoid startup dependency

    resolved = state.get("resolved_company") or state["company_name"]
    task = _build_warp_search_task(resolved)
    model = "deepseek-v4-flash"

    logger.info("Searching WARP for: %s (model=%s)", resolved, model)

    try:
        result = await BrowserService.instance().execute_task(
            user=None,
            task=task,
            inputs={"url": WARP_PERMIT_TRACKING, "company_name": resolved},
            max_steps=35,
            timeout_seconds=900,
            extraction_schema=EXTRACTION_SCHEMA,
            model=model,
        )

        if result.get("success") and result.get("extracted_data"):
            extracted = result["extracted_data"]
            count = extracted.get("total_permits_found", len(extracted.get("permits", [])))
            logger.info("WARP search succeeded: %s permits found", count)
            msg = AIMessage(content=f"Found {count} permits for {resolved}")
            return {"permit_results": extracted, "messages": [msg]}

        error_detail = result.get("result", "Unknown error")
        if isinstance(error_detail, dict):
            errors = error_detail.get("errors", [])
            error_msg = errors[0] if errors else "Unknown error"
        else:
            error_msg = str(error_detail)
        logger.warning("WARP search failed: %s", error_msg)
        return {"error": error_msg, "messages": [AIMessage(content=f"WARP search failed: {error_msg}")]}

    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Graceful error handling for browser failures
        logger.exception("WARP search exception: %s", e)
        return {"error": str(e), "messages": [AIMessage(content=f"WARP search exception: {e}")]}


def create_permit_search_agent(
    checkpointer: Optional[Any] = None,
) -> Any:
    """Create a LangGraph agent for WARP air permit search.

    Builds a 2-node StateGraph:
    1. resolve_name_node - Resolves company legal name via Tavily
    2. search_warp_node - Searches WARP via browser automation

    The browser node uses openai/gpt-4o for reliable web navigation.
    Args:
        checkpointer: Optional LangGraph checkpointer for persistence.

    Returns:
        Compiled LangGraph StateGraph.
    """
    graph = StateGraph(PermitSearchState)

    graph.add_node("resolve_name", resolve_name_node)
    graph.add_node("search_warp", search_warp_node)

    graph.add_edge(START, "resolve_name")
    graph.add_edge("resolve_name", "search_warp")
    graph.add_edge("search_warp", END)

    compiled = graph.compile(checkpointer=checkpointer)

    logger.info("Created permit search agent (linear pipeline: resolve_name -> search_warp)")
    return compiled
