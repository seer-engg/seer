"""
EXA enrichment for permit search results.

Takes permit agent output and enriches with:
- Company website, Crunchbase, LinkedIn
- Growth signals (funding, expansion, hiring)
- Compliance posture (LCFS, CFR, ISCC mentions)
- Contact LinkedIn profiles
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from seer.config import config
from seer.logger import get_logger

logger = get_logger(__name__)

EXA_API_URL = "https://api.exa.ai/search"


def _get_api_key() -> str:
    key = config.exa_api_key or os.environ.get("EXA_API_KEY")
    if not key:
        raise ValueError("EXA_API_KEY not found in environment or config")
    return key


async def _exa_search(query: str, num_results: int = 3) -> dict[str, Any]:
    """Make a single EXA search request."""
    headers = {
        "x-api-key": _get_api_key(),
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "numResults": num_results,
        "type": "auto",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(EXA_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()


async def _safe_exa_search(query: str, num_results: int = 3) -> list[dict[str, Any]]:
    """Search EXA with error handling. Returns list of result dicts."""
    try:
        data = await _exa_search(query, num_results)
        return data.get("results", [])
    except (httpx.HTTPError, ValueError, KeyError) as e:
        logger.warning("EXA search failed for '%s': %s", query, e)
        return []


def _fmt_results(results: list[dict[str, Any]]) -> str:
    """Format EXA results as a markdown bullet list."""
    if not results:
        return "_No results_"
    lines = []
    for r in results:
        title = r.get("title", r.get("url", ""))
        url = r.get("url", "")
        lines.append(f"- [{title}]({url})")
    return "\n".join(lines)


async def _enrich_contacts(
    contacts: list[dict[str, Any]], short_name: str
) -> None:
    """Enrich contacts with LinkedIn URLs via EXA (mutates in place)."""
    if not contacts:
        return

    searches = [
        _safe_exa_search(f"{c.get('name', '')} {short_name} LinkedIn profile", 1)
        if c.get("name") else _noop()
        for c in contacts
    ]
    contact_results = await asyncio.gather(*searches, return_exceptions=True)

    for i, cr in enumerate(contact_results):
        if not isinstance(cr, list) or not cr:
            continue
        for r in cr:
            url = r.get("url", "")
            if "linkedin.com/in/" in url:
                contacts[i]["linkedin"] = url
                break


async def _noop() -> list[dict[str, Any]]:
    """Return empty list for skipped contact searches."""
    return []


async def enrich_permit_data(permit_data: dict[str, Any]) -> dict[str, Any]:
    """
    Enrich permit extraction output with EXA web search.

    Args:
        permit_data: Output from the WARP permit search agent.
            Expected keys: company_name, permits[], contacts[]

    Returns:
        The same dict with added enrichment fields:
        - website: Best-guess company website URL
        - exa_company_results: Raw EXA results for company search
        - exa_growth_signals: Results for growth/funding/expansion
        - exa_compliance: Results for regulatory compliance mentions
        - contacts[].linkedin: LinkedIn URL for each contact
    """
    company_name = permit_data.get("company_name", "")
    if not company_name:
        logger.warning("No company_name in permit data, skipping enrichment")
        return permit_data

    short_name = " ".join(company_name.split()[:4]).rstrip(",").rstrip(".")

    # Run company-level EXA searches in parallel
    results = await asyncio.gather(
        _safe_exa_search(f"{short_name} company website"),
        _safe_exa_search(f"{short_name} RNG biogas funding expansion"),
        _safe_exa_search(f"{short_name} regulatory compliance LCFS CFR ISCC"),
        return_exceptions=True,
    )

    company_results = results[0] if not isinstance(results[0], BaseException) else []
    growth_results = results[1] if not isinstance(results[1], BaseException) else []
    compliance_results = results[2] if not isinstance(results[2], BaseException) else []

    # Best-guess website
    website = ""
    for r in company_results:
        url = r.get("url", "")
        if url and not any(s in url for s in ("linkedin.com", "crunchbase.com", "wikipedia.org")):
            website = url
            break

    # Enrich contacts with LinkedIn
    contacts = permit_data.get("contacts", [])
    if isinstance(contacts, list):
        await _enrich_contacts(contacts, short_name)

    enriched = dict(permit_data)
    enriched["website"] = website
    enriched["exa_company_results"] = _fmt_results(company_results)
    enriched["exa_growth_signals"] = _fmt_results(growth_results)
    enriched["exa_compliance"] = _fmt_results(compliance_results)

    logger.info("EXA enrichment complete for '%s': website=%s", short_name, website)
    return enriched


def enrich_permit_data_sync(permit_data: dict[str, Any]) -> dict[str, Any]:
    """Synchronous wrapper for enrich_permit_data."""
    return asyncio.run(enrich_permit_data(permit_data))
