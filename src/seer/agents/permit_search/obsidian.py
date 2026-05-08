"""
Obsidian vault export for permit search results.

Writes enriched permit data to the Seer Shared Sales Obsidian vault:
- Sales Funnel/Accounts/{Company Name}.md
- Sales Funnel/Leads/{Contact Name}.md
- Generates [[wikilinks]] between company and contact notes.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from seer.logger import get_logger

logger = get_logger(__name__)

# Google Drive shared vault path
VAULT_PATH = Path(
    os.path.expanduser(
        "~/Library/CloudStorage/GoogleDrive-akshay@getseer.dev/Shared drives/Seer Shared/Sales"
    )
)
ACCOUNTS_DIR = VAULT_PATH / "Sales Funnel" / "Accounts"
LEADS_DIR = VAULT_PATH / "Sales Funnel" / "Leads"
DOCUMENTS_DIR = VAULT_PATH / "Regulations" / "US Air Permitting" / "Wisconsin"


def _slugify(name: str) -> str:
    """Convert a name to a filesystem-safe slug."""
    return re.sub(r"[^\w\- ]", "", name).strip()


def _format_date(date_str: str | None) -> str:
    """Normalize date strings."""
    if not date_str or date_str.lower() in ("n/a", "none", ""):
        return ""
    return date_str.strip()


def _escape_yaml(value: str) -> str:
    """Escape a string for YAML frontmatter if it contains special chars."""
    if any(c in value for c in ('"', ":", "{", "}", "[", "]", "#", "&", "*", "!", ">", "|", "@", "`")):
        return f'"{value}"'
    return value


def _build_permit_fm(permit: dict[str, Any]) -> list[str]:
    """Build YAML lines for a single permit in frontmatter."""
    lines = [f'  - id: "{permit.get("permit_id", "")}"']
    for field in ["type", "status", "issue_date", "expiration_date"]:
        fval = _format_date(permit.get(field)) if field in ("issue_date", "expiration_date") else permit.get(field, "")
        lines.append(f'    {field}: "{fval}"')
    return lines


def _build_contact_fm(contact: dict[str, Any]) -> list[str]:
    """Build YAML lines for a single contact in frontmatter."""
    lines = [f'  - name: "{contact.get("name", "")}"']
    for field in ["title", "email", "phone", "role"]:
        lines.append(f'    {field}: "{contact.get(field, "")}"')
    linkedin = contact.get("linkedin", "")
    if linkedin:
        lines.append(f'    linkedin: "{linkedin}"')
    return lines


def _build_frontmatter(data: dict[str, Any], today: str) -> tuple[list[str], str, dict[str, Any]]:
    """Build YAML frontmatter lines for company note."""
    company_name = data.get("company_name", "Unknown").split("(")[0].strip()
    # pylint: disable=too-many-branches
    lines = ["---"]
    lines.append(f'company_name: "{company_name}"')
    for key, yaml_key in [("fid", "fid"), ("location", "location"), ("county", "county"),
                          ("state", "state"), ("naics", "naics"), ("sic", "sic")]:
        val = data.get(key, "")
        if val:
            lines.append(f'{yaml_key}: "{val}"')
    lines.append("rng_type: ")
    lines.append("sales_stage: prospecting")
    lines.append("icp_score: ")
    lines.append(f'last_scraped: "{today}"')
    for key in ("warp_url", "website"):
        val = data.get(key, "")
        if val:
            lines.append(f'{key}: "{val}"')
    for key, builder in _FM_BUILDERS:
        items = data.get(key, [])
        if items:
            lines.append(f"{key}:")
            for item in items:
                lines.extend(builder(item))
    lines.append("---")
    lines.append("")
    return lines, company_name, data


_FM_BUILDERS = [
    ("permits", _build_permit_fm),
    ("contacts", _build_contact_fm),
    ("documents", lambda d: [f'  - "{d.get("name", str(d))}"']),
]


def _build_company_body(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    lines: list[str], company_name: str, data: dict[str, Any]
) -> None:
    """Append markdown body sections for company note."""
    fid = data.get("fid", "")
    location = data.get("location", "")
    county = data.get("county", "")
    naics = data.get("naics", "")
    sic = data.get("sic", "")
    warp_url = data.get("warp_url", "")
    website = data.get("website", "")
    permits = data.get("permits", [])
    contacts = data.get("contacts", [])
    documents = data.get("documents", [])
    exa_summary = data.get("exa_company_results", "")
    growth = data.get("exa_growth_signals", "")
    compliance = data.get("exa_compliance", "")

    lines.append(f"# {company_name}")
    lines.append("")

    if fid or location:
        parts = [f"**FID:** {fid}"] if fid else []
        if location:
            parts.append(f"**Location:** {location}")
        if county:
            parts.append(f"**County:** {county}")
        lines.append(" | ".join(parts))
        lines.append("")

    if naics or sic:
        parts = [f"**NAICS:** {naics}"] if naics else []
        if sic:
            parts.append(f"**SIC:** {sic}")
        lines.append(" | ".join(parts))
        lines.append("")

    if warp_url:
        lines.append(f"**WARP:** [Permit Tracking]({warp_url})")
    if website:
        lines.append(f"**Website:** {website}")
    if warp_url or website:
        lines.append("")

    _build_permits_table(lines, permits)
    _build_contacts_table(lines, contacts)
    _build_documents_list(lines, documents)
    _build_exa_section(lines, exa_summary, growth, compliance)
    lines.append("## Notes")
    lines.append("")
    lines.append("-")


def _build_permits_table(lines: list[str], permits: list[dict[str, Any]]) -> None:
    """Append permits table to lines."""
    if not permits:
        return
    lines.append("## Air Permits")
    lines.append("")
    lines.append("| Permit # | Type | Status | Issue Date | Expiration |")
    lines.append("|----------|------|--------|------------|------------|")
    for p in permits:
        pid = p.get("permit_id", "")
        ptype = p.get("permit_type", "")
        pstatus = p.get("status", "")
        pissued = _format_date(p.get("issue_date"))
        pexpires = _format_date(p.get("expiration_date"))
        lines.append(f"| {pid} | {ptype} | {pstatus} | {pissued} | {pexpires} |")
    lines.append("")


def _build_contacts_table(lines: list[str], contacts: list[dict[str, Any]]) -> None:
    """Append contacts table to lines."""
    if not contacts:
        return
    lines.append("## Contacts")
    lines.append("")
    lines.append("| Name | Title | Email | Phone | Role |")
    lines.append("|------|-------|-------|-------|------|")
    for c in contacts:
        name = c.get("name", "")
        title = c.get("title", "")
        email = c.get("email", "")
        phone = c.get("phone", "")
        role = c.get("role", "")
        lines.append(f"| [[{_slugify(name)}]] | {title} | {email} | {phone} | {role} |")
    lines.append("")


def _build_documents_list(lines: list[str], documents: list[dict[str, Any]]) -> None:
    """Append documents section to lines."""
    if not documents:
        return
    lines.append("## Documents")
    lines.append("")
    for d in documents:
        lines.append(f"- {d.get('name', str(d))}")
    lines.append("")


def _build_exa_section(
    lines: list[str], company: str, growth: str, compliance: str
) -> None:
    """Append EXA research section to lines."""
    if not any([company, growth, compliance]):
        return
    lines.append("## EXA Research")
    lines.append("")
    for title, content in [("Company", company), ("Growth Signals", growth),
                           ("Compliance Posture", compliance)]:
        if content:
            lines.append(f"### {title}")
            lines.append(content)
            lines.append("")


def _build_company_markdown(data: dict[str, Any]) -> str:  # pylint: disable=too-many-locals
    """Build a company account note from enriched permit data."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines, company_name, data = _build_frontmatter(data, today)
    _build_company_body(lines, company_name, data)
    return "\n".join(lines)


def _build_contact_markdown(
    contact: dict[str, Any], company_name: str
) -> str:
    """Build a contact/lead note from enriched contact data."""
    name = contact.get("name", "Unknown")
    title = contact.get("title", "")
    email = contact.get("email", "")
    phone = contact.get("phone", "")
    role = contact.get("role", "")
    linkedin = contact.get("linkedin", "")

    lines = ["---"]
    lines.append(f'name: "{name}"')
    lines.append(f'company: "[[{company_name}]]"')
    lines.append(f'title: "{title}"')
    lines.append(f'email: "{email}"')
    lines.append(f'phone: "{phone}"')
    lines.append(f'role: "{role}"')
    if linkedin:
        lines.append(f'linkedin: "{linkedin}"')
    lines.append("---")
    lines.append("")
    lines.append(f"# {name}")
    lines.append("")
    lines.append(f"**Company:** [[{company_name}]] | **Title:** {title} | **Role:** {role}")
    lines.append("")
    lines.append(f"**Email:** {email} | **Phone:** {phone}")
    lines.append("")
    if linkedin:
        lines.append(f"**LinkedIn:** {linkedin}")
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("-")

    return "\n".join(lines)


def _update_existing_note(path: Path, new_content: str) -> bool:
    """Merge new content into existing note. Returns True if updated."""
    if not path.exists():
        return False

    existing = path.read_text()
    # Simple strategy: if existing note has frontmatter, preserve body content
    # but update frontmatter with new fields
    if existing.startswith("---"):
        # Find end of existing frontmatter
        end_idx = existing.find("---", 3)
        if end_idx > 0:
            existing_body = existing[end_idx + 3:]
            # Extract frontmatter from new content
            new_fm_end = new_content.find("---", 3)
            if new_fm_end > 0:
                new_fm = new_content[3:new_fm_end]
                # Merge: use new frontmatter, keep old body below new body
                merged = f"---{new_fm}---\n{new_content[new_fm_end+3:]}\n\n## Previous Notes\n{existing_body}"
                path.write_text(merged)
                return True

    # Fallback: just prepend new content
    path.write_text(new_content)
    return True


def export_to_obsidian(enriched_data: dict[str, Any], dry_run: bool = False) -> dict[str, list[str]]:
    """
    Export enriched permit data to Obsidian vault.

    Args:
        enriched_data: Output from enrich_permit_data (or raw permit data)
        dry_run: If True, don't write files, just return what would be written

    Returns:
        Dict with 'created' and 'updated' lists of file paths.
    """
    company_name = enriched_data.get("company_name", "Unknown").split("(")[0].strip()
    contacts = enriched_data.get("contacts", [])
    result = {"created": [], "updated": []}

    # Ensure directories exist
    if not dry_run:
        ACCOUNTS_DIR.mkdir(parents=True, exist_ok=True)
        LEADS_DIR.mkdir(parents=True, exist_ok=True)

    # Write company note
    company_slug = _slugify(company_name)
    company_path = ACCOUNTS_DIR / f"{company_slug}.md"
    company_md = _build_company_markdown(enriched_data)

    if not dry_run:
        if company_path.exists():
            _update_existing_note(company_path, company_md)
            result["updated"].append(str(company_path))
            logger.info("Updated company note: %s", company_path)
        else:
            company_path.write_text(company_md)
            result["created"].append(str(company_path))
            logger.info("Created company note: %s", company_path)
    else:
        action = "UPDATE" if company_path.exists() else "CREATE"
        result["created"].append(f"[DRY RUN] {action} {company_path}")

    # Write contact notes
    for contact in contacts:
        name = contact.get("name", "")
        if not name:
            continue
        contact_slug = _slugify(name)
        contact_path = LEADS_DIR / f"{contact_slug}.md"
        contact_md = _build_contact_markdown(contact, company_name)

        if not dry_run:
            if contact_path.exists():
                _update_existing_note(contact_path, contact_md)
                result["updated"].append(str(contact_path))
                logger.info("Updated contact note: %s", contact_path)
            else:
                contact_path.write_text(contact_md)
                result["created"].append(str(contact_path))
                logger.info("Created contact note: %s", contact_path)
        else:
            action = "UPDATE" if contact_path.exists() else "CREATE"
            result["created"].append(f"[DRY RUN] {action} {contact_path}")

    return result


def export_json_to_obsidian(json_str: str, dry_run: bool = False) -> dict[str, list[str]]:
    """Parse JSON and export to Obsidian."""
    data = json.loads(json_str)
    return export_to_obsidian(data, dry_run=dry_run)
