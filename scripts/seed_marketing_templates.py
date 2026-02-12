#!/usr/bin/env python3
"""Seed marketing-focused workflow templates into the database.

This script loads template JSON files from src/seer/agents/nexus/templates/
and creates WorkflowTemplate records in the database.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

from tortoise import Tortoise

from seer.database import TemplateCategory, TemplateSource, WorkflowTemplate
from seer.database.config import TORTOISE_ORM


# Template definitions with metadata
TEMPLATES_TO_SEED = [
    {
        "file": "email_inbox_triage.json",
        "slug": "email-inbox-triage",
        "category": TemplateCategory.MARKETING,
        "is_featured": True,
        "icon": "Inbox",
        "required_integrations": [
            {
                "provider": "google",
                "integration_type": "gmail",
                "reason": "Required to read and analyze unread emails",
            }
        ],
    },
    {
        "file": "linkedin_content_generator.json",
        "slug": "linkedin-content-generator",
        "category": TemplateCategory.MARKETING,
        "is_featured": True,
        "icon": "FileText",
        "required_integrations": [
            {
                "provider": "google",
                "integration_type": "google_docs",
                "reason": "Required to save generated LinkedIn posts",
            }
        ],
    },
    {
        "file": "lead_classification_routing.json",
        "slug": "lead-classification-routing",
        "category": TemplateCategory.SALES,
        "is_featured": True,
        "icon": "Users",
        "required_integrations": [
            {
                "provider": "google",
                "integration_type": "gmail",
                "reason": "Required to read leads and send alerts",
            },
            {
                "provider": "google",
                "integration_type": "google_sheets",
                "reason": "Optional: Log leads to Google Sheets",
            },
        ],
    },
    {
        "file": "email_support_chatbot.json",
        "slug": "email-support-knowledge-base",
        "category": TemplateCategory.CUSTOMER_SUPPORT,
        "is_featured": True,
        "icon": "MessageSquare",
        "required_integrations": [
            {
                "provider": "google",
                "integration_type": "gmail",
                "reason": "Required to read and respond to support emails",
            },
            {
                "provider": "google",
                "integration_type": "google_docs",
                "reason": "Required to access knowledge base documentation",
            },
            {
                "provider": "supabase",
                "integration_type": "supabase",
                "reason": "Optional: Log conversations and search knowledge base",
            },
        ],
    },
]


def load_template_json(template_dir: Path, filename: str) -> Dict[str, Any]:
    """Load template JSON file."""
    file_path = template_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Template file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


async def seed_template(
    template_dir: Path,
    template_def: Dict[str, Any],
    update_existing: bool = False,
) -> None:
    """Seed a single template into the database."""
    # Load template JSON
    template_json = load_template_json(template_dir, template_def["file"])

    slug = template_def["slug"]

    # Check if template already exists
    existing = await WorkflowTemplate.filter(slug=slug).first()

    if existing:
        if update_existing:
            print(f"Updating existing template: {slug}")
            await WorkflowTemplate.filter(id=existing.id).update(
                name=template_json["name"],
                description=template_json["description"],
                category=template_def["category"],
                tags=template_json.get("tags", []),
                spec=template_json["spec"],
                required_integrations=template_def["required_integrations"],
                icon=template_def.get("icon"),
                is_featured=template_def.get("is_featured", False),
                is_published=True,
            )
            print(f"✓ Updated template: {slug}")
        else:
            print(f"⊘ Template already exists (skipping): {slug}")
        return

    # Create new template
    template = await WorkflowTemplate.create(
        slug=slug,
        name=template_json["name"],
        description=template_json["description"],
        category=template_def["category"],
        tags=template_json.get("tags", []),
        source=TemplateSource.SYSTEM,
        spec=template_json["spec"],
        required_integrations=template_def["required_integrations"],
        icon=template_def.get("icon"),
        preview_image_url=template_def.get("preview_image_url"),
        is_published=True,
        is_featured=template_def.get("is_featured", False),
        usage_count=0,
        created_by=None,  # System template
    )

    print(f"✓ Created template: {template.slug} (id={template.id})")


async def seed_all_templates(update_existing: bool = False) -> None:
    """Seed all marketing templates into the database."""
    # Initialize Tortoise ORM
    await Tortoise.init(config=TORTOISE_ORM)

    # Get template directory
    template_dir = Path(__file__).parent.parent / "src" / "seer" / "agents" / "nexus" / "templates"
    if not template_dir.exists():
        raise FileNotFoundError(f"Template directory not found: {template_dir}")

    print(f"Seeding {len(TEMPLATES_TO_SEED)} marketing templates from {template_dir}")
    print(f"Update existing: {update_existing}")
    print()

    try:
        for template_def in TEMPLATES_TO_SEED:
            try:
                await seed_template(template_dir, template_def, update_existing)
            except Exception as e:  # pylint: disable=broad-exception-caught  # Continue seeding remaining templates on individual failures
                print(f"✗ Error seeding {template_def['file']}: {e}")
                continue

        print()
        total = await WorkflowTemplate.all().count()
        featured = await WorkflowTemplate.filter(is_featured=True).count()
        print(f"Database now has {total} total templates ({featured} featured)")

    finally:
        await Tortoise.close_connections()


async def main() -> None:
    """Main entry point."""
    update_existing = "--update" in sys.argv or "-u" in sys.argv

    try:
        await seed_all_templates(update_existing=update_existing)
    except Exception as e:  # pylint: disable=broad-exception-caught  # Top-level error handler for script execution
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
