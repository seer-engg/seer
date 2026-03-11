"""Insert Frederica's avatar + 3 workflow templates into prod."""

import json
import os

import psycopg2

DATABASE_URL = os.environ["DATABASE_URL"]

AVATAR_URL = (
    "https://media.licdn.com/dms/image/v2/D560BAQGEX5EPq27V9w/"
    "company-logo_200_200/B56ZwEVzKdHcAI-/0/1769599350026"
    "?e=1774483200&v=beta&t=S6SfTzgODknv9DbPHJJUmkPMmMoGar9-NyZ4j9ionZA"
)

INSERT_SQL = """
    INSERT INTO workflow_templates (
        slug, name, description, category, tags, source,
        created_by_id, spec, required_integrations, is_published,
        created_at, updated_at
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
    ON CONFLICT (slug) DO UPDATE SET
        name = EXCLUDED.name, description = EXCLUDED.description,
        spec = EXCLUDED.spec,
        required_integrations = EXCLUDED.required_integrations,
        tags = EXCLUDED.tags,
        is_published = EXCLUDED.is_published, updated_at = NOW()
"""

DESC_A = (
    "Start every morning with purpose. This workflow automatically "
    "generates a fresh, AI-crafted motivational message and delivers "
    "it straight to your inbox on a daily schedule. Each email "
    "includes an inspiring quote and an actionable personal "
    "development tip to set the tone for your day."
)

PROMPT_A = (
    "Generate a short, uplifting motivational message for the day. "
    "Include an inspiring quote and a brief actionable tip for "
    "personal growth."
)

DESC_B = (
    "Automate your document review pipeline. This workflow receives "
    "application documents (PDFs) via webhook, extracts their "
    "contents, and uses AI to analyze completeness, extract key "
    "fields, and generate a structured compliance report. Results "
    "are emailed directly to your review team."
)

PROMPT_B = (
    "Analyze this application document. Extract key fields (name, "
    "date, role applied for, qualifications). Check for completeness "
    "and flag any missing required sections. Provide a structured "
    "summary with a compliance score.\n\nDocument content:\n"
    "${nodes.extract_pdf.output}"
)

DESC_C = (
    "Turn Google Docs into high-performing LinkedIn content "
    "automatically. This workflow monitors a Google Drive folder for "
    "new documents, uses AI with the SNEAC framework (Story, Nugget, "
    "Empathy, Actionable, CTA) and GEO optimization to craft "
    "engaging LinkedIn posts, sends them for email approval, and "
    "publishes directly to your LinkedIn profile."
)

PROMPT_C = (
    "You are a LinkedIn content strategist using the SNEAC framework "
    "(Story, Nugget, Empathy, Actionable, CTA) and GEO (Generative "
    "Engine Optimization) principles. Transform the following "
    "document into a compelling LinkedIn post. Keep it professional, "
    "engaging, and optimized for LinkedIn's algorithm.\n\n"
    "Source document:\n${nodes.read_doc.output}"
)

APPROVAL_BODY = (
    "A new LinkedIn post has been drafted from your Google Doc. "
    "Please review:\n\n${nodes.sneac_agent.output}\n\n"
    "Reply APPROVE to publish."
)


def main():  # pylint: disable=too-many-statements  # one-off data script
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        # 1. Update avatar
        cur.execute(
            "UPDATE user_profiles SET avatar_url = %s, "
            "updated_at = NOW() WHERE user_id = 81",
            (AVATAR_URL,),
        )
        print(f"Avatar updated: {cur.rowcount} row(s)")

        # 2. Template A: Daily Motivational Inspiration Email
        spec_a = {
            "version": "2",
            "triggers": [
                {
                    "id": "schedule_trigger",
                    "key": "schedule.cron",
                    "mode": "polling",
                    "event_schema": {},
                    "meta": {"requires_connection": False},
                    "filters": {},
                    "provider_config": {
                        "cron": "0 7 * * *",
                        "timezone": "${config.timezone}",
                    },
                    "ui_meta": {},
                }
            ],
            "nodes": [
                {
                    "id": "generate",
                    "type": "agent",
                    "inputs": {
                        "model": "claude-sonnet-4-20250514",
                        "prompt": PROMPT_A,
                        "tools": [],
                    },
                    "outputs": {"mode": "text"},
                },
                {
                    "id": "send_email",
                    "type": "tool",
                    "tool": "gmail.send_email",
                    "inputs": {
                        "to": "${config.recipient_email}",
                        "subject": "Your Daily Dose of Inspiration",
                        "body": "${nodes.generate.output}",
                    },
                },
            ],
            "edges": [
                {
                    "source": "schedule_trigger",
                    "target": "generate",
                    "type": "trigger",
                },
                {
                    "source": "generate",
                    "target": "send_email",
                    "type": "default",
                },
            ],
        }

        integrations_a = [
            {
                "provider": "google",
                "integration_type": "gmail",
                "reason": "Send daily email",
            }
        ]

        cur.execute(
            INSERT_SQL,
            (
                "daily-motivational-inspiration-email",
                "Daily Motivational Inspiration Email",
                DESC_A,
                "productivity",
                json.dumps(["motivation", "email", "personal-development"]),
                "community",
                81,
                json.dumps(spec_a),
                json.dumps(integrations_a),
                True,
            ),
        )
        print(f"Template A inserted: {cur.rowcount} row(s)")

        # 3. Template B: Application Analyzer
        spec_b = {
            "version": "2",
            "triggers": [
                {
                    "id": "webhook_trigger",
                    "key": "webhook.incoming",
                    "mode": "webhook",
                    "event_schema": {},
                    "meta": {"requires_connection": False},
                    "filters": {},
                    "provider_config": {
                        "path": "/analyze-application",
                    },
                    "ui_meta": {},
                }
            ],
            "nodes": [
                {
                    "id": "extract_pdf",
                    "type": "tool",
                    "tool": "pdf.extract_text",
                    "inputs": {
                        "file": "${trigger.body.file_url}",
                    },
                },
                {
                    "id": "analyze",
                    "type": "agent",
                    "inputs": {
                        "model": "claude-sonnet-4-20250514",
                        "prompt": PROMPT_B,
                        "tools": [],
                    },
                    "outputs": {"mode": "text"},
                },
                {
                    "id": "send_results",
                    "type": "tool",
                    "tool": "gmail.send_email",
                    "inputs": {
                        "to": "${config.reviewer_email}",
                        "subject": (
                            "Application Analysis: "
                            "${nodes.analyze.output.applicant_name}"
                        ),
                        "body": "${nodes.analyze.output}",
                    },
                },
            ],
            "edges": [
                {
                    "source": "webhook_trigger",
                    "target": "extract_pdf",
                    "type": "trigger",
                },
                {
                    "source": "extract_pdf",
                    "target": "analyze",
                    "type": "default",
                },
                {
                    "source": "analyze",
                    "target": "send_results",
                    "type": "default",
                },
            ],
        }

        integrations_b = [
            {
                "provider": "google",
                "integration_type": "gmail",
                "reason": "Send analysis results",
            }
        ]

        cur.execute(
            INSERT_SQL,
            (
                "application-analyzer",
                "Application Analyzer",
                DESC_B,
                "productivity",
                json.dumps(
                    ["document-analysis", "pdf", "compliance", "hr"]
                ),
                "community",
                81,
                json.dumps(spec_b),
                json.dumps(integrations_b),
                True,
            ),
        )
        print(f"Template B inserted: {cur.rowcount} row(s)")

        # 4. Template C: LinkedIn SNEAC GEO
        spec_c = {
            "version": "2",
            "triggers": [
                {
                    "id": "gdrive_trigger",
                    "key": "google_drive.file_created",
                    "mode": "polling",
                    "event_schema": {},
                    "meta": {"requires_connection": True},
                    "filters": {},
                    "provider_config": {
                        "folder_id": "${config.google_drive_folder_id}",
                    },
                    "ui_meta": {},
                }
            ],
            "nodes": [
                {
                    "id": "read_doc",
                    "type": "tool",
                    "tool": "google_drive.read_file",
                    "inputs": {
                        "file_id": "${trigger.file_id}",
                    },
                },
                {
                    "id": "sneac_agent",
                    "type": "agent",
                    "inputs": {
                        "model": "claude-sonnet-4-20250514",
                        "prompt": PROMPT_C,
                        "tools": [],
                    },
                    "outputs": {"mode": "text"},
                },
                {
                    "id": "approval_email",
                    "type": "tool",
                    "tool": "gmail.send_email",
                    "inputs": {
                        "to": "${config.approval_email}",
                        "subject": "LinkedIn Post Ready for Review",
                        "body": APPROVAL_BODY,
                    },
                },
                {
                    "id": "publish",
                    "type": "tool",
                    "tool": "linkedin.create_post",
                    "inputs": {
                        "content": "${nodes.sneac_agent.output}",
                    },
                },
            ],
            "edges": [
                {
                    "source": "gdrive_trigger",
                    "target": "read_doc",
                    "type": "trigger",
                },
                {
                    "source": "read_doc",
                    "target": "sneac_agent",
                    "type": "default",
                },
                {
                    "source": "sneac_agent",
                    "target": "approval_email",
                    "type": "default",
                },
                {
                    "source": "approval_email",
                    "target": "publish",
                    "type": "default",
                },
            ],
        }

        integrations_c = [
            {
                "provider": "google",
                "integration_type": "google_drive",
                "reason": "Monitor Google Docs",
            },
            {
                "provider": "google",
                "integration_type": "gmail",
                "reason": "Approval email",
            },
            {
                "provider": "linkedin",
                "integration_type": "linkedin",
                "reason": "Publish posts",
            },
        ]

        cur.execute(
            INSERT_SQL,
            (
                "linkedin-sneac-geo",
                "LinkedIn SNEAC GEO",
                DESC_C,
                "productivity",
                json.dumps(
                    ["linkedin", "content-creation", "no-code", "automation"]
                ),
                "community",
                81,
                json.dumps(spec_c),
                json.dumps(integrations_c),
                True,
            ),
        )
        print(f"Template C inserted: {cur.rowcount} row(s)")

        conn.commit()
        print("\nAll changes committed successfully.")

        # Verify
        cur.execute(
            "SELECT slug, name, is_published "
            "FROM workflow_templates WHERE created_by_id = 81"
        )
        rows = cur.fetchall()
        print("\nVerification - templates for user 81:")
        for row in rows:
            print(f"  {row[0]}: {row[1]} (published={row[2]})")

        cur.execute(
            "SELECT avatar_url FROM user_profiles WHERE user_id = 81"
        )
        row = cur.fetchone()
        print(f"\nAvatar URL: {row[0][:80]}...")

    except Exception as exc:
        conn.rollback()
        print(f"Error: {exc}")
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
