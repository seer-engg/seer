from __future__ import annotations

from typing import Literal, List, Dict, Any
from pydantic import BaseModel, Field


# Supported artifact types
ArtifactType = Literal[
    "Lesson",
    "AntiPattern",
    "Pattern",
    "FailureSignature",
    "TestCaseStub",
]


class Artifact(BaseModel):
    """
    A compact, human-like memory artifact capturing a reusable rule or pattern.
    The primary retrieval surface is the short `rule` sentence. Tags add recall hooks.
    """
    type: ArtifactType
    rule: str = Field(min_length=6, max_length=256)
    tags: List[str] = Field(default_factory=list, description="Short tokens like 'python', 'edge-cases', 'input-validation'")
    snippet: str | None = Field(default=None, description="Optional tiny snippet (<= 10 lines) if crucial")


class ArtifactList(BaseModel):
    artifacts: List[Artifact] = Field(default_factory=list)


def normalize_tags(tags: List[str]) -> List[str]:
    normalized: List[str] = []
    for t in tags or []:
        token = (t or "").strip().lower().replace(" ", "-")
        if token and token not in normalized:
            normalized.append(token)
    return normalized


def artifact_to_assistant_content(artifact: Artifact) -> str:
    """
    Render artifact as a compact assistant content string optimized for semantic recall.
    The first line is the RULE sentence to maximize vector emphasis.
    Subsequent lines carry type, tags, and an optional snippet for human readability.
    """
    tags_line = ", ".join(normalize_tags(artifact.tags))
    lines = [
        f"RULE: {artifact.rule}",
        f"TYPE: {artifact.type}",
        f"TAGS: {tags_line}" if tags_line else "TAGS:",
    ]
    if artifact.snippet:
        snippet = artifact.snippet.strip()
        # Hard limit to avoid storing large code
        snippet_lines = snippet.splitlines()[:10]
        lines.append("SNIPPET:\n" + "\n".join(snippet_lines))
    return "\n".join(lines)


def build_mem0_messages_for_artifact(artifact: Artifact, user_message: str) -> List[Dict[str, str]]:
    """
    Build a minimal Mem0 messages payload: a short user task followed by the artifact.
    We keep the user message compact to provide retrieval context, but the RULE is primary.
    """
    compact_user = user_message.strip()
    if len(compact_user) > 300:
        compact_user = compact_user[:297] + "..."
    return [
        {"role": "user", "content": f"Task: {compact_user}"},
        {"role": "assistant", "content": artifact_to_assistant_content(artifact)},
    ]


def parse_artifact_from_content(content: str) -> Dict[str, Any]:
    """
    Best-effort parser to reconstruct an artifact-like dict from stored content.
    Used during retrieval to extract RULE/TYPE/TAGS if present. Falls back gracefully.
    """
    result: Dict[str, Any] = {"rule": content.strip()}
    try:
        lines = [l.strip() for l in content.splitlines()]
        for i, line in enumerate(lines):
            if line.startswith("RULE:"):
                result["rule"] = line[len("RULE:"):].strip()
            elif line.startswith("TYPE:"):
                result["type"] = line[len("TYPE:"):].strip()
            elif line.startswith("TAGS:"):
                raw = line[len("TAGS:"):].strip()
                tags = [t.strip() for t in raw.split(",") if t.strip()]
                result["tags"] = normalize_tags(tags)
            elif line.startswith("SNIPPET:"):
                snippet = "\n".join(lines[i + 1:])
                result["snippet"] = snippet
                break
    except Exception:
        # Leave best-effort fields
        pass
    return result


