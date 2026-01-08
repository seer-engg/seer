---
sidebar_position: 3
---

# Workflow Proposals

## Overview
The workflow chat agent can generate complete, compiler-ready workflow specifications.

## How It Works
- The workflow chat agent calls `submit_workflow_spec` to emit complete, compiler-ready JSON specs (no more incremental patch ops).
- Agent prompts include the trimmed WorkflowSpec schema plus a canonical example from `workflow_compiler/schema`.

## Proposal APIs
- Proposal APIs (`/api/agents/workflow/.../proposals`) return the captured spec so clients can preview or apply it directly.
- Accepting a proposal replaces the workflow definition atomically with the validated spec; rejecting leaves the workflow untouched.

## Use Cases
- AI-assisted workflow creation
- Rapid prototyping of complex workflows
- Learning workflow schema structure
