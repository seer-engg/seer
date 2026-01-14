---
sidebar_position: 1
---

# Supabase Integration

## Multi-Credential Integration Setup

Supabase support adds persisted resources plus non-OAuth secrets:

1. Connect your Supabase management account via OAuth (`supabase_mgmt` provider).
2. Browse projects with the existing resource picker (`supabase_project` type) and call `POST /integrations/supabase/projects/bind` to persist the selection.
3. The binding stores project metadata in `integration_resources` and the anon/service-role API keys in `integration_secrets`.
4. Workflow nodes reference the binding via `integration_resource_id`, and the credential resolver injects the REST URL + service key automatically (used by `supabase_table_query`).

## API Endpoints

- `GET /integrations/supabase/resources/bindings` – list persisted projects for the signed-in user.
- `GET /integrations/resources/{resource_id}/secrets` – view secret fingerprints + metadata linked to a resource.
- `DELETE /integrations/resources/{resource_id}` – revoke a resource binding and deactivate attached secrets.

## Related Documentation
- [Getting Started](/)
- [Configuration Reference](../advanced/CONFIGURATION)
