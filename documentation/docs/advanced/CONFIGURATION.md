# Configuration Management with AWS Parameter Store

## Overview

Seer uses a hierarchical configuration system built on Pydantic Settings with AWS Parameter Store integration. This system provides a flexible, type-safe way to manage configuration across different environments (local development, staging, production) without requiring AWS credentials for local development.

## Priority Order

Configuration values are loaded in the following priority (highest to lowest):

1. **Environment Variables** - Explicitly set in your shell or container
2. **.env File** - Local development configuration file
3. **AWS Parameter Store** - Production secrets and configuration
4. **Default Values** - Fallback defaults defined in code

This means:
- Setting an environment variable always wins
- `.env` file overrides AWS Parameter Store and defaults
- AWS Parameter Store is used only if not found in env vars or `.env`
- Default values are used as the final fallback

## How It Works

### Custom Settings Source

The implementation uses Pydantic's `settings_customise_sources` feature with a custom `AwsSsmSettingsSource` class that:

1. **Gracefully handles missing AWS credentials** - If boto3 cannot initialize (no credentials, no AWS CLI configured), it silently fails and allows fallback to defaults
2. **Fetches parameters in bulk** - Uses `get_parameters_by_path` to efficiently load all parameters at once
3. **Uses environment-based paths** - Parameters are organized by environment: `/{ENV}/{parameter_name}`

### Parameter Store Structure

Parameters in AWS SSM Parameter Store should follow this naming convention:

```
/{environment}/{parameter_name}
```

For example:
- `/dev/openai_api_key`
- `/prod/openai_api_key`
- `/staging/DATABASE_URL`
- `/prod/clerk_jwks_url`

The `{environment}` is determined by the `ENV` environment variable (defaults to `dev`).

### Type Conversion

Pydantic automatically handles type conversion from string values (as stored in Parameter Store) to the correct Python types:

- `"true"` or `"false"` → `bool`
- `"123"` → `int`
- `"45.67"` → `float`
- JSON strings → `dict` or `list`

## Usage Examples

### Local Development (No AWS)

Create a `.env` file in the project root:

```env
ENV=dev
OPENAI_API_KEY=sk-proj-xxx
DATABASE_URL=postgresql://localhost/seer_dev
AUTO_OPEN_BROWSER=true
DEFAULT_LLM_MODEL=gpt-4o-mini
```

The app will use these values. AWS Parameter Store is gracefully skipped if credentials aren't available.

### Local Development with AWS

If you have AWS credentials configured (via `aws configure` or environment variables), you can use Parameter Store:

1. Store secrets in Parameter Store:
```bash
aws ssm put-parameter \
  --name "/dev/openai_api_key" \
  --value "sk-proj-xxx" \
  --type "SecureString" \
  --region us-east-1
```

2. Run the app (with AWS credentials configured):
```bash
export ENV=dev
uv run main.py
```

The app will fetch values from Parameter Store for any configuration not in `.env` or environment variables.

### Production Deployment

1. Store all production secrets in Parameter Store:

```bash
# Store secrets
aws ssm put-parameter --name "/prod/openai_api_key" --value "sk-xxx" --type "SecureString"
aws ssm put-parameter --name "/prod/anthropic_api_key" --value "sk-ant-xxx" --type "SecureString"
aws ssm put-parameter --name "/prod/DATABASE_URL" --value "postgresql://..." --type "SecureString"
aws ssm put-parameter --name "/prod/clerk_jwks_url" --value "https://..." --type "String"

# Store configuration flags
aws ssm put-parameter --name "/prod/postgres_write_requires_approval" --value "false" --type "String"
aws ssm put-parameter --name "/prod/auto_open_browser" --value "false" --type "String"
aws ssm put-parameter --name "/prod/trigger_poller_enabled" --value "true" --type "String"
```

2. Deploy with environment variable:

```bash
export ENV=prod
export AWS_REGION=us-east-1
uv run main.py
```

### Override in Production

Even in production with Parameter Store, you can override specific values via environment variables:

```bash
# Use Parameter Store for most config, but override LLM model
export ENV=prod
export DEFAULT_LLM_MODEL=gpt-4o
uv run main.py
```

## Configuration Reference

### Environment Detection

- `ENV` - Environment name (dev, staging, prod). Used for Parameter Store paths. Default: `dev`

### Secrets (Typically from Parameter Store in Production)

- `openai_api_key` - OpenAI API key
- `anthropic_api_key` - Anthropic/Claude API key
- `DATABASE_URL` - PostgreSQL connection string
- `clerk_jwks_url` - Clerk JWKS URL for JWT verification
- `clerk_issuer` - Clerk JWT issuer
- `clerk_audience` - Clerk JWT audience
- `clerk_secret_key` - Clerk secret key for metadata updates
- `stripe_secret_key` - Stripe API secret key
- `stripe_webhook_secret` - Stripe webhook signing secret
- `slack_bot_token` - Slack bot OAuth token
- `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` - GitHub OAuth credentials
- `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` - Google OAuth credentials
- `supabase_client_id` / `supabase_client_secret` - Supabase management credentials

### Feature Flags (Safe to Have Defaults)

- `postgres_write_requires_approval` - Require approval for DB writes. Default: `true`
- `auto_open_browser` - Auto-open browser on startup (self-hosted). Default: `true`
- `trigger_poller_enabled` - Enable workflow trigger polling. Default: `true`
- `request_profiling_enabled` - Enable request profiling. Default: `false`
- `MLFLOW_ENABLED` - Enable MLflow logging. Default: `false`
- `slack_notifications_enabled` - Enable Slack error notifications. Default: `false`

### Other Configuration

- `seer_mode` - Deployment mode: 'self-hosted' or 'cloud'. Default: `self-hosted`
- `default_llm_model` - Default LLM model. Default: `gpt-4o-mini`
- `redis_url` - Redis/Valkey connection string. Default: `redis://localhost:6379/0`
- `FRONTEND_URL` - Frontend application URL. Default: `http://localhost:5173`
- `nexus_max_agent_steps` - Max agent steps for Nexus. Default: `75`
- `trigger_poller_interval_seconds` - Trigger polling interval. Default: `5`

## Testing Configuration Priority

Run the demo script to see which source is being used for each configuration value:

```bash
uv run examples/config_priority_demo.py
```

This will show:
- 🔵 Environment Variable
- 🟢 .env File
- 🟡 AWS Parameter Store
- ⚪ Default Value

For each configuration field.

## AWS IAM Permissions

The application needs the following IAM permissions to read from Parameter Store:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ssm:GetParameter",
        "ssm:GetParameters",
        "ssm:GetParametersByPath"
      ],
      "Resource": [
        "arn:aws:ssm:*:*:parameter/dev/*",
        "arn:aws:ssm:*:*:parameter/prod/*",
        "arn:aws:ssm:*:*:parameter/staging/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "kms:Decrypt"
      ],
      "Resource": [
        "arn:aws:kms:*:*:key/*"
      ],
      "Condition": {
        "StringEquals": {
          "kms:ViaService": "ssm.us-east-1.amazonaws.com"
        }
      }
    }
  ]
}
```

## Best Practices

### For Local Development

1. **Use `.env` file for development** - Keep secrets out of environment variables
2. **Never commit `.env`** - Already in `.gitignore`
3. **Use default values** - Provide sensible defaults for non-sensitive config
4. **Document required secrets** - Make it clear what needs to be configured

### For Production

1. **Use Parameter Store for all secrets** - Never hardcode or use `.env` in production
2. **Use SecureString type** - For sensitive values (API keys, passwords)
3. **Environment-based paths** - Separate dev/staging/prod parameters
4. **Principle of least privilege** - Only grant necessary IAM permissions
5. **Regular rotation** - Rotate secrets periodically
6. **Audit access** - Monitor Parameter Store access via CloudTrail

### Naming Conventions

1. **Lowercase with underscores** - Match Python field names: `openai_api_key`
2. **Environment prefix** - Always use `/{env}/` prefix: `/prod/openai_api_key`
3. **Descriptive names** - Make it clear what the parameter is for
4. **Consistent casing** - Follow Python naming conventions

## Troubleshooting

### "No AWS credentials found"

This is normal for local development. The system gracefully falls back to `.env` and defaults.

### "Parameter not found"

Check:
1. Parameter name matches field name exactly (case-sensitive in SSM)
2. Environment prefix is correct (`/dev/`, `/prod/`, etc.)
3. Region is correct (set `AWS_REGION` environment variable)
4. IAM permissions are configured

### Values not updating

Remember the priority order. If a value is set in environment variables or `.env`, it will override Parameter Store. To use Parameter Store value, remove from `.env` and environment.

### Type conversion errors

Parameter Store stores everything as strings. Pydantic handles conversion, but ensure:
- Booleans: use `"true"` or `"false"` (lowercase)
- Numbers: use plain digits `"123"`
- JSON: use valid JSON syntax

## Migration from Old System

The old `get_param()` function has been deprecated and replaced with the new priority system. If you have existing code using `get_param()`:

**Old:**
```python
openai_api_key: Optional[str] = Field(
    default=get_param("openai_api_key"),
    description="..."
)
```

**New:**
```python
openai_api_key: Optional[str] = Field(
    default=None,  # or a sensible default
    description="..."
)
```

The new system automatically handles AWS Parameter Store via `settings_customise_sources`.

## Related Files

- `src/seer/config.py` - Main configuration class
- `src/seer/utilities/aws/parameter_store.py` - AWS SSM integration
- `examples/config_priority_demo.py` - Demo script showing priority
- `.env.example` - Example environment file (create this for your team)
