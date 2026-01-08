---
sidebar_position: 1
---

# Railway Deployment Guide

This guide walks you through creating a Railway template for one-click Seer deployment. Railway templates are created through their UI, not declarative config files.

## Overview

The Seer Railway deployment consists of 4 services:
1. **PostgreSQL** - Database (Railway-managed)
2. **Redis** - Task queue and cache (Railway-managed)
3. **seer-api** - FastAPI backend server (custom)
4. **seer-worker** - Background task worker (custom)

## Step 1: Initial Manual Deployment

### 1.1 Create New Railway Project

1. Go to [Railway](https://railway.app)
2. Sign in to your account
3. Click "New Project"
4. Select "Empty Project"
5. Name it "seer-template" (you can change this later)

### 1.2 Add PostgreSQL Service

1. Click "+ New" in your project
2. Select "Database" → "PostgreSQL"
3. Railway will automatically provision the database
4. Note: `DATABASE_URL` is automatically created as a reference variable

### 1.3 Add Redis Service

1. Click "+ New" in your project
2. Select "Database" → "Redis"
3. Railway will automatically provision Redis
4. Note: `REDIS_URL` is automatically created as a reference variable

### 1.4 Add API Service

1. Click "+ New" in your project
2. Select "GitHub Repo"
3. Connect your Seer repository
4. **Service Settings:**
   - **Name:** `seer-api`
   - **Root Directory:** `/` (leave blank)
   - **Build Command:** (leave default, uses Dockerfile)
   - **Start Command:** (managed by railway.toml)

5. **Environment Variables:**
   Click "Variables" tab and add:
   ```
   DATABASE_URL=${{Postgres.DATABASE_URL}}
   REDIS_URL=${{Redis.REDIS_URL}}
   PORT=8000
   AUTO_APPLY_DATABASE_MIGRATIONS=true
   OPENAI_API_KEY=[your-key-for-testing]
   ```

6. **Service Configuration:**
   - Go to "Settings" tab
   - Under "Railway Config File Path", enter: `railway.toml`
   - Enable "Public Networking"
   - Click "Generate Domain" to create a public URL

7. **Deploy:**
   - Click "Deploy" or wait for automatic deployment
   - Monitor build logs to ensure success

### 1.5 Add Worker Service

1. Click "+ New" in your project
2. Select "GitHub Repo"
3. Select the same Seer repository
4. **Service Settings:**
   - **Name:** `seer-worker`
   - **Root Directory:** `/` (leave blank)
   - **Build Command:** (leave default, uses Dockerfile)
   - **Start Command:** (managed by railway.worker.toml)

5. **Environment Variables:**
   Click "Variables" tab and add:
   ```
   DATABASE_URL=${{Postgres.DATABASE_URL}}
   REDIS_URL=${{Redis.REDIS_URL}}
   AUTO_APPLY_DATABASE_MIGRATIONS=false
   OPENAI_API_KEY=${{seer-api.OPENAI_API_KEY}}
   ```
   Note: Worker references API's OPENAI_API_KEY to avoid duplication

6. **Service Configuration:**
   - Go to "Settings" tab
   - Under "Railway Config File Path", enter: `railway.worker.toml`
   - **Do NOT** enable "Public Networking" (worker is private)

7. **Deploy:**
   - Click "Deploy" or wait for automatic deployment
   - Monitor build logs to ensure success

## Step 2: Test the Deployment

### 2.1 Verify Services Are Running

Check that all 4 services show "Active" status:
- ✅ Postgres (green)
- ✅ Redis (green)
- ✅ seer-api (green)
- ✅ seer-worker (green)

### 2.2 Test the API

1. Click on the `seer-api` service
2. Copy the generated public URL (e.g., `https://seer-production-xxxx.up.railway.app`)
3. Test the health endpoint:
   ```bash
   curl https://your-railway-url.up.railway.app/health
   ```
   Expected response: `{"status":"ok"}` or similar

### 2.3 Verify Database Migrations

1. Click on `seer-api` service
2. Check the deployment logs
3. Look for "Running database migrations..." message
4. Ensure migrations completed without errors

### 2.4 Test Workflow Creation

1. Visit your Railway URL in a browser
2. Try creating a simple workflow
3. Verify the workflow executes successfully

## Step 3: Convert to Template

### 3.1 Create Template

1. In your Railway project, click the project name dropdown (top left)
2. Select "Project Settings"
3. Scroll to "Template" section
4. Click "Create Template from Project"

### 3.2 Configure Template Metadata

**Template Name:**
```
Seer - AI Workflow Builder
```

**Template Description:**
```
Deploy Seer, an AI-powered workflow automation platform with visual builder,
integrated tools (Google Workspace, GitHub, Supabase), and fine-grained control.

Includes:
- FastAPI backend with workflow execution engine
- Background task worker for triggers and polling
- PostgreSQL database for persistence
- Redis for task queuing

Perfect for self-hosting workflow automation with enterprise-grade features.
```

**Template README:**
```markdown
# Seer Deployment

This template deploys a complete Seer instance with all required services.

## What's Included

- **API Server**: FastAPI backend with workflow builder
- **Worker**: Background task processor for triggers and workflow execution
- **PostgreSQL**: Workflow and user data storage
- **Redis**: Task queue and caching

## Required Configuration

You'll need to provide an OpenAI API key during deployment:

- `OPENAI_API_KEY`: Get from [OpenAI Platform](https://platform.openai.com/api-keys)

## Post-Deployment Setup

1. Access your Seer instance at the generated Railway URL
2. Add optional integrations in Settings → Variables:
   - `TAVILY_API_KEY` - Web search capabilities
   - `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` - Google Workspace
   - `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` - GitHub integration
   - `ANTHROPIC_API_KEY` - Alternative LLM provider

## Estimated Cost

- Free tier: $5/month credit (good for testing)
- Production: $15-30/month
- Includes PostgreSQL backups, SSL certificates, DDoS protection

## Documentation

- [Seer GitHub](https://github.com/your-org/seer)
- [Documentation](https://docs.your-domain.com)
- [Support](https://github.com/your-org/seer/issues)
```

### 3.3 Configure Template Variables

Define which environment variables users must provide:

**Required Variables:**
| Variable | Description | Default Value |
|----------|-------------|---------------|
| `OPENAI_API_KEY` | OpenAI API key for workflow execution and AI assistance | (none - user must provide) |

**Note:** All other variables (DATABASE_URL, REDIS_URL, etc.) are automatically configured via service references.

### 3.4 Set Template Icon and Tags

1. **Icon:** Upload Seer logo or choose from Railway's icon library
2. **Tags:** Add relevant tags:
   - `automation`
   - `ai`
   - `workflow`
   - `python`
   - `fastapi`

### 3.5 Publish Template

1. Review all template settings
2. Click "Publish Template"
3. Copy the template URL (format: `https://railway.app/template/[template-id]`)

## Step 4: Update Seer Repository

### 4.1 Add Deploy Button to README

Edit the main README.md file and add:

```markdown
### Deploy to Railway

Deploy Seer to Railway with one click:

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/[YOUR-TEMPLATE-ID])
```

Replace `[YOUR-TEMPLATE-ID]` with the actual template ID from Step 3.5.

### 4.2 Update Documentation

Add a "Deployment" section to your docs with:
- Link to this Railway deployment guide
- Cost estimates
- Configuration instructions
- Troubleshooting tips

## Step 5: Test Template Deployment

### 5.1 Test from Another Account

1. Log out of Railway (or use incognito mode)
2. Sign in with a different Railway account
3. Click the "Deploy on Railway" button
4. Verify:
   - All 4 services are created
   - Environment variable prompt appears for OPENAI_API_KEY
   - Deployment completes successfully
   - Public URL is accessible
   - Health check passes

### 5.2 Troubleshooting Common Issues

**Build fails:**
- Check Dockerfile syntax
- Verify all dependencies in pyproject.toml
- Review build logs for specific errors

**Database connection fails:**
- Ensure DATABASE_URL reference is correct: `${{Postgres.DATABASE_URL}}`
- Check that Postgres service is healthy before API starts
- Verify migrations ran successfully

**Worker not starting:**
- Check railway.worker.toml configuration
- Verify REDIS_URL is correctly referenced
- Review worker logs for errors

**Health check fails:**
- Confirm /health endpoint exists in api/main.py
- Check PORT environment variable is set correctly
- Verify uvicorn is binding to 0.0.0.0, not 127.0.0.1

## Maintenance

### Updating the Template

When you update the Seer codebase:

1. Push changes to the repository
2. Railway will auto-deploy to your template project
3. Test the changes in your template project
4. If everything works, the template automatically uses the latest code
5. Users deploying from the template will get the latest version

### Template Versioning

Railway templates always deploy from the latest commit on the default branch. To maintain stability:

1. Use a `main` or `production` branch for stable releases
2. Test new features in separate branches
3. Only merge to main when ready for production

## Support

If you encounter issues:

1. Check Railway's [documentation](https://docs.railway.app)
2. Review [Seer's GitHub issues](https://github.com/your-org/seer/issues)
3. Contact Railway support through their Discord or help portal
4. Open an issue in the Seer repository with the `deployment` label

## Cost Optimization Tips

1. **Start small**: Use Railway's free tier ($5/month credit) for testing
2. **Scale gradually**: Upgrade services only when needed
3. **Monitor usage**: Check Railway's usage dashboard regularly
4. **Optimize resources**:
   - Use smaller PostgreSQL instance for development
   - Reduce worker replicas if not needed
   - Enable hibernation for non-production instances

## Security Best Practices

1. **Rotate API keys regularly** in Railway's environment variables
2. **Use Railway's secret management** (variables are encrypted at rest)
3. **Enable 2FA** on your Railway account
4. **Limit public access**: Only seer-api should have public networking enabled
5. **Monitor logs**: Regularly review deployment and application logs for anomalies
6. **Keep dependencies updated**: Regularly update Python packages and base images
