## Seer

[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/seer-engg/seer/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/seer-engg/seer?style=social)](https://github.com/seer-engg/seer/stargazers)
[![Documentation](https://img.shields.io/badge/docs-docs.getseer.dev-blue)](https://docs.getseer.dev)
[![Discord](https://img.shields.io/badge/discord-join-7289DA?logo=discord&logoColor=white)](https://discord.gg/NuYsDdhJ)
[![Twitter Follow](https://img.shields.io/twitter/follow/get_seer?style=social)](https://x.com/get_seer)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-get--seer-0077B5?logo=linkedin)](https://www.linkedin.com/company/get-seer)

Seer is a **open-source workflow builder with fine-grained control** for creating and executing automated workflows with integrated tools and services.

## Quick Start

### Using Github

```bash
git clone https://github.com/seer-engg/seer
cd seer
docker compose up
```

That's it! Starts Docker services (Postgres, Redis, backend, worker), streams logs, and waits for readiness.

### Deploy to Railway

Deploy Seer to Railway with one click:

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/[TEMPLATE-ID])

**What gets deployed:** FastAPI backend, background worker, PostgreSQL, and Redis

**Setup:** Click button, enter `OPENAI_API_KEY`, wait 5-7 minutes. Estimated cost: $15-30/month.

For detailed deployment instructions, see the [Railway Deployment Guide](https://docs.getseer.dev/deployment/RAILWAY).

### Using the Workflow Editor

After running `docker compose up`, the workflow editor is available at:
- **Frontend**: http://localhost:5173/workflows?backend=http://localhost:8000
- **Backend API**: http://localhost:8000

### Configuration

Create a `.env` file:

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional integrations (add as needed)
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
TAVILY_API_KEY=...
```

Docker automatically configures `DATABASE_URL` and `REDIS_URL`.

For complete configuration options, see the [Configuration Reference](https://docs.getseer.dev/advanced/CONFIGURATION).

### Usage

**Start development environment:**
```bash
docker compose up
```

**View logs:**
```bash
docker compose logs -f
```

**Stop services:**
```bash
docker compose down
```

For complete configuration options, see the [Configuration Reference](https://docs.getseer.dev/advanced/CONFIGURATION).

### Key Features

**🛠️ Visual Workflow Builder**
- Drag-and-drop interface for creating automation workflows
- Node-based editor with custom blocks and integrations
- Real-time workflow validation and execution

**🤖 AI-Assisted Development**
- Chat interface for workflow design and debugging
- AI suggestions for workflow improvements
- Intelligent error handling and recovery

**🔗 Rich Integrations**
- **Google Workspace**: Gmail, Drive, Sheets with OAuth
- **GitHub**: Repository management, issues, PRs
- **Web Tools**: Search, content fetching, APIs
- **Databases**: PostgreSQL with approval-based write controls

**🔒 Enterprise-Ready**
- Self-hosted or cloud deployment options
- OAuth-based authentication (Clerk integration)
- Role-based access control
- Audit trails and execution history

### Documentation

📚 **[Complete Documentation](https://docs.getseer.dev)** - Full docs site with guides, API reference, and examples

- [Quick Start](#quick-start) - Get running in 60 seconds
- [Railway Deployment](https://docs.getseer.dev/deployment/RAILWAY) - Production deployment guide
- [Worker Setup](./worker/README.md) - Background task worker configuration
- [Integrations](https://docs.getseer.dev/integrations/SUPABASE) - Google, GitHub, Supabase setup
- [Advanced Features](https://docs.getseer.dev/advanced/TRIGGERS) - Triggers, proposals, and more
- [Configuration Reference](https://docs.getseer.dev/advanced/CONFIGURATION) - Complete configuration options

### License

Seer is open source under the MIT license. Enterprise features (if any exist)
reside in the `ee/` directory and are licensed separately. See LICENSE for details.
