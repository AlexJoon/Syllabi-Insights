# Syllabi Insights Agent

A RAG-powered AI agent built with [Scale Agentex](https://github.com/scaleapi/scale-agentex) for extracting insights from academic syllabi documents.

## Features

- **Semantic Search**: Query your syllabi using natural language
- **OpenAI Vector Store**: Documents stored and indexed via OpenAI's hosted infrastructure
- **GPT-4 Responses**: Intelligent answers with source citations
- **Streaming**: Real-time response streaming
- **Agentex UI**: Built-in web interface for testing and debugging

## Prerequisites

Install these before running:

```bash
# Docker (required for Agentex backend)
brew install docker docker-compose

# Node.js (required for Agentex UI)
brew install node

# uv - Python package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Agentex CLI
uv tool install agentex-sdk
```

## Setup

### 1. Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
```

### 2. Create Vector Store & Upload Syllabi

1. Go to: https://platform.openai.com/storage/vector_stores
2. Click **Create** to make a new vector store
3. **Upload your syllabi files** (PDF, DOCX, TXT)
4. Copy the vector store ID (starts with `vs_`)
5. Add it to `.env`:
   ```
   OPENAI_VECTOR_STORE_ID=vs_your-id-here
   ```

### 3. Start Everything

```bash
./start.sh
```

This single command:
- Stops conflicting local redis
- Starts Agentex backend (Docker)
- Starts Agentex UI (localhost:3000)
- Starts your agent

### 4. Open the UI

Go to **http://localhost:3000** and select "syllabi-insights-agent"

## Usage

### Example Questions

- "What files are in the vector store?"
- "What are the prerequisites for CS 101?"
- "When is the midterm exam?"
- "Compare the grading policies across courses"
- "What textbooks are required?"

### Available Tools

The agent has access to these tools:

| Tool | Description |
|------|-------------|
| `search_syllabi` | Semantic search across uploaded syllabi |
| `upload_syllabus` | Upload a new file to the vector store |
| `get_index_stats` | Check vector store statistics |
| `list_indexed_files` | List all indexed files |

## Project Structure

```
syllabi-insights-agent/
├── start.sh                # One-command startup script
├── manifest.yaml           # Agentex agent configuration
├── .env                    # Your API keys (not committed)
├── project/
│   ├── acp.py              # Agent entry point (streaming + tools)
│   └── lib/
│       ├── vectorstore.py  # OpenAI Vector Store integration
│       └── tools.py        # Tool definitions for GPT-4
└── infrastructure/         # Agentex backend + UI (cloned)
    ├── agentex/            # Backend services (Docker)
    └── agentex-ui/         # Web interface (React)
```

## Manual Startup (Alternative)

If you prefer to run services separately:

**Terminal 1 - Backend:**
```bash
cd infrastructure/agentex
uv venv && source .venv/bin/activate && uv sync
make dev
```

**Terminal 2 - UI:**
```bash
cd infrastructure/agentex-ui
npm install
npm run dev
```

**Terminal 3 - Agent:**
```bash
uv venv && source .venv/bin/activate && uv sync
agentex agents run --manifest manifest.yaml
```

## Troubleshooting

### Redis conflict
```bash
brew services stop redis
```

### Docker not running
Make sure Docker Desktop is open and running.

### Services not healthy
Wait 30-60 seconds after starting. Use `lazydocker` to monitor container health:
```bash
brew install lazydocker
lazydocker
```

### Agent not appearing in UI
Check that all Docker containers are healthy and the agent terminal shows no errors.

## Resources

- [Scale Agentex](https://github.com/scaleapi/scale-agentex)
- [Agentex Python SDK](https://github.com/scaleapi/scale-agentex-python)
- [OpenAI Vector Stores](https://platform.openai.com/docs/assistants/tools/file-search)

## License

MIT
