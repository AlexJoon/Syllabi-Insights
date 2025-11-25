# Syllabi Insights Agent

A RAG-powered AI agent built with [Scale Agentex](https://github.com/scaleapi/scale-agentex) for extracting insights from academic syllabi documents.

This agent uses:
- **OpenAI GPT-4** for intelligent question answering with tool calling
- **OpenAI Embeddings** (`text-embedding-3-small`) for semantic search
- **ChromaDB** for vector storage and persistence
- **Scale Agentex** for agent infrastructure, UI, and deployment

## What Agentex Provides

Agentex is **not** just another SDK wrapper. It provides:

1. **Agent Server** - Backend runtime with health monitoring via Docker
2. **Developer UI** - Web interface at `localhost:3000` to test and debug your agent
3. **Streaming Support** - Real-time response streaming out of the box
4. **Traces & Observability** - Investigate agent behavior in the UI
5. **Scalable Architecture** - Progress from L1 (chatbot) to L5 (autonomous) agents

## Prerequisites

```bash
# Install Python 3.12+
# https://www.python.org/downloads/

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Docker and Node.js
brew install docker docker-compose node

# Stop redis (conflicts with Agentex's docker-compose)
brew services stop redis

# Install the Agentex CLI globally
uv tool install agentex-sdk
```

Verify installation:
```bash
agentex -h
```

## Project Structure

```
syllabi-insights-agent/
├── manifest.yaml           # Agentex agent configuration
├── pyproject.toml          # Python dependencies
├── .env                    # Your API keys (create from .env.example)
├── project/
│   ├── acp.py              # Main agent entry point (streaming + tools)
│   └── lib/
│       ├── vectorstore.py  # OpenAI embeddings + ChromaDB
│       └── tools.py        # RAG tools for GPT-4 function calling
└── data/
    └── vectorstore/        # Persisted vector database (auto-created)
```

## Quick Start

### 1. Clone the Agentex Repository

You need the Agentex backend server running locally:

```bash
# Clone Scale Agentex
git clone https://github.com/scaleapi/scale-agentex.git
cd scale-agentex
```

### 2. Start the Agentex Backend (Terminal 1)

```bash
cd agentex/
uv venv && source .venv/bin/activate && uv sync
make dev
```

This starts the Agent Server via Docker. Use `lazydocker` in another terminal to monitor health.

### 3. Start the Developer UI (Terminal 2)

```bash
cd agentex-ui/
make install
make dev
```

The UI will be available at `http://localhost:3000`.

### 4. Setup Your Agent (Terminal 3)

```bash
cd syllabi-insights-agent/

# Create virtual environment
uv venv && source .venv/bin/activate && uv sync

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 5. Run Your Agent

```bash
agentex agents run --manifest manifest.yaml
```

Your agent will appear in the Developer UI at `http://localhost:3000`.

## Using the Agent

### In the Developer UI

1. Open `http://localhost:3000`
2. Select "syllabi-insights-agent"
3. Start chatting!

### Example Conversations

**Index a syllabus:**
> "Index the syllabus at /path/to/cs101_syllabus.pdf"

**Search for information:**
> "What are the prerequisites for CS 101?"
> "When is the midterm exam?"
> "What textbooks are required?"

**Compare courses:**
> "Which courses cover machine learning?"
> "Compare the grading policies between CS 101 and CS 201"

**Get structured insights:**
> "Give me a summary of the Data Structures course"

## Agent Tools

The agent has access to these tools (via OpenAI function calling):

| Tool | Description |
|------|-------------|
| `search_syllabi` | Semantic search across indexed syllabi |
| `index_syllabus` | Add a new PDF/DOCX/TXT syllabus to the index |
| `get_course_insights` | Get structured info about a specific course |
| `get_index_stats` | Check how many documents are indexed |

## Configuration

### Environment Variables

Create a `.env` file at the project root:

```bash
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional
VECTORSTORE_PATH=./data/vectorstore
```

### manifest.yaml

The manifest tells Agentex how to run your agent:

```yaml
name: syllabi-insights-agent
version: "1.0.0"
description: "RAG-powered agent for academic syllabi"

project:
  path: project
  entry_point: acp.py
```

## Next Steps

Once you have the basic agent running, you can:

1. **Add more tools** - Extend `lib/tools.py` with new capabilities
2. **Make it agentic** - Switch to the async ACP pattern for complex workflows
3. **Add sub-agents** - Use Agentex's Agent Developer Kit (ADK) for multi-agent systems
4. **Go async with Temporal** - For long-running, durable workflows

See the [Agentex Python SDK tutorials](https://github.com/scaleapi/scale-agentex-python) for more advanced examples.

## Resources

- [Scale Agentex GitHub](https://github.com/scaleapi/scale-agentex) - Main framework repo
- [Agentex Python SDK](https://github.com/scaleapi/scale-agentex-python) - SDK with tutorials
- [Agentex Blog Post](https://scale.com/blog/agentex) - Introduction and architecture
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings) - Embedding models

## License

MIT
