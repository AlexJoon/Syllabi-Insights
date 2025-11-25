#!/bin/bash

# Syllabi Insights Agent - Complete Startup Script
# This script starts all required services for the Agentex-based RAG agent

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFRA_DIR="$PROJECT_DIR/infrastructure"

# Required for Agentex local development
export ENVIRONMENT=development

echo "=========================================="
echo "  Syllabi Insights Agent - Startup"
echo "=========================================="

# Check prerequisites
check_prerequisites() {
    echo ""
    echo "[1/5] Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed. Install with: brew install docker"
        exit 1
    fi

    if ! command -v node &> /dev/null; then
        echo "❌ Node.js is not installed. Install with: brew install node"
        exit 1
    fi

    if ! command -v uv &> /dev/null; then
        echo "❌ uv is not installed. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi

    if [ ! -f "$PROJECT_DIR/.env" ]; then
        echo "❌ .env file not found. Copy .env.example to .env and add your API keys."
        exit 1
    fi

    # Check for required env vars
    source "$PROJECT_DIR/.env"
    if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" == "sk-your-openai-api-key-here" ]; then
        echo "❌ OPENAI_API_KEY not set in .env"
        exit 1
    fi

    echo "✅ Prerequisites OK"
}

# Stop any existing redis that might conflict
stop_redis() {
    echo ""
    echo "[2/5] Stopping local redis (if running)..."
    brew services stop redis 2>/dev/null || true
    echo "✅ Redis stopped"
}

# Start Agentex backend services
start_backend() {
    echo ""
    echo "[3/5] Starting Agentex backend services..."
    cd "$INFRA_DIR/agentex"

    # Setup Python environment if needed
    if [ ! -d ".venv" ]; then
        echo "    Setting up Python environment..."
        uv venv
        source .venv/bin/activate
        uv sync
    else
        source .venv/bin/activate
    fi

    # Start docker services using make dev (runs in background via docker-compose)
    echo "    Starting Docker containers via 'make dev'..."
    make dev &
    BACKEND_PID=$!

    # Wait for services to be ready
    echo "    Waiting for services to start..."
    sleep 10

    echo "✅ Backend services starting (wait ~30s for healthy status)"
}

# Start Agentex UI
start_ui() {
    echo ""
    echo "[4/5] Starting Agentex UI..."
    cd "$INFRA_DIR/agentex-ui"

    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        echo "    Installing UI dependencies..."
        npm install
    fi

    # Start UI in background
    echo "    Starting UI server..."
    npm run dev &
    UI_PID=$!
    echo "    UI PID: $UI_PID"

    echo "✅ UI starting at http://localhost:3000"
}

# Start the agent
start_agent() {
    echo ""
    echo "[5/5] Starting Syllabi Insights Agent..."
    cd "$PROJECT_DIR"

    # Setup Python environment if needed
    if [ ! -d ".venv" ]; then
        echo "    Setting up Python environment..."
        uv venv
    fi

    source .venv/bin/activate
    uv sync

    # Copy .env to where agent can find it
    cp "$PROJECT_DIR/.env" "$PROJECT_DIR/project/.env" 2>/dev/null || true

    echo ""
    echo "=========================================="
    echo "  All services starting!"
    echo "=========================================="
    echo ""
    echo "  📊 Agentex UI:    http://localhost:3000"
    echo "  🤖 Agent:         syllabi-insights-agent"
    echo ""
    echo "  Wait ~30-60 seconds for all services to be healthy."
    echo "  Then open http://localhost:3000 in your browser."
    echo ""
    echo "  Press Ctrl+C to stop all services."
    echo "=========================================="
    echo ""

    # Start agent (this blocks)
    agentex agents run --manifest manifest.yaml
}

# Cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down services..."
    cd "$INFRA_DIR/agentex"
    make dev-stop 2>/dev/null || docker-compose down 2>/dev/null || true
    kill $UI_PID 2>/dev/null || true
    kill $BACKEND_PID 2>/dev/null || true
    echo "Done."
}

trap cleanup EXIT

# Run all steps
check_prerequisites
stop_redis
start_backend
start_ui
start_agent
