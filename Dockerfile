# Syllabi Insights Agent - Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv for fast package management
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies
RUN uv sync --no-dev

# Copy project code
COPY project/ ./project/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default port for ACP server
EXPOSE 8000

# Run the ACP server
CMD ["uv", "run", "uvicorn", "project.acp:app", "--host", "0.0.0.0", "--port", "8000"]
