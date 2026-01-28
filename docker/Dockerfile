FROM ubuntu:24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir git+https://github.com/gyorilab/indra_agent.git@deploy

# Run the MCP server
CMD ["gunicorn", "indra_agent.mcp_server.server:app", "--bind", "0.0.0.0:8778", \
     "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker"]
