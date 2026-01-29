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
    pip install --no-cache-dir git+https://github.com/gyorilab/indra_agent.git

# Setup resources
RUN python -m protmapper.resources && \
    python -m indra.ontology.bio build && \
    python -c "from indra.ontology.bio.sqlite_ontology import build_sqlite_ontology; build_sqlite_ontology()" && \
    python -m adeft.download && \
    python -m gilda.resources && \
    python -m gilda.resources.sqlite_adapter

RUN mkdir /app
WORKDIR /app

COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

# Run the MCP server
CMD ["/app/startup.sh"]
