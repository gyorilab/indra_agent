#!/bin/bash

GILDA_VERSION=$(python -c "import gilda; print(gilda.__version__)")
export GILDA_TERMS="/root/.data/gilda/${GILDA_VERSION}/grounding_terms.db"

gunicorn indra_agent.mcp_server.server:app --bind 0.0.0.0:8778 -w 4 --worker-class uvicorn.workers.UvicornWorker
