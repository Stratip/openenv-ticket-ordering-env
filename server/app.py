# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Ticket Ordering Environment.

This module creates an HTTP server that exposes the TicketOrderingEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from models import TicketOrderingAction, TicketOrderingObservation
    from server.ticket_ordering_environment import TicketOrderingEnvironment
except ModuleNotFoundError:
    from ..models import TicketOrderingAction, TicketOrderingObservation
    from .ticket_ordering_environment import TicketOrderingEnvironment


# Create the app with web interface and README integration
app = create_app(
    TicketOrderingEnvironment,
    TicketOrderingAction,
    TicketOrderingObservation,
    env_name="ticket_ordering",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


def main():
    """
    Entry point for openenv and direct execution.
    """
    import argparse
    import uvicorn

    # Move argument parsing inside main or handle defaults
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    
    # parse_known_args prevents errors if openenv passes unexpected flags
    args, _ = parser.parse_known_args()

    # 'app' must be defined globally in this file (e.g., app = FastAPI())
    uvicorn.run("server.app:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()
