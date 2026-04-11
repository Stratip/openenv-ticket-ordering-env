---
title: Ticket Ordering Environment Server
emoji: 🎟️
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Ticket Ordering Environment

**Ticket Ordering** is an environment for turning a messy “who shouts loudest” backlog into a **repeatable, auditable ordering** of work items (GitHub issues, Jira tickets, support requests, bugs, etc.) using an **objective scoring rubric**.

It’s especially useful when **QA testers**, **engineering**, and **third-party stakeholders** (customers, partners, vendors) have **contradictory priorities** depending on the situation:

- QA may optimize for *risk reduction* and *release confidence*.
- Third parties may optimize for *their own timelines*, *contracts*, and *visibility*.
- Engineering may optimize for *system health*, *tech debt*, and *delivery cost*.

When incentives diverge, “priority” becomes a negotiation. This project makes priority **measurable** by asking an agent (human or model) to assign scores *relative to a shared criteria string* and provides feedback via a reward function—so you can iterate toward a ranking that’s **consistent**, not political.

## Why this matters

Teams usually fail at prioritization for one of two reasons:

- **Subjectivity**: two people read the same ticket and give different priorities for valid reasons.
- **Context drift**: the “right” priority changes with release phase, customer escalations, regressions, or risk posture.

An “objective scorer” doesn’t remove judgment—it **structures it**. You get:

- **Comparable decisions**: scores are on the same scale across tickets.
- **Explainability**: the criteria and summaries capture the *why*, not just the number.
- **Repeatability**: rerun ordering when constraints change (release week vs normal week) and measure deltas.
- **Less conflict**: disagreements become “which criterion weights are wrong?” rather than “your priority is wrong.”

## What you get

- **A FastAPI + WebSocket server** exposing the environment
- **A Python client** (`TicketOrderingEnv`) with Docker and direct-URL connection options
- **A web UI** (when deployed) for interacting with the environment
- **A training/evaluation loop** primitive: order tickets, get reward, improve the scorer

## Table of contents

- [Quick Start](#quick-start)
- [Building the Docker Image](#building-the-docker-image)
- [Deploying to Hugging Face Spaces](#deploying-to-hugging-face-spaces)
- [Environment Details](#environment-details)
- [Advanced Usage](#advanced-usage)
- [Development & Testing](#development--testing)
- [Project Structure](#project-structure)

## Quick Start

The simplest way to use the Ticket Ordering environment is through the `TicketOrderingEnv` class:

```python
from ticket_ordering import TicketOrderingAction, TicketOrderingEnv
import random

try:
    # Create environment from Docker image
    ticket_orderingenv = TicketOrderingEnv.from_docker_image("ticket_ordering-env:latest")

    # Reset
    result = ticket_orderingenv.reset()
    obs = result.observation
    print(f"Reset: {obs}")

    result = ticket_orderingenv.step(
        TicketOrderingAction(
            candidate_priority=50,
            candidate_summary="issue",
            next_reference_ids=[],
            # Pick a ticket ID to evaluate next (example strategy)
            next_candidate_id=random.choice(list(obs.ticket_heuristics.keys())),
            end_ordering=False,
        )
    )
    print(f"Step observation: {result.observation}")
    print(f"Step reward: {result.reward}")

finally:
    # Always clean up
    ticket_orderingenv.close()
```

That's it! The `TicketOrderingEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t ticket_ordering-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action
**TicketOrderingAction**: Contains the following fields:
- `candidate_priority` (int) - Assigned priority score for the candidate ticket, relative to other tickets.
- `candidate_summary` (str) - Short summary describing the candidate ticket, capturing key context for future comparisons.
- `next_reference_ids` (list[int]) - IDs of tickets to request as references in the next step, used to guide further comparisons.
- `next_candidate_id` (int) - ID of the next ticket to evaluate as the candidate in the following step.
- `end_ordering` (bool) - Whether the agent thinks it has finished ordering all tickets and wants to terminate the episode.

### Observation
**TicketOrderingObservation**: Contains the following:
- `reward` (float) - Reward based on last priority assignment and end ordering decision.
- `done` (bool) - True if `end_ordering` is set to true in the action or the episode has reached the maximum step count.
- `metadata` (dict) - Additional info.

- `ordering_criteria` (str) - Criteria by which tickets should be ordered.
- `reference_tickets` (list[Ticket]) - Subset of tickets provided as references to help evaluate and compare the current candidate ticket.
- `candidate_ticket` (Ticket) - The ticket currently being evaluated and assigned a heuristic by the agent.
- `ticket_heuristics` (dict[int, TicketHeuristic]) - Known heuristics for previously evaluated tickets, keyed by ticket ID, used for relative comparison.
- `total_tickets` (int) - Total number of tickets in the environment that need to be ordered.
- `completed_iterations` (int) - Number of ordering steps completed so far in the current episode.


### Reward
The reward is calculated as: `(post_action_optimality - pre_action_optimality) - (1.5 - current_step / max_steps) * action_end_ordering`

## Advanced Usage

### Connecting to an Existing Server

If you already have a Ticket Ordering environment server running, you can connect directly:

```python
from ticket_ordering import TicketOrderingEnv

# Connect to existing server
ticket_orderingenv = TicketOrderingEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = ticket_orderingenv.reset()
result = ticket_orderingenv.step(...)
```

Note: When connecting to an existing server, `ticket_orderingenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from ticket_ordering import TicketOrderingAction, TicketOrderingEnv

# Connect with context manager (auto-connects and closes)
with TicketOrderingEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    result = env.step(...)
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    TicketOrderingEnvironment,  # Pass class, not instance
    TicketOrderingAction,
    TicketOrderingObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously.


## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/ticket_ordering_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
ticket_ordering/
├── client.py              # Client for interacting with the Ticket Ordering environment
├── docker-build.sh        # Script to build the Docker image for the project
├── .dockerignore          # Files and directories excluded from Docker builds
├── inference.py           # Main inference logic for running ticket ordering or model predictions
├── __init__.py            # Package initialization and exports
├── load-env-vars.sh       # Script to load environment variables (e.g., API keys, configs)
├── models.py              # Pydantic models (Ticket, Observation, Action, etc.)
├── openenv.yaml           # OpenEnv configuration / environment manifest
├── problem_generator.py   # Generates synthetic ticket ordering problems for testing/training
├── pyproject.toml         # Project metadata and dependency definitions
├── README.md              # Project documentation and usage instructions
├── server/                # Server-side implementation
│   ├── app.py             # FastAPI app exposing HTTP/WebSocket endpoints
│   ├── Dockerfile         # Docker image definition for the server
│   ├── __init__.py        # Server module exports
│   ├── requirements.txt   # Python dependencies for the server environment
│   └── ticket_ordering_environment.py  # Core environment logic and state transitions
├── uv.lock                # Locked dependency versions (generated by uv)
└── validation-script.sh   # Script to validate environment behavior or submission correctness
```
